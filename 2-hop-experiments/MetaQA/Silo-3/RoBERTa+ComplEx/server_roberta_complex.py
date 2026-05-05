# src/server_roberta_complex.py — Federated server for RoBERTa + ComplEx
#
# Differences from server_roberta_distmult.py (RoBERTa+DistMult):
#   - ComplEx entity embeddings are 2*embed_dim (complex [re|im] representation)
#   - ENTITY_DIM = 2d = 512 per silo  (not d = 256)
#   - JOINT_DIM  = 6d = 1536          (not 3d = 768)
#   - MLP final output layer → 1536   (not 768)
#   - Imports from config_roberta_complex
#
# Differences from server_distilbert_complex.py (DistilBert+ComplEx):
#   - RobertaTokenizer + RobertaModel instead of DistilBertTokenizer + DistilBertModel
#   - RoBERTa: 12 layers, ~125M params  vs  DistilBERT: 6 layers, ~66M params
#   - Both output 768-dim → same JOINT_DIM=1536, same MLP architecture
#
# Everything else — fusion, cosine scoring, margin ranking loss,
# topic anchoring, gradient splitting — is identical across all experiments.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel

from config_roberta_complex import (
    BERT_MODEL, MLP_HIDDEN_DIMS, MLP_DROPOUT,
    KGE_EMBED_DIM, USE_TOPIC_ANCHORING, ROBERTA_DIM,
    ENTITY_DIM, JOINT_DIM
)


class QuestionEncoder(nn.Module):
    """
    RoBERTa <s> token → MLP → q_embed ∈ ℝ^(6d) = ℝ^1536

    ComplEx stores entity embeddings as 2d-dim real vectors [re|im]:
        h(e) ∈ ℝ^(2d) = ℝ^512  per silo

    Joint embedding:
        h_joint = [h_A || h_B || h_C] ∈ ℝ^(3 × 2d) = ℝ^1536

    The MLP must output 1536 to match: 768 → 512 → 1536.

    RoBERTa-base properties:
      - 12 transformer layers    (same as BERT-base, more than DistilBERT's 6)
      - ~125M parameters         (BERT-base: ~110M, DistilBERT: ~66M)
      - 768-dim hidden output    (identical to BERT-base → same MLP input dim)
      - Trained on 160GB text, larger batches (8K), dynamic masking, no NSP
      - BPE tokenizer — no token_type_ids
      - last_hidden_state[:, 0, :] = <s> token representation (≡ BERT [CLS])
      - Generally outperforms BERT-base on downstream NLU tasks

    RoBERTa is frozen; only the MLP projection head is trained.
    """

    def __init__(self):
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(BERT_MODEL)
        self.roberta   = RobertaModel.from_pretrained(BERT_MODEL)

        # Freeze RoBERTa — only MLP head trains during federated QA
        for param in self.roberta.parameters():
            param.requires_grad = False

        # MLP: 768 → 512 → 1536  (output = JOINT_DIM = 6d)
        layers, in_dim = [], ROBERTA_DIM   # 768
        for h_dim in MLP_HIDDEN_DIMS:     # [768, 512]
            layers += [nn.Linear(in_dim, h_dim), nn.ReLU(),
                       nn.Dropout(MLP_DROPOUT)]
            in_dim = h_dim
        layers += [nn.Linear(in_dim, JOINT_DIM)]   # → 1536
        self.mlp = nn.Sequential(*layers)

    def forward(self, questions, device):
        """
        Args:
            questions : list[str]  — raw question strings
            device    : torch.device
        Returns:
            q_embed   : (B, 6d) = (B, 1536)
        """
        enc = self.tokenizer(
            questions, return_tensors="pt", padding=True,
            truncation=True, max_length=64
        ).to(device)
        with torch.no_grad():
            # RoBERTa: last_hidden_state[:, 0, :] = <s> token representation
            out = self.roberta(**enc)
        cls = out.last_hidden_state[:, 0, :]    # (B, 768)
        return self.mlp(cls)                    # (B, 1536)


class FedVServer(nn.Module):
    """
    Federated server for RoBERTa + ComplEx experiment (3 silos, 2-hop).

    ComplEx entity embeddings are 2d-dimensional (complex [re|im] representation):
        h(e) ∈ ℝ^(2d) = ℝ^512  per silo

    Joint embedding after federated fusion:
        h_joint(e) = [h_A(e) || h_B(e) || h_C(e)] ∈ ℝ^(6d) = ℝ^1536

    This matches Bert+ComplEx and DistilBert+ComplEx in joint embedding
    dimension. The only difference from DistilBert+ComplEx is the encoder:
    RoBERTa (125M, 12 layers) instead of DistilBERT (66M, 6 layers).

    ComplEx scoring (for reference — happens in silos during Phase 1):
        φ(h, r, t) = Re(<h, r, conj(t)>)
                   = Σ h_re·r_re·t_re + h_re·r_im·t_im
                       + h_im·r_re·t_im − h_im·r_im·t_re

    Gradient splitting at concat boundary:
        ∂L/∂h_joint → [∂L/∂h_A || ∂L/∂h_B || ∂L/∂h_C]
        each of size (N, 512) — sent to respective silo.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim        = embed_dim
        self.question_encoder = QuestionEncoder()

    # ── Fusion ────────────────────────────────────────────────────────────────

    def fuse(self, h_a, h_b, h_c):
        """
        Concatenate per-silo ComplEx entity embeddings.
        h_a, h_b, h_c : (N, 2d) = (N, 512)
        returns        : (N, 6d) = (N, 1536)
        """
        return torch.cat([h_a, h_b, h_c], dim=-1)

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score_candidates(self, q_embed, h_joint, candidate_ids):
        """
        Cosine similarity between question vector and candidate entities.

        q_embed       : (B, 6d)
        h_joint       : (N, 6d)
        candidate_ids : (B, K)   — -1 = padding
        returns sim   : (B, K)
        """
        safe_ids = candidate_ids.clamp(min=0)
        h_cands  = h_joint[safe_ids]                                # (B, K, 6d)
        q_norm   = F.normalize(q_embed, p=2, dim=-1).unsqueeze(1)  # (B, 1, 6d)
        h_norm   = F.normalize(h_cands, p=2, dim=-1)               # (B, K, 6d)
        return (q_norm * h_norm).sum(dim=-1)                        # (B, K)

    # ── Loss ──────────────────────────────────────────────────────────────────

    def ranking_loss(self, sim, answer_ids_batch, candidate_ids, margin=1.0):
        """
        Margin ranking loss with hardest negative mining:
            L = max(0, γ + sim(e⁻) − sim(e⁺))
        """
        device = sim.device
        losses = []

        for i, answer_ids in enumerate(answer_ids_batch):
            cands      = candidate_ids[i]
            valid_mask = cands >= 0
            cand_list  = cands[valid_mask].tolist()
            scores     = sim[i][valid_mask]

            answer_set  = set(answer_ids)
            pos_indices = [j for j, c in enumerate(cand_list)
                           if c in answer_set]
            if not pos_indices:
                continue

            best_pos = scores[pos_indices].max()

            neg_mask = torch.ones(len(cand_list), dtype=torch.bool, device=device)
            for j in pos_indices:
                neg_mask[j] = False
            if neg_mask.sum() == 0:
                continue

            hard_neg = scores[neg_mask].max()
            losses.append(F.relu(margin + hard_neg - best_pos))

        if not losses:
            return torch.tensor(0.0, requires_grad=True, device=device)
        return torch.stack(losses).mean()

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, questions, topic_ids, h_a, h_b, h_c,
                answer_ids_batch, candidate_ids, device, margin=1.0):
        """
        Full forward pass for one training batch.

        h_a, h_b, h_c : (N, 2d) = (N, 512) — ComplEx entity embeddings [re|im]
        h_joint        : (N, 6d) = (N, 1536) — after fusion
        q_embed        : (B, 6d) = (B, 1536) — from RoBERTa + MLP
        """
        h_joint = self.fuse(h_a, h_b, h_c)                         # (N, 1536)
        q_embed = self.question_encoder(questions, device)          # (B, 1536)

        if USE_TOPIC_ANCHORING:
            q_final = q_embed + h_joint[topic_ids]
        else:
            q_final = q_embed

        candidate_ids = candidate_ids.to(device)
        sim  = self.score_candidates(q_final, h_joint, candidate_ids)
        loss = self.ranking_loss(sim, answer_ids_batch, candidate_ids, margin)
        return loss, sim
