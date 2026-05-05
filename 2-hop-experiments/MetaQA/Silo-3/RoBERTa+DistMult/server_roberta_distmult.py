# src/server_roberta_distmult.py — Federated server for RoBERTa + DistMult
#
# Differences from server_roberta_transe.py (RoBERTa+TransE):
#   - Imports from config_roberta_distmult (different CHECKPOINT_DIR)
#   - DistMult entity embeddings are d-dim real (same as TransE) →
#     h_joint, MLP output, and JOINT_DIM are all unchanged at 768
#
# Differences from server_distilbert_distmult.py (DistilBert+DistMult):
#   - RobertaTokenizer + RobertaModel instead of DistilBertTokenizer + DistilBertModel
#   - RoBERTa: 12 layers, ~125M params  vs  DistilBERT: 6 layers, ~66M params
#   - Both output 768-dim → identical downstream architecture
#
# Everything else — fusion, cosine scoring, margin ranking loss,
# topic anchoring, gradient splitting — is identical across all experiments.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel

from config_roberta_distmult import (
    BERT_MODEL, MLP_HIDDEN_DIMS, MLP_DROPOUT,
    KGE_EMBED_DIM, USE_TOPIC_ANCHORING, ROBERTA_DIM
)

JOINT_DIM = 3 * KGE_EMBED_DIM   # 3 × 256 = 768


class QuestionEncoder(nn.Module):
    """
    RoBERTa <s> token → MLP → q_embed ∈ ℝ^(3d) = ℝ^768

    RoBERTa-base properties:
      - 12 transformer layers    (same depth as BERT-base)
      - ~125M parameters         (BERT-base: ~110M, DistilBERT: ~66M)
      - 768-dim hidden output    (identical to BERT-base)
      - Trained on 160GB text with larger batches and dynamic masking
      - BPE tokenizer, no token_type_ids, <s> token as sequence representation
      - Typically outperforms BERT-base on downstream NLU tasks

    RoBERTa is frozen; only the MLP projection head is trained.
    The output dimension 3d = 768 matches h_joint = [h_A || h_B || h_C],
    where each silo contributes a d=256 dim DistMult entity embedding.
    """

    def __init__(self):
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(BERT_MODEL)
        self.roberta   = RobertaModel.from_pretrained(BERT_MODEL)

        # Freeze RoBERTa — only MLP head trains during federated QA
        for param in self.roberta.parameters():
            param.requires_grad = False

        # MLP: 768 → 512 → 768  (identical to all other 3d-joint experiments)
        layers, in_dim = [], ROBERTA_DIM   # 768
        for h_dim in MLP_HIDDEN_DIMS:
            layers += [nn.Linear(in_dim, h_dim), nn.ReLU(),
                       nn.Dropout(MLP_DROPOUT)]
            in_dim = h_dim
        layers += [nn.Linear(in_dim, JOINT_DIM)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, questions, device):
        """
        Args:
            questions : list[str]  — raw question strings
            device    : torch.device
        Returns:
            q_embed   : (B, 3d) = (B, 768)
        """
        enc = self.tokenizer(
            questions, return_tensors="pt", padding=True,
            truncation=True, max_length=64
        ).to(device)
        with torch.no_grad():
            # RoBERTa: last_hidden_state[:, 0, :] = <s> token representation
            out = self.roberta(**enc)
        cls = out.last_hidden_state[:, 0, :]    # (B, 768)
        return self.mlp(cls)                    # (B, 3d)


class FedVServer(nn.Module):
    """
    Federated server for RoBERTa + DistMult experiment (3 silos, 2-hop).

    DistMult entity embeddings are d-dim real vectors (same as TransE):
        h(e) ∈ ℝᵈ = ℝ^256  per silo

    Joint embedding after federated fusion:
        h_joint(e) = [h_A(e) || h_B(e) || h_C(e)] ∈ ℝ^(3d) = ℝ^768

    This is the same joint space as RoBERTa+TransE, BERT+DistMult, and
    DistilBERT+DistMult. The KGE scoring function differs (bilinear diagonal
    vs translation), but the server-side architecture is identical.

    Gradient splitting at concat boundary:
        ∂L/∂h_joint → [∂L/∂h_A || ∂L/∂h_B || ∂L/∂h_C]
        each of size (N, 256) — sent to respective silo.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim        = embed_dim
        self.question_encoder = QuestionEncoder()

    # ── Fusion ────────────────────────────────────────────────────────────────

    def fuse(self, h_a, h_b, h_c):
        """
        Concatenate per-silo DistMult entity embeddings.
        h_a, h_b, h_c : (N, d) = (N, 256)
        returns        : (N, 3d) = (N, 768)
        """
        return torch.cat([h_a, h_b, h_c], dim=-1)

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score_candidates(self, q_embed, h_joint, candidate_ids):
        """
        Cosine similarity between question vector and candidate entities.

        q_embed       : (B, 3d)
        h_joint       : (N, 3d)
        candidate_ids : (B, K)   — -1 = padding
        returns sim   : (B, K)
        """
        safe_ids = candidate_ids.clamp(min=0)
        h_cands  = h_joint[safe_ids]                                # (B, K, 3d)
        q_norm   = F.normalize(q_embed, p=2, dim=-1).unsqueeze(1)  # (B, 1, 3d)
        h_norm   = F.normalize(h_cands, p=2, dim=-1)               # (B, K, 3d)
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

        h_a, h_b, h_c : (N, d)  = (N, 256) — DistMult entity embeddings
        h_joint        : (N, 3d) = (N, 768) — after fusion
        q_embed        : (B, 3d) = (B, 768) — from RoBERTa + MLP
        """
        h_joint = self.fuse(h_a, h_b, h_c)
        q_embed = self.question_encoder(questions, device)

        if USE_TOPIC_ANCHORING:
            q_final = q_embed + h_joint[topic_ids]
        else:
            q_final = q_embed

        candidate_ids = candidate_ids.to(device)
        sim  = self.score_candidates(q_final, h_joint, candidate_ids)
        loss = self.ranking_loss(sim, answer_ids_batch, candidate_ids, margin)
        return loss, sim
