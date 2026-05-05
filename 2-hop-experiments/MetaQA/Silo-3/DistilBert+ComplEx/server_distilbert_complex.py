# src/server_distilbert_complex.py — Federated server for DistilBERT + ComplEx
#
# Differences from server_complex.py (Bert+ComplEx):
#   - DistilBertTokenizer + DistilBertModel instead of BertTokenizer + BertModel
#   - Imports from config_distilbert_complex
#
# Differences from server_distilbert_distmult.py (DistilBert+DistMult):
#   - ComplEx entity embeddings are 2*embed_dim (complex [re|im] representation)
#   - ENTITY_DIM = 2d = 512 per silo  (not d = 256)
#   - JOINT_DIM  = 6d = 1536          (not 3d = 768)
#   - MLP final output layer → 1536   (not 768)
#
# Identical to server_distilbert_rotate.py in structure — only the config import
# differs (ComplEx vs RotatE both use 2d entity representations).
#
# Everything else — fusion, cosine scoring, margin ranking loss,
# topic anchoring, gradient splitting — is identical across all experiments.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertModel

from config_distilbert_complex import (
    BERT_MODEL, MLP_HIDDEN_DIMS, MLP_DROPOUT,
    KGE_EMBED_DIM, USE_TOPIC_ANCHORING, DISTILBERT_DIM,
    ENTITY_DIM, JOINT_DIM
)


class QuestionEncoder(nn.Module):
    """
    DistilBERT [CLS] → MLP → q_embed ∈ ℝ^(6d) = ℝ^1536

    ComplEx stores entity embeddings as 2d-dim real vectors
    (real + imaginary parts of the complex embedding):
        h(e) ∈ ℝ^(2d) = ℝ^512  per silo

    Therefore the joint embedding is:
        h_joint = [h_A || h_B || h_C] ∈ ℝ^(3 × 2d) = ℝ^1536

    The MLP must output 1536 to match: 768 → 512 → 1536.

    DistilBERT specs (same as DistilBert+RotatE):
      - 6 transformer layers   (BERT-base has 12)
      - ~66M parameters        (BERT-base has ~110M)
      - 768-dim [CLS] output   (same as BERT-base → same MLP input dim)
      - ~60% faster, ~40% smaller, retains ~97% of BERT NLP performance

    DistilBERT is frozen; only the MLP is trained during federated QA.
    """

    def __init__(self):
        super().__init__()
        self.tokenizer  = DistilBertTokenizer.from_pretrained(BERT_MODEL)
        self.distilbert = DistilBertModel.from_pretrained(BERT_MODEL)

        # Freeze DistilBERT — only MLP head trains
        for param in self.distilbert.parameters():
            param.requires_grad = False

        # MLP: 768 → 512 → 1536  (output = JOINT_DIM = 6d)
        layers, in_dim = [], DISTILBERT_DIM   # 768
        for h_dim in MLP_HIDDEN_DIMS:         # [768, 512]
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
            # DistilBERT: last_hidden_state[:,0,:] = [CLS] representation
            # (no pooler_output unlike BERT)
            out = self.distilbert(**enc)
        cls = out.last_hidden_state[:, 0, :]    # (B, 768)
        return self.mlp(cls)                    # (B, 1536)


class FedVServer(nn.Module):
    """
    Federated server for DistilBERT + ComplEx experiment (3 silos, 2-hop).

    ComplEx entity embeddings are 2d-dimensional (complex [re|im] representation):
        h(e) ∈ ℝ^(2d) = ℝ^512  per silo

    Joint embedding after federated fusion:
        h_joint(e) = [h_A(e) || h_B(e) || h_C(e)] ∈ ℝ^(6d) = ℝ^1536

    This matches DistilBert+RotatE in joint embedding dimension (both 6d=1536).
    The KGE scoring function differs (Hermitian dot product vs complex rotation),
    but the server-side architecture is identical.

    ComplEx advantage over RotatE: both entity and relation embeddings are
    fully complex (unconstrained), allowing richer asymmetric pattern modelling.
    ComplEx advantage over DistMult: handles anti-symmetric relations (e.g.
    directed_by ≠ directed) via the imaginary component, critical for OWL
    inverseOf axioms in the enriched KB.

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

        sim(e) = cos(q_final, h_joint(e))
               = (q_final · h_joint(e)) / (||q_final|| · ||h_joint(e)||)

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

        e⁺ = highest-scoring correct answer in candidate set
        e⁻ = highest-scoring incorrect candidate (hardest negative)
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

        h_a, h_b, h_c : (N, 2d) = (N, 512)  — ComplEx entity embeddings [re|im]
        h_joint        : (N, 6d) = (N, 1536) — after fusion
        q_embed        : (B, 6d) = (B, 1536) — from DistilBERT + MLP

        Returns:
            loss : scalar margin ranking loss
            sim  : (B, K) cosine similarities over candidate set
        """
        h_joint = self.fuse(h_a, h_b, h_c)                         # (N, 1536)
        q_embed = self.question_encoder(questions, device)          # (B, 1536)

        if USE_TOPIC_ANCHORING:
            # q_final = q_embed + h_joint[topic]
            # Grounds the question at the topic entity in joint embedding space
            q_final = q_embed + h_joint[topic_ids]
        else:
            q_final = q_embed

        candidate_ids = candidate_ids.to(device)
        sim  = self.score_candidates(q_final, h_joint, candidate_ids)
        loss = self.ranking_loss(sim, answer_ids_batch, candidate_ids, margin)
        return loss, sim
