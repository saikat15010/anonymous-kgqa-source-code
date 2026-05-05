# src/server_distilbert_distmult.py — Federated server for DistilBERT + DistMult
#
# Differences from server.py (Bert+TransE):
#   - DistilBertTokenizer + DistilBertModel instead of BertTokenizer + BertModel
#   - Imports from config_distilbert_distmult
#
# DistMult entity embeddings are d-dim real vectors (same as TransE),
# so h_joint = [h_A || h_B || h_C] ∈ ℝ^(3d) = ℝ^768  — identical to
# Bert+TransE and Bert+DistMult servers.
# Fusion, cosine scoring, margin ranking loss, topic anchoring — all unchanged.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertModel

from config_distilbert_distmult import (
    BERT_MODEL, MLP_HIDDEN_DIMS, MLP_DROPOUT,
    KGE_EMBED_DIM, USE_TOPIC_ANCHORING, DISTILBERT_DIM
)

JOINT_DIM = 3 * KGE_EMBED_DIM   # 3 × 256 = 768


class QuestionEncoder(nn.Module):
    """
    DistilBERT [CLS] → MLP → q_embed ∈ ℝ^(3d)

    DistilBERT vs BERT-base:
      - 6 transformer layers   (BERT has 12)
      - ~66M parameters        (BERT has ~110M)
      - 768-dim hidden output  (identical to BERT-base)
      - ~60% faster, ~40% smaller, retains ~97% of BERT performance

    DistilBERT is frozen; only the MLP projection head is trained.
    The output dimension 3d = 768 matches h_joint = [h_A || h_B || h_C],
    keeping the architecture fully compatible with all other experiments.
    """

    def __init__(self):
        super().__init__()
        self.tokenizer  = DistilBertTokenizer.from_pretrained(BERT_MODEL)
        self.distilbert = DistilBertModel.from_pretrained(BERT_MODEL)

        # Freeze DistilBERT — only the MLP head trains during federated QA
        for param in self.distilbert.parameters():
            param.requires_grad = False

        # MLP: 768 → 512 → 768  (matches BERT+DistMult architecture exactly)
        layers, in_dim = [], DISTILBERT_DIM
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
            q_embed   : (B, 3d)
        """
        enc = self.tokenizer(
            questions, return_tensors="pt", padding=True,
            truncation=True, max_length=64
        ).to(device)
        with torch.no_grad():
            # DistilBERT returns BaseModelOutput with last_hidden_state
            # (no pooler_output unlike BERT — we take [CLS] = index 0)
            out = self.distilbert(**enc)
        cls = out.last_hidden_state[:, 0, :]    # (B, 768)
        return self.mlp(cls)                    # (B, 3d)


class FedVServer(nn.Module):
    """
    Federated server for DistilBERT + DistMult experiment (3 silos, 2-hop).

    h_joint(e) = [h_A(e) || h_B(e) || h_C(e)] ∈ ℝ^(3d)

    DistMult entity embeddings are d-dim real vectors — same dimensionality
    as TransE — so the joint embedding space is identical (ℝ^768).
    The only architectural difference from Bert+DistMult is the encoder:
    DistilBERT instead of BERT.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim        = embed_dim
        self.question_encoder = QuestionEncoder()

    # ── Fusion ────────────────────────────────────────────────────────────────

    def fuse(self, h_a, h_b, h_c):
        """
        Concatenate per-silo entity embeddings.
        h_a, h_b, h_c : (N, d)  — one row per shared entity
        returns        : (N, 3d)
        """
        return torch.cat([h_a, h_b, h_c], dim=-1)

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score_candidates(self, q_embed, h_joint, candidate_ids):
        """
        Cosine similarity between question vector and each candidate entity.

        sim(e) = cos(q_final, h_joint(e))
               = (q_final · h_joint(e)) / (||q_final|| · ||h_joint(e)||)

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

        e⁺ = highest-scoring correct answer in candidate set
        e⁻ = highest-scoring incorrect answer in candidate set
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

        Returns:
            loss : scalar
            sim  : (B, K) cosine similarities over candidate set
        """
        h_joint = self.fuse(h_a, h_b, h_c)                         # (N, 3d)
        q_embed = self.question_encoder(questions, device)          # (B, 3d)

        if USE_TOPIC_ANCHORING:
            # q_final = q_embed + h_joint[topic]
            # Grounds the question at the topic entity's position in joint space
            q_final = q_embed + h_joint[topic_ids]
        else:
            q_final = q_embed

        candidate_ids = candidate_ids.to(device)
        sim  = self.score_candidates(q_final, h_joint, candidate_ids)
        loss = self.ranking_loss(sim, answer_ids_batch, candidate_ids, margin)
        return loss, sim
