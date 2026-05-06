# server.py — Adapted EmbedKGQA (VFL)
#
# Faithful adaptation of Saxena et al. (ACL 2020) to VFL.
#
# Key differences from FedV-KGQA:
#   1. Fusion: average pooling (not concatenation) → loses silo geometry
#   2. No topic anchoring → query is ungrounded in KG space
#   3. Scoring dimension: d (not 3d)
#   4. Training: margin ranking over ALL entities (not just candidates)
#
# The combination of avg pooling + no anchoring + global ranking makes
# this substantially weaker than FedV-KGQA on multi-hop questions.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

from config import BERT_MODEL, MLP_HIDDEN_DIMS, MLP_DROPOUT, KGE_EMBED_DIM


class QuestionEncoder(nn.Module):
    """BERT [CLS] → MLP → q_embed ∈ ℝ^d"""

    def __init__(self, embed_dim, hidden_dims=None, dropout=0.1):
        super().__init__()
        output_dim = embed_dim  # d, not 3d

        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        self.bert      = BertModel.from_pretrained(BERT_MODEL)

        for param in self.bert.parameters():
            param.requires_grad = False

        if hidden_dims is None:
            hidden_dims = MLP_HIDDEN_DIMS

        layers, in_dim = [], 768
        for h_dim in hidden_dims:
            layers += [nn.Linear(in_dim, h_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h_dim
        layers += [nn.Linear(in_dim, output_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, questions, device):
        enc = self.tokenizer(
            questions, return_tensors="pt", padding=True,
            truncation=True, max_length=64
        ).to(device)
        with torch.no_grad():
            out = self.bert(**enc)
        cls = out.last_hidden_state[:, 0, :]
        return self.mlp(cls)


class FedVServer(nn.Module):
    """
    Adapted EmbedKGQA for VFL.

    Training: margin ranking over ALL entities (hard negative mining
    from the entire entity set, not just precomputed candidates).
    This is harder than FedV-KGQA's candidate-based ranking.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim        = embed_dim
        self.question_encoder = QuestionEncoder(embed_dim)

    def fuse(self, h_a, h_b, h_c):
        """Average pooling across 3 silos → (N, d)"""
        return (h_a + h_b + h_c) / 3.0

    def score_all_entities(self, q_embed, h_joint):
        """
        Score ALL entities globally.
        q_embed : (B, d)
        h_joint : (N, d)
        returns : (B, N)
        """
        q_norm = F.normalize(q_embed, p=2, dim=-1)
        h_norm = F.normalize(h_joint, p=2, dim=-1)
        return torch.matmul(q_norm, h_norm.t())

    def score_candidates(self, q_embed, h_joint, candidate_ids):
        """Score precomputed candidates (for backward compat with evaluate)."""
        safe_ids = candidate_ids.clamp(min=0)
        h_cands  = h_joint[safe_ids]
        q_norm   = F.normalize(q_embed, p=2, dim=-1).unsqueeze(1)
        h_norm   = F.normalize(h_cands, p=2, dim=-1)
        return (q_norm * h_norm).sum(dim=-1)

    def global_ranking_loss(self, scores_all, answer_ids_batch, margin=1.0):
        """
        Margin ranking loss over ALL entities.

        For each question:
          - best_pos = max score among correct answers
          - hard_neg = max score among ALL incorrect entities
          - loss = max(0, margin + hard_neg - best_pos)

        This is harder than candidate-based ranking because the hard
        negative is mined from the entire entity set (N entities),
        not just ~100-200 precomputed candidates.
        """
        device = scores_all.device
        B, N = scores_all.shape
        losses = []

        for i, answer_ids in enumerate(answer_ids_batch):
            if not answer_ids:
                continue

            scores = scores_all[i]  # (N,)

            # Positive mask
            pos_mask = torch.zeros(N, dtype=torch.bool, device=device)
            for aid in answer_ids:
                if 0 <= aid < N:
                    pos_mask[aid] = True

            if pos_mask.sum() == 0:
                continue

            # Negative mask = everything that is not a positive
            neg_mask = ~pos_mask

            if neg_mask.sum() == 0:
                continue

            best_pos = scores[pos_mask].max()
            hard_neg = scores[neg_mask].max()

            losses.append(F.relu(margin + hard_neg - best_pos))

        if not losses:
            return torch.tensor(0.0, requires_grad=True, device=device)
        return torch.stack(losses).mean()

    def ranking_loss(self, sim, answer_ids_batch, candidate_ids, margin=1.0):
        """Candidate-based ranking loss (kept for API compatibility)."""
        device = sim.device
        losses = []
        for i, answer_ids in enumerate(answer_ids_batch):
            cands      = candidate_ids[i]
            valid_mask = cands >= 0
            cand_list  = cands[valid_mask].tolist()
            scores     = sim[i][valid_mask]
            answer_set = set(answer_ids)
            pos_indices = [j for j, c in enumerate(cand_list) if c in answer_set]
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

    def forward(self, questions, topic_ids, h_a, h_b, h_c,
                answer_ids_batch, candidate_ids, device, margin=1.0):
        """
        Forward pass.
        Training loss: margin ranking over ALL entities.
        Also returns candidate scores for evaluate compatibility.
        """
        h_joint = self.fuse(h_a, h_b, h_c)                      # (N, d)
        q_embed = self.question_encoder(questions, device)        # (B, d)

        # No topic anchoring
        q_final = q_embed

        # Score ALL entities for training loss
        scores_all = self.score_all_entities(q_final, h_joint)    # (B, N)

        # Global margin ranking loss (harder than candidate-based)
        loss = self.global_ranking_loss(scores_all, answer_ids_batch, margin)

        # Also compute candidate scores for evaluate compatibility
        candidate_ids = candidate_ids.to(device)
        sim = self.score_candidates(q_final, h_joint, candidate_ids)

        return loss, sim