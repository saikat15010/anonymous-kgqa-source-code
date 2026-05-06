# server.py — Adapted FL-KG-QA (VFL)
#
# Faithful adaptation of Gunti et al. (JISEM 2024) to VFL.
#
# FL-KG-QA is a simple federated QA method designed for single-hop
# questions. It does not have multi-hop reasoning, topic anchoring,
# or sophisticated fusion mechanisms.
#
# Key differences from FedV-KGQA:
#   1. Fusion: average pooling (simple federation, no silo geometry)
#   2. No topic anchoring
#   3. Candidates: 1-hop only (CANDIDATE_HOP2_CAP=0 in config)
#   4. Scoring dimension: d (not 3d)
#
# The 1-hop restriction is the critical difference: on 2-hop questions,
# the correct answer is often unreachable within 1-hop of the topic
# entity, causing a fundamental coverage gap.

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
    Adapted FL-KG-QA for VFL.

    Simple single-hop federated QA with average pooling fusion.
    Candidates are restricted to 1-hop neighbors (via config).
    No topic anchoring, no multi-hop candidate expansion.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim        = embed_dim
        self.question_encoder = QuestionEncoder(embed_dim)

    def fuse(self, h_a, h_b, h_c):
        """Average pooling across 3 silos → (N, d)"""
        return (h_a + h_b + h_c) / 3.0

    def score_candidates(self, q_embed, h_joint, candidate_ids):
        """Score 1-hop candidates only (candidates come from config with HOP2=0)."""
        safe_ids = candidate_ids.clamp(min=0)
        h_cands  = h_joint[safe_ids]
        q_norm   = F.normalize(q_embed, p=2, dim=-1).unsqueeze(1)
        h_norm   = F.normalize(h_cands, p=2, dim=-1)
        return (q_norm * h_norm).sum(dim=-1)

    def ranking_loss(self, sim, answer_ids_batch, candidate_ids, margin=1.0):
        """Margin ranking loss over 1-hop candidates."""
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
        h_joint = self.fuse(h_a, h_b, h_c)
        q_embed = self.question_encoder(questions, device)

        # No topic anchoring
        q_final = q_embed

        candidate_ids = candidate_ids.to(device)
        sim  = self.score_candidates(q_final, h_joint, candidate_ids)
        loss = self.ranking_loss(sim, answer_ids_batch, candidate_ids, margin)
        return loss, sim
