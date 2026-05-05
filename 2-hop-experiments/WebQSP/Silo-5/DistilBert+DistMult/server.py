# server.py — Federated server for WebQSP Client5 DistilBERT+DistMult (5 silos)
#
# DistMult entity_dim = d = 256
# Client5: joint_dim = 5 * d = 5 * 256 = 1280
# MLP: 768 → 512 → 1280
#
# DistilBERT: NO token_type_ids, uses DistilBertTokenizer/DistilBertModel.
# Same 768-dim output as BERT — joint_dim and MLP unchanged.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertModel

from config import (DISTILBERT_MODEL, DISTILBERT_DIM, MLP_HIDDEN_DIMS,
                    MLP_DROPOUT, KGE_EMBED_DIM, USE_TOPIC_ANCHORING)

NUM_SILOS  = 5
ENTITY_DIM = KGE_EMBED_DIM           # d = 256
JOINT_DIM  = NUM_SILOS * ENTITY_DIM  # 5 × 256 = 1280


class QuestionEncoder(nn.Module):
    """
    DistilBERT [CLS] → MLP → q_embed in R^1280. DistilBERT frozen.
    MLP: 768 → 512 → 1280.
    token_type_ids are NOT passed — DistilBERT does not support them.
    """

    def __init__(self):
        super().__init__()
        self.tokenizer  = DistilBertTokenizer.from_pretrained(DISTILBERT_MODEL)
        self.distilbert = DistilBertModel.from_pretrained(DISTILBERT_MODEL)

        for param in self.distilbert.parameters():
            param.requires_grad = False

        layers, in_dim = [], DISTILBERT_DIM
        for h_dim in MLP_HIDDEN_DIMS:
            layers += [nn.Linear(in_dim, h_dim), nn.ReLU(),
                       nn.Dropout(MLP_DROPOUT)]
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, JOINT_DIM))   # → 1280
        self.mlp = nn.Sequential(*layers)

    def forward(self, questions, device):
        enc = self.tokenizer(
            questions, return_tensors="pt", padding=True,
            truncation=True, max_length=64).to(device)
        with torch.no_grad():
            out = self.distilbert(**enc)
        cls = out.last_hidden_state[:, 0, :]
        return self.mlp(cls)   # (B, 1280)


class FedVServer(nn.Module):
    """Central server — WebQSP Client5 DistilBERT+DistMult (5 silos)."""

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim        = embed_dim
        self.question_encoder = QuestionEncoder()

    def fuse(self, h_a, h_b, h_c, h_d, h_e):
        """(N, d) × 5  →  (N, 5d=1280)"""
        return torch.cat([h_a, h_b, h_c, h_d, h_e], dim=-1)

    def score_candidates(self, q_embed, h_joint, candidate_ids):
        safe_ids = candidate_ids.clamp(min=0)
        h_cands  = h_joint[safe_ids]
        q_norm   = F.normalize(q_embed, p=2, dim=-1)
        h_norm   = F.normalize(h_cands, p=2, dim=-1)
        return (q_norm.unsqueeze(1) * h_norm).sum(dim=-1)

    def ranking_loss(self, sim, answer_ids_batch, candidate_ids, margin=1.0):
        device = sim.device
        losses = []
        for i, answer_ids in enumerate(answer_ids_batch):
            cands      = candidate_ids[i]
            valid_mask = cands >= 0
            cand_list  = cands[valid_mask].tolist()
            scores     = sim[i][valid_mask]
            answer_set = set(answer_ids)
            pos_idx    = [j for j, c in enumerate(cand_list)
                          if c in answer_set]
            if not pos_idx:
                continue
            best_pos = scores[pos_idx].max()
            neg_mask = torch.ones(len(cand_list), dtype=torch.bool,
                                  device=device)
            for j in pos_idx:
                neg_mask[j] = False
            if neg_mask.sum() == 0:
                continue
            losses.append(F.relu(margin + scores[neg_mask].max() - best_pos))
        if not losses:
            return torch.tensor(0.0, requires_grad=True, device=device)
        return torch.stack(losses).mean()

    def forward(self, questions, topic_ids, h_a, h_b, h_c, h_d, h_e,
                answer_ids_batch, candidate_ids, device, margin=1.0):
        h_joint  = self.fuse(h_a, h_b, h_c, h_d, h_e)
        q_proj   = self.question_encoder(questions, device)
        q_final  = q_proj + h_joint[topic_ids] if USE_TOPIC_ANCHORING \
                   else q_proj
        candidate_ids = candidate_ids.to(device)
        sim  = self.score_candidates(q_final, h_joint, candidate_ids)
        loss = self.ranking_loss(sim, answer_ids_batch, candidate_ids, margin)
        return loss, sim
