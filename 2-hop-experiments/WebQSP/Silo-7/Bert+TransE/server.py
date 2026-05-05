# server.py — Federated server for WebQSP Client7 BERT+TransE (7 silos)
#
# Client7 has 7 silos → joint_dim = 7 * entity_dim = 7 * 256 = 1792
# (Client5 had 5 silos → 1280; Client3 had 3 silos → 768)
#
# MLP: 768 → 512 → 1792
# BERT: token_type_ids ARE used — always passed to the model.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

from config import (BERT_MODEL, BERT_DIM, MLP_HIDDEN_DIMS, MLP_DROPOUT,
                    KGE_EMBED_DIM, USE_TOPIC_ANCHORING)

NUM_SILOS  = 7
ENTITY_DIM = KGE_EMBED_DIM           # d = 256
JOINT_DIM  = NUM_SILOS * ENTITY_DIM  # 7 × 256 = 1792


class QuestionEncoder(nn.Module):
    """
    BERT [CLS] → MLP → q_embed in R^1792. BERT frozen.
    MLP: 768 → 512 → 1792.
    """

    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        self.bert      = BertModel.from_pretrained(BERT_MODEL)

        for param in self.bert.parameters():
            param.requires_grad = False

        # MLP: 768 → 512 → 1792
        layers, in_dim = [], BERT_DIM
        for h_dim in MLP_HIDDEN_DIMS:
            layers += [nn.Linear(in_dim, h_dim), nn.ReLU(),
                       nn.Dropout(MLP_DROPOUT)]
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, JOINT_DIM))   # → 1792
        self.mlp = nn.Sequential(*layers)

    def forward(self, questions, device):
        enc = self.tokenizer(
            questions, return_tensors="pt", padding=True,
            truncation=True, max_length=64).to(device)
        with torch.no_grad():
            out = self.bert(**enc)
        cls = out.last_hidden_state[:, 0, :]
        return self.mlp(cls)   # (B, 1792)


class FedVServer(nn.Module):
    """Central server — WebQSP Client7 BERT+TransE (7 silos)."""

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim        = embed_dim
        self.question_encoder = QuestionEncoder()

    def fuse(self, h_a, h_b, h_c, h_d, h_e, h_f, h_g):
        """(N, d) × 7  →  (N, 7d=1792)"""
        return torch.cat([h_a, h_b, h_c, h_d, h_e, h_f, h_g], dim=-1)

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

    def forward(self, questions, topic_ids, h_a, h_b, h_c, h_d, h_e, h_f, h_g,
                answer_ids_batch, candidate_ids, device, margin=1.0):
        h_joint  = self.fuse(h_a, h_b, h_c, h_d, h_e, h_f, h_g)
        q_proj   = self.question_encoder(questions, device)
        q_final  = q_proj + h_joint[topic_ids] if USE_TOPIC_ANCHORING \
                   else q_proj
        candidate_ids = candidate_ids.to(device)
        sim  = self.score_candidates(q_final, h_joint, candidate_ids)
        loss = self.ranking_loss(sim, answer_ids_batch, candidate_ids, margin)
        return loss, sim
