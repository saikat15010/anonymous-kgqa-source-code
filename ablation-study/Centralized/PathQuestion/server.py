# server.py — Centralized baseline server
# Single TransE model, no federation, no fusion.
# joint_dim = d = 256 (single model embedding).
# DistilBERT [CLS] → MLP → q_embed ∈ ℝ^d.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertModel

from config import (BERT_MODEL, MLP_HIDDEN_DIMS, MLP_DROPOUT,
                    DISTILBERT_DIM, JOINT_DIM, USE_TOPIC_ANCHORING)


class QuestionEncoder(nn.Module):
    """DistilBERT [CLS] → MLP → q_embed ∈ ℝ^d (256)."""

    def __init__(self):
        super().__init__()
        self.tokenizer  = DistilBertTokenizer.from_pretrained(BERT_MODEL)
        self.distilbert = DistilBertModel.from_pretrained(BERT_MODEL)

        for param in self.distilbert.parameters():
            param.requires_grad = False

        # MLP: 768 → 512 → 256
        layers, in_dim = [], DISTILBERT_DIM
        for h_dim in MLP_HIDDEN_DIMS:
            layers += [nn.Linear(in_dim, h_dim), nn.ReLU(),
                       nn.Dropout(MLP_DROPOUT)]
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, JOINT_DIM))
        self.mlp = nn.Sequential(*layers)

    def forward(self, questions, device):
        enc = self.tokenizer(
            questions, return_tensors="pt", padding=True,
            truncation=True, max_length=64).to(device)
        with torch.no_grad():
            out = self.distilbert(**enc)
        cls = out.last_hidden_state[:, 0, :]   # (B, 768)
        return self.mlp(cls)                    # (B, 256)


class CentralizedServer(nn.Module):
    """
    Centralized baseline — NO federation.
    Single TransE model, entity embeddings used directly (no fusion).
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim        = embed_dim
        self.question_encoder = QuestionEncoder()

    def score_candidates(self, q_embed, h_all, candidate_ids):
        safe_ids = candidate_ids.clamp(min=0)
        h_cands  = h_all[safe_ids]                                 # (B, K, d)
        q_norm   = F.normalize(q_embed, p=2, dim=-1).unsqueeze(1)  # (B, 1, d)
        h_norm   = F.normalize(h_cands, p=2, dim=-1)               # (B, K, d)
        return (q_norm * h_norm).sum(dim=-1)                        # (B, K)

    def ranking_loss(self, sim, answer_ids_batch, candidate_ids, margin=1.0):
        device = sim.device
        losses = []
        for i, answer_ids in enumerate(answer_ids_batch):
            cands      = candidate_ids[i]
            valid_mask = cands >= 0
            cand_list  = cands[valid_mask].tolist()
            scores     = sim[i][valid_mask]
            answer_set = set(answer_ids)
            pos_idx    = [j for j, c in enumerate(cand_list) if c in answer_set]
            if not pos_idx:
                continue
            best_pos = scores[pos_idx].max()
            neg_mask = torch.ones(len(cand_list), dtype=torch.bool, device=device)
            for j in pos_idx:
                neg_mask[j] = False
            if neg_mask.sum() == 0:
                continue
            losses.append(F.relu(margin + scores[neg_mask].max() - best_pos))
        if not losses:
            return torch.tensor(0.0, requires_grad=True, device=device)
        return torch.stack(losses).mean()

    def forward(self, questions, topic_ids, h_all,
                answer_ids_batch, candidate_ids, device, margin=1.0):
        """
        h_all   : (N, d) — single TransE entity embeddings (no fusion)
        q_embed : (B, d) — from DistilBERT + MLP
        """
        q_embed = self.question_encoder(questions, device)    # (B, d)

        if USE_TOPIC_ANCHORING:
            q_final = q_embed + h_all[topic_ids]
        else:
            q_final = q_embed

        candidate_ids = candidate_ids.to(device)
        sim  = self.score_candidates(q_final, h_all, candidate_ids)
        loss = self.ranking_loss(sim, answer_ids_batch, candidate_ids, margin)
        return loss, sim
