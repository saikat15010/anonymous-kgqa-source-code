# server.py — Adapted FedE (VFL)
#
# Faithful adaptation of Chen et al. (IJCKG 2021) to VFL.
#
# FedE's core mechanism: share entity embeddings across clients via
# FedAvg after each training round. All clients converge toward a
# unified entity representation.
#
# Key differences from FedV-KGQA:
#   1. FedAvg on entity embeddings after each epoch (in train_fedv.py)
#      → all silos get identical entity embeddings
#   2. Fusion: average pooling (d-dim), since silos are homogenized
#      by FedAvg, concatenation would just repeat the same embedding 3x
#   3. No topic anchoring
#   4. No OWL2 enrichment benefit (FedAvg destroys silo-specific
#      geometry that enrichment creates)
#
# This tests whether homogenizing entity embeddings via FedAvg
# preserves enough information for multi-hop QA, or whether
# silo-specific representations (as in FedV-KGQA) are necessary.

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


def fedavg_entity_embeddings(model_a, model_b, model_c):
    """
    FedAvg on entity embeddings: average across all 3 silos,
    then broadcast back. This is FedE's core mechanism.

    After this call, all 3 silos have identical entity embeddings.
    This destroys silo-specific geometry but creates a unified
    entity representation space.
    """
    with torch.no_grad():
        avg = (model_a.ent_embed.weight.data +
               model_b.ent_embed.weight.data +
               model_c.ent_embed.weight.data) / 3.0
        avg = F.normalize(avg, p=2, dim=-1)
        model_a.ent_embed.weight.data.copy_(avg)
        model_b.ent_embed.weight.data.copy_(avg)
        model_c.ent_embed.weight.data.copy_(avg)


class FedVServer(nn.Module):
    """
    Adapted FedE for VFL.

    Since FedAvg makes all silo embeddings identical, we use average
    pooling (equivalent to using any single silo's embeddings) in
    d-dimensional space. No topic anchoring.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim        = embed_dim
        self.question_encoder = QuestionEncoder(embed_dim)

    def fuse(self, h_a, h_b, h_c):
        """
        Average pooling across 3 silos → (N, d).
        After FedAvg, h_a ≈ h_b ≈ h_c, so this is effectively
        just using the unified embedding.
        """
        return (h_a + h_b + h_c) / 3.0

    def score_candidates(self, q_embed, h_joint, candidate_ids):
        safe_ids = candidate_ids.clamp(min=0)
        h_cands  = h_joint[safe_ids]
        q_norm   = F.normalize(q_embed, p=2, dim=-1).unsqueeze(1)
        h_norm   = F.normalize(h_cands, p=2, dim=-1)
        return (q_norm * h_norm).sum(dim=-1)

    def ranking_loss(self, sim, answer_ids_batch, candidate_ids, margin=1.0):
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
