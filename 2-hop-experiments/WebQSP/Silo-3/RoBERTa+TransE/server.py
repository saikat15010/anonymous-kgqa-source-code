# server.py — Federated server for WebQSP Client3 RoBERTa+TransE (3 silos)
#
# TransE entity_dim = d = 256  →  joint_dim = 3 * d = 768
# MLP: 768 → 512 → 768
#
# RoBERTa differences vs BERT:
#   1. RobertaTokenizer / RobertaModel (NOT BertTokenizer / BertModel)
#   2. NO token_type_ids — RoBERTa does not use segment embeddings.
#      Passing token_type_ids raises an error.
#   3. [CLS] token is still index 0 in last_hidden_state.
#   4. Same 768-dim output → joint_dim and MLP architecture unchanged.
#
# RoBERTa differences vs DistilBERT:
#   - 12 layers vs 6 (larger, slower, generally more accurate)
#   - Both have no token_type_ids
#   - Both use the same MLP architecture for the same KGE model

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel

from config import (ROBERTA_MODEL, ROBERTA_DIM, MLP_HIDDEN_DIMS,
                    MLP_DROPOUT, KGE_EMBED_DIM, USE_TOPIC_ANCHORING)

ENTITY_DIM = KGE_EMBED_DIM        # d = 256
JOINT_DIM  = 3 * ENTITY_DIM       # 3 × 256 = 768


class QuestionEncoder(nn.Module):
    """
    RoBERTa [CLS] → MLP → q_embed in R^(3d=768). RoBERTa frozen.
    MLP: 768 → 512 → 768.

    Critical: token_type_ids are NOT passed to RoBERTa.
    RoBERTa uses BPE tokenisation (not WordPiece like BERT).
    Output dim is 768 — same as BERT-base and DistilBERT-base.
    """

    def __init__(self):
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL)
        self.roberta   = RobertaModel.from_pretrained(ROBERTA_MODEL)

        # Freeze RoBERTa — only the MLP is trained during QA fine-tuning
        for param in self.roberta.parameters():
            param.requires_grad = False

        # MLP: 768 → 512 → 768
        layers, in_dim = [], ROBERTA_DIM
        for h_dim in MLP_HIDDEN_DIMS:
            layers += [nn.Linear(in_dim, h_dim), nn.ReLU(),
                       nn.Dropout(MLP_DROPOUT)]
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, JOINT_DIM))
        self.mlp = nn.Sequential(*layers)

    def forward(self, questions, device):
        # token_type_ids deliberately omitted — RoBERTa does not support them
        enc = self.tokenizer(
            questions, return_tensors="pt", padding=True,
            truncation=True, max_length=64).to(device)
        with torch.no_grad():
            out = self.roberta(**enc)
        cls = out.last_hidden_state[:, 0, :]   # (B, 768) — [CLS] token
        return self.mlp(cls)                    # (B, 768)


class FedVServer(nn.Module):
    """Central server — WebQSP Client3 RoBERTa+TransE (3 silos)."""

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim        = embed_dim
        self.question_encoder = QuestionEncoder()

    def fuse(self, h_a, h_b, h_c):
        """(N, d) × 3  →  (N, 3d=768)"""
        return torch.cat([h_a, h_b, h_c], dim=-1)

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

    def forward(self, questions, topic_ids, h_a, h_b, h_c,
                answer_ids_batch, candidate_ids, device, margin=1.0):
        h_joint  = self.fuse(h_a, h_b, h_c)
        q_proj   = self.question_encoder(questions, device)
        q_final  = q_proj + h_joint[topic_ids] if USE_TOPIC_ANCHORING \
                   else q_proj
        candidate_ids = candidate_ids.to(device)
        sim  = self.score_candidates(q_final, h_joint, candidate_ids)
        loss = self.ranking_loss(sim, answer_ids_batch, candidate_ids, margin)
        return loss, sim
