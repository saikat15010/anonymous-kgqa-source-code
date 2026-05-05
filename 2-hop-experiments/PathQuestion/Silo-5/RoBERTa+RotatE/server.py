# server.py — Federated server for Client5 RoBERTa+RotatE (5 silos, PQ2H)
#
# RotatE entity embeddings are 2*embed_dim (complex representation).
# h_joint = [h_A || h_B || h_C || h_D || h_E] in R^(5 * 2d) = R^2560
#
# Encoder: RoBERTa (roberta-base)
#   - 12 transformer layers, ~125M params, 768-dim output
#   - No token_type_ids (BPE tokenizer — key difference from BERT/DistilBERT)
#   - <s> token is the sequence representation: last_hidden_state[:, 0, :]
#   - MLP: 768 -> 512 -> 2560
#
# vs Client5/RoBERTa+TransE:   JOINT_DIM = 2560 not 1280 (RotatE uses 2d)
# vs Client5/Bert+RotatE:      only RobertaTokenizer/RobertaModel vs Bert

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel

from config import (BERT_MODEL, MLP_HIDDEN_DIMS, MLP_DROPOUT,
                    KGE_EMBED_DIM, USE_TOPIC_ANCHORING, ROBERTA_DIM)

ENTITY_DIM = 2 * KGE_EMBED_DIM   # 512 per silo
JOINT_DIM  = 5 * ENTITY_DIM      # 5 x 512 = 2560


class QuestionEncoder(nn.Module):
    """
    RoBERTa <s> -> MLP -> q_embed in R^(10d=2560). RoBERTa frozen.

    RoBERTa output = 768-dim (same as BERT-base).
    RotatE uses 2d-dim entity embeddings -> joint_dim = 10d = 2560.
    MLP: 768 -> 512 -> 2560.

    Critical RoBERTa note: no token_type_ids; <s> token via
    last_hidden_state[:, 0, :].
    """

    def __init__(self):
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(BERT_MODEL)
        self.roberta   = RobertaModel.from_pretrained(BERT_MODEL)

        for param in self.roberta.parameters():
            param.requires_grad = False

        # MLP: 768 -> 512 -> 2560
        layers, in_dim = [], ROBERTA_DIM
        for h_dim in MLP_HIDDEN_DIMS:
            layers += [nn.Linear(in_dim, h_dim), nn.ReLU(),
                       nn.Dropout(MLP_DROPOUT)]
            in_dim = h_dim
        layers += [nn.Linear(in_dim, JOINT_DIM)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, questions, device):
        enc = self.tokenizer(
            questions, return_tensors="pt", padding=True,
            truncation=True, max_length=64
        ).to(device)
        with torch.no_grad():
            out = self.roberta(**enc)
        cls = out.last_hidden_state[:, 0, :]    # (B, 768) — <s> token
        return self.mlp(cls)                    # (B, 2560)


class FedVServer(nn.Module):
    """
    Central server — Client5 RoBERTa+RotatE (5 silos).
    h_joint(e) = [h_A(e) || h_B(e) || h_C(e) || h_D(e) || h_E(e)] in R^2560
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim        = embed_dim
        self.question_encoder = QuestionEncoder()

    def fuse(self, h_a, h_b, h_c, h_d, h_e):
        """h_a ... h_e : (N, 2d) each  ->  (N, 10d=2560)"""
        return torch.cat([h_a, h_b, h_c, h_d, h_e], dim=-1)

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
            answer_set  = set(answer_ids)
            pos_indices = [j for j, c in enumerate(cand_list)
                           if c in answer_set]
            if not pos_indices:
                continue
            best_pos = scores[pos_indices].max()
            neg_mask = torch.ones(len(cand_list), dtype=torch.bool,
                                  device=device)
            for j in pos_indices:
                neg_mask[j] = False
            if neg_mask.sum() == 0:
                continue
            hard_neg = scores[neg_mask].max()
            losses.append(F.relu(margin + hard_neg - best_pos))
        if not losses:
            return torch.tensor(0.0, requires_grad=True, device=device)
        return torch.stack(losses).mean()

    def forward(self, questions, topic_ids, h_a, h_b, h_c, h_d, h_e,
                answer_ids_batch, candidate_ids, device, margin=1.0):
        h_joint = self.fuse(h_a, h_b, h_c, h_d, h_e)
        q_embed = self.question_encoder(questions, device)
        if USE_TOPIC_ANCHORING:
            q_final = q_embed + h_joint[topic_ids]
        else:
            q_final = q_embed
        candidate_ids = candidate_ids.to(device)
        sim  = self.score_candidates(q_final, h_joint, candidate_ids)
        loss = self.ranking_loss(sim, answer_ids_batch, candidate_ids, margin)
        return loss, sim
