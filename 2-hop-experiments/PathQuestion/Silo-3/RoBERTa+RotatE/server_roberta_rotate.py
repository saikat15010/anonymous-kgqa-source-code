# server_roberta_rotate.py — Federated server for RoBERTa + RotatE experiment (PQ2H)
#
# RotatE entity embeddings are 2*embed_dim (complex representation).
# h_joint = [h_A || h_B || h_C] ∈ ℝ^(3 * 2d) = ℝ^(6d) = ℝ^1536
# QuestionEncoder MLP output dim = 6d = 1536
#
# Differences from server_roberta_transe.py / server_roberta_distmult.py:
#   - JOINT_DIM = 6d = 1536 instead of 3d = 768
#   - MLP final layer outputs 1536 instead of 768
#   - Imports from config_roberta_rotate
# RoBERTa encoder, fusion, scoring, loss — all identical.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel

from config_roberta_rotate import (
    BERT_MODEL, MLP_HIDDEN_DIMS, MLP_DROPOUT,
    KGE_EMBED_DIM, USE_TOPIC_ANCHORING, ROBERTA_DIM
)

# RotatE stores 2*embed_dim reals per entity
ENTITY_DIM = 2 * KGE_EMBED_DIM   # 512 per silo
JOINT_DIM  = 3 * ENTITY_DIM      # 1536 for h_joint


class QuestionEncoder(nn.Module):
    """
    RoBERTa <s> token → MLP → q_embed ∈ ℝ^(6d)

    RoBERTa is frozen; only the MLP projection head is trained.
    Output dimension 6d = 1536 matches h_joint = [h_A || h_B || h_C]
    where each h is 2d-dimensional (RotatE complex representation).

    Key RoBERTa notes:
      - No token_type_ids — tokenizer handles this automatically
      - <s> token is the sequence representation (equivalent to [CLS])
      - Access via: out.last_hidden_state[:, 0, :]
    """

    def __init__(self):
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(BERT_MODEL)
        self.roberta   = RobertaModel.from_pretrained(BERT_MODEL)

        for param in self.roberta.parameters():
            param.requires_grad = False

        layers, in_dim = [], ROBERTA_DIM   # 768
        for h_dim in MLP_HIDDEN_DIMS:
            layers += [nn.Linear(in_dim, h_dim), nn.ReLU(),
                       nn.Dropout(MLP_DROPOUT)]
            in_dim = h_dim
        layers += [nn.Linear(in_dim, JOINT_DIM)]   # → 1536
        self.mlp = nn.Sequential(*layers)

    def forward(self, questions, device):
        enc = self.tokenizer(
            questions, return_tensors="pt", padding=True,
            truncation=True, max_length=64
        ).to(device)
        with torch.no_grad():
            out = self.roberta(**enc)
        cls = out.last_hidden_state[:, 0, :]    # (B, 768) — <s> token
        return self.mlp(cls)                    # (B, 6d=1536)


class FedVServer(nn.Module):
    """
    Central server for RoBERTa + RotatE FedV-KGQA (3 silos, PQ2H).
    h_joint(e) = [h_A(e) || h_B(e) || h_C(e)] ∈ ℝ^(6d)
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim        = embed_dim
        self.question_encoder = QuestionEncoder()

    def fuse(self, h_a, h_b, h_c):
        """
        h_a, h_b, h_c : (N, 2d)
        returns        : (N, 6d)
        """
        return torch.cat([h_a, h_b, h_c], dim=-1)

    def score_candidates(self, q_embed, h_joint, candidate_ids):
        """
        q_embed       : (B, 6d)
        h_joint       : (N, 6d)
        candidate_ids : (B, K)   (-1 = padding)
        returns sim   : (B, K)
        """
        safe_ids = candidate_ids.clamp(min=0)
        h_cands  = h_joint[safe_ids]                               # (B, K, 6d)
        q_norm   = F.normalize(q_embed, p=2, dim=-1).unsqueeze(1)  # (B, 1, 6d)
        h_norm   = F.normalize(h_cands, p=2, dim=-1)               # (B, K, 6d)
        return (q_norm * h_norm).sum(dim=-1)                        # (B, K)

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

    def forward(self, questions, topic_ids, h_a, h_b, h_c,
                answer_ids_batch, candidate_ids, device, margin=1.0):
        h_joint = self.fuse(h_a, h_b, h_c)                        # (N, 6d)
        q_embed = self.question_encoder(questions, device)         # (B, 6d)

        if USE_TOPIC_ANCHORING:
            q_final = q_embed + h_joint[topic_ids]
        else:
            q_final = q_embed

        candidate_ids = candidate_ids.to(device)
        sim  = self.score_candidates(q_final, h_joint, candidate_ids)
        loss = self.ranking_loss(sim, answer_ids_batch,
                                 candidate_ids, margin)
        return loss, sim
