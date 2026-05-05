# server.py — Federated server for Client7 DistilBert+ComplEx (7 silos, PQ2H)
#
# ComplEx entity embeddings are 2*embed_dim (complex representation).
# h_joint = [h_A || h_B || h_C || h_D || h_E || h_F || h_G]
#          in R^(7 * 2d) = R^3584
#
# Encoder: DistilBERT (distilbert-base-uncased)
#   - 6 transformer layers, ~66M params, 768-dim output
#   - No token_type_ids (key difference from BERT)
#   - CLS token via last_hidden_state[:, 0, :]
#   - MLP: 768 -> 512 -> 3584
#
# Structurally identical to Client7/DistilBert+RotatE server.py.
# (Both ComplEx and RotatE use 2d entity embeddings -> JOINT_DIM = 3584.)
# Only config import points to DistilBert+ComplEx paths.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertModel

from config import (BERT_MODEL, MLP_HIDDEN_DIMS, MLP_DROPOUT,
                    KGE_EMBED_DIM, USE_TOPIC_ANCHORING, DISTILBERT_DIM)

ENTITY_DIM = 2 * KGE_EMBED_DIM   # 512 per silo
JOINT_DIM  = 7 * ENTITY_DIM      # 7 x 512 = 3584


class QuestionEncoder(nn.Module):
    """DistilBERT [CLS] -> MLP -> q_embed in R^(14d=3584). DistilBERT frozen."""

    def __init__(self):
        super().__init__()
        self.tokenizer  = DistilBertTokenizer.from_pretrained(BERT_MODEL)
        self.distilbert = DistilBertModel.from_pretrained(BERT_MODEL)

        for param in self.distilbert.parameters():
            param.requires_grad = False

        # MLP: 768 -> 512 -> 3584
        layers, in_dim = [], DISTILBERT_DIM
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
            out = self.distilbert(**enc)
        cls = out.last_hidden_state[:, 0, :]    # (B, 768)
        return self.mlp(cls)                    # (B, 3584)


class FedVServer(nn.Module):
    """
    Central server — Client7 DistilBert+ComplEx (7 silos).
    h_joint(e) = [h_A(e) || ... || h_G(e)] in R^(7*2d=3584)
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim        = embed_dim
        self.question_encoder = QuestionEncoder()

    def fuse(self, h_a, h_b, h_c, h_d, h_e, h_f, h_g):
        """h_a ... h_g : (N, 2d) each  ->  (N, 14d=3584)"""
        return torch.cat([h_a, h_b, h_c, h_d, h_e, h_f, h_g], dim=-1)

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

    def forward(self, questions, topic_ids,
                h_a, h_b, h_c, h_d, h_e, h_f, h_g,
                answer_ids_batch, candidate_ids, device, margin=1.0):
        h_joint = self.fuse(h_a, h_b, h_c, h_d, h_e, h_f, h_g)
        q_embed = self.question_encoder(questions, device)
        if USE_TOPIC_ANCHORING:
            q_final = q_embed + h_joint[topic_ids]
        else:
            q_final = q_embed
        candidate_ids = candidate_ids.to(device)
        sim  = self.score_candidates(q_final, h_joint, candidate_ids)
        loss = self.ranking_loss(sim, answer_ids_batch, candidate_ids, margin)
        return loss, sim
