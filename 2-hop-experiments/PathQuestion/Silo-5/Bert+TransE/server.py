# server.py — Federated server for Client5 (5 silos, PQ2H)
#
# KEY DIFFERENCE FROM CLIENT3:
#   Client3 (3 silos): h_joint = [h_A || h_B || h_C] ∈ ℝ^(3d) = ℝ^768
#   Client5 (5 silos): h_joint = [h_A || h_B || h_C || h_D || h_E] ∈ ℝ^(5d) = ℝ^1280
#
# The MLP now projects BERT's 768-dim CLS output to 1280 (5d) instead of 768 (3d).
# All other logic — cosine scoring, margin ranking loss, topic anchoring — is identical.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

from config import (BERT_MODEL, MLP_HIDDEN_DIMS, MLP_DROPOUT,
                    KGE_EMBED_DIM, USE_TOPIC_ANCHORING)

# 5 silos × d dims each
JOINT_DIM = 5 * KGE_EMBED_DIM   # 5 × 256 = 1280


class QuestionEncoder(nn.Module):
    """
    BERT [CLS] → MLP → q_embed ∈ ℝ^(5d)

    Output dimension 5d = 1280 matches
    h_joint = [h_A || h_B || h_C || h_D || h_E].
    BERT is frozen; only MLP is trained.
    """

    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        self.bert      = BertModel.from_pretrained(BERT_MODEL)

        for param in self.bert.parameters():
            param.requires_grad = False

        # MLP: 768 → 512 → 1280
        layers, in_dim = [], 768
        for h_dim in MLP_HIDDEN_DIMS:
            layers += [nn.Linear(in_dim, h_dim), nn.ReLU(),
                       nn.Dropout(MLP_DROPOUT)]
            in_dim = h_dim
        layers += [nn.Linear(in_dim, JOINT_DIM)]   # final: 512 → 1280
        self.mlp = nn.Sequential(*layers)

    def forward(self, questions, device):
        enc = self.tokenizer(
            questions, return_tensors="pt", padding=True,
            truncation=True, max_length=64
        ).to(device)
        with torch.no_grad():
            out = self.bert(**enc)
        cls = out.last_hidden_state[:, 0, :]    # (B, 768)
        return self.mlp(cls)                    # (B, 5d=1280)


class FedVServer(nn.Module):
    """
    Central server for FedV-KGQA — Client5 (5 silos, PQ2H).

    Fusion:
        h_joint(e) = [h_A(e) || h_B(e) || h_C(e) || h_D(e) || h_E(e)] ∈ ℝ^(5d)

    Gradient splitting:
        ∂L/∂h_joint is split at concat boundaries → each silo receives (N, d).
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim        = embed_dim
        self.question_encoder = QuestionEncoder()

    # ── Fusion ────────────────────────────────────────────────────────────────

    def fuse(self, h_a, h_b, h_c, h_d, h_e):
        """
        Concatenate 5 per-silo entity embeddings.

        h_a, h_b, h_c, h_d, h_e : (N, d) each
        returns                  : (N, 5d)
        """
        return torch.cat([h_a, h_b, h_c, h_d, h_e], dim=-1)

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score_candidates(self, q_embed, h_joint, candidate_ids):
        """
        Cosine similarity against precomputed candidate entities.

        q_embed       : (B, 5d)
        h_joint       : (N, 5d)
        candidate_ids : (B, K)   (-1 = padding)
        returns sim   : (B, K)
        """
        safe_ids = candidate_ids.clamp(min=0)
        h_cands  = h_joint[safe_ids]                               # (B, K, 5d)
        q_norm   = F.normalize(q_embed, p=2, dim=-1).unsqueeze(1)  # (B, 1, 5d)
        h_norm   = F.normalize(h_cands, p=2, dim=-1)               # (B, K, 5d)
        return (q_norm * h_norm).sum(dim=-1)                        # (B, K)

    # ── Loss ──────────────────────────────────────────────────────────────────

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

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, questions, topic_ids, h_a, h_b, h_c, h_d, h_e,
                answer_ids_batch, candidate_ids, device, margin=1.0):
        """
        Full server forward pass — 5 silo embeddings.

        h_a … h_e : (N, d) — TransE entity embeddings from each silo
        h_joint   : (N, 5d) — after fusion
        q_embed   : (B, 5d) — from BERT + MLP
        """
        h_joint = self.fuse(h_a, h_b, h_c, h_d, h_e)              # (N, 5d)
        q_embed = self.question_encoder(questions, device)          # (B, 5d)

        if USE_TOPIC_ANCHORING:
            q_final = q_embed + h_joint[topic_ids]
        else:
            q_final = q_embed

        candidate_ids = candidate_ids.to(device)
        sim  = self.score_candidates(q_final, h_joint, candidate_ids)
        loss = self.ranking_loss(sim, answer_ids_batch,
                                 candidate_ids, margin)
        return loss, sim
