# server.py — Federated server for BERT + DistMult | 5 silos | Client5
#
# Differences from Client3 server.py (3 silos):
#   - fuse() takes 5 args (h_a, h_b, h_c, h_d, h_e)
#   - output_dim = 5 * embed_dim = 1280  (not 3 * 256 = 768)
#   - forward() signature takes h_a, h_b, h_c, h_d, h_e
#
# DistMult entity embeddings are d-dim real (same as TransE) →
# server architecture identical to Bert+TransE Client5.
# Everything else — BERT encoder, MLP, cosine scoring, margin loss,
# topic anchoring, gradient splitting — is identical to Client3.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

from config_distmult import (
    BERT_MODEL, MLP_HIDDEN_DIMS, MLP_DROPOUT,
    KGE_EMBED_DIM, USE_TOPIC_ANCHORING, JOINT_DIM
)


class QuestionEncoder(nn.Module):
    """
    BERT [CLS] → MLP → q_embed ∈ ℝ^(5d) = ℝ^1280

    DistMult entity embeddings ∈ ℝᵈ = ℝ^256 per silo (same as TransE).
    Joint embedding: h_joint = [h_A || h_B || h_C || h_D || h_E] ∈ ℝ^1280
    MLP output must match: 768 → 512 → 1280.

    BERT is frozen; only MLP trains during federated QA.
    """

    def __init__(self, embed_dim, hidden_dims=None, dropout=0.1):
        super().__init__()
        output_dim = 5 * embed_dim   # 1280

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
        cls = out.last_hidden_state[:, 0, :]   # (B, 768)
        return self.mlp(cls)                   # (B, 1280)


class FedVServer(nn.Module):
    """
    Federated server for BERT + DistMult | 5 silos | Client5.

    Fusion:
        h_joint(e) = [h_A(e) || h_B(e) || h_C(e) || h_D(e) || h_E(e)]
                   ∈ ℝ^(5d) = ℝ^1280

    Gradient splitting at concat boundary:
        ∂L/∂h_joint → 5 slices of (N, 256) — one per silo.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim        = embed_dim
        self.question_encoder = QuestionEncoder(embed_dim)

    def fuse(self, h_a, h_b, h_c, h_d, h_e):
        """
        Concatenate 5 silo DistMult embeddings.
        h_a … h_e : (N, 256)
        returns    : (N, 1280)
        """
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

    def forward(self, questions, topic_ids, h_a, h_b, h_c, h_d, h_e,
                answer_ids_batch, candidate_ids, device, margin=1.0):
        """
        h_a … h_e    : (N, 256) — DistMult entity embeddings from 5 silos
        h_joint      : (N, 1280) — after fusion
        q_embed      : (B, 1280) — from BERT + MLP
        """
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
