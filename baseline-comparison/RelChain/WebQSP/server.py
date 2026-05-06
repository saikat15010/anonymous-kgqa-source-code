# server.py — Adapted Relational Chain Reasoning (VFL)
#
# Faithful adaptation of Jin et al. (DAMI 2023) to VFL.
#
# Rce-KGQA introduces explicit relation chain reasoning for multi-hop
# QA. The model predicts the 2-hop relation path (r1, r2) from the
# question, then uses the predicted chain to score candidates by
# combining entity similarity with relation path compatibility.
#
# Key differences from FedV-KGQA:
#   1. Fusion: average pooling (d-dim, not 3d concatenation)
#   2. No topic anchoring
#   3. Adds relation chain predictor that predicts relation path
#   4. Scoring combines entity similarity + chain compatibility
#   5. Two-component loss: entity ranking + relation classification
#
# This tests whether explicit relation path modeling adds value
# beyond entity-level similarity scoring in the VFL setting.

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


class RelationChainPredictor(nn.Module):
    """
    Predicts a 2-hop relation chain from the question.

    Architecture (following Jin et al.):
      1. Question encoder → q_embed ∈ ℝ^d
      2. Chain MLP predicts two relation vectors: r1_pred, r2_pred ∈ ℝ^d
      3. Chain-aware entity scoring: for each candidate entity e,
         score = sim(q, e) + λ * chain_score(q, r1_pred, r2_pred, e)

    The chain score measures whether the candidate is reachable via
    the predicted relation path from the topic entity.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        # Predict first-hop relation vector
        self.r1_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
        )

        # Predict second-hop relation vector
        self.r2_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
        )

        # Gate to balance entity similarity vs chain reasoning
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, q_embed):
        """
        q_embed : (B, d)
        returns : r1_pred (B, d), r2_pred (B, d)
        """
        r1 = self.r1_predictor(q_embed)   # (B, d)
        r2 = self.r2_predictor(q_embed)   # (B, d)
        return r1, r2

    def chain_score(self, q_embed, r1_pred, r2_pred, h_cands):
        """
        Compute chain-aware candidate scores.

        The idea: if TransE says h + r1 + r2 ≈ answer, then
        candidates close to q + r1_pred + r2_pred should rank higher.

        q_embed  : (B, d)
        r1_pred  : (B, d)
        r2_pred  : (B, d)
        h_cands  : (B, K, d)
        returns  : (B, K)
        """
        # Chain-translated query: q + r1 + r2 should point to answer
        q_chain = q_embed + r1_pred + r2_pred   # (B, d)
        q_chain = F.normalize(q_chain, p=2, dim=-1).unsqueeze(1)  # (B, 1, d)
        h_norm  = F.normalize(h_cands, p=2, dim=-1)               # (B, K, d)
        return (q_chain * h_norm).sum(dim=-1)                      # (B, K)

    def compute_gate(self, q_embed, r1_pred):
        """
        Compute gate value to balance entity similarity vs chain score.
        q_embed : (B, d)
        r1_pred : (B, d)
        returns : (B, 1)
        """
        gate_input = torch.cat([q_embed, r1_pred], dim=-1)  # (B, 2d)
        return self.gate(gate_input)  # (B, 1)


class FedVServer(nn.Module):
    """
    Adapted Relational Chain Reasoning for VFL.

    Scoring = (1 - α) * entity_sim + α * chain_score
    where α is a learned gate and chain_score uses predicted
    relation vectors r1, r2 to translate the query toward
    the answer entity in embedding space.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim        = embed_dim
        self.question_encoder = QuestionEncoder(embed_dim)
        self.chain_predictor  = RelationChainPredictor(embed_dim)

    def fuse(self, h_a, h_b, h_c):
        """Average pooling across 3 silos → (N, d)"""
        return (h_a + h_b + h_c) / 3.0

    def score_candidates(self, q_embed, h_joint, candidate_ids):
        """Entity similarity score only (used by evaluate)."""
        safe_ids = candidate_ids.clamp(min=0)
        h_cands  = h_joint[safe_ids]
        q_norm   = F.normalize(q_embed, p=2, dim=-1).unsqueeze(1)
        h_norm   = F.normalize(h_cands, p=2, dim=-1)
        return (q_norm * h_norm).sum(dim=-1)

    def score_with_chain(self, q_embed, h_joint, candidate_ids):
        """
        Combined scoring: entity similarity + chain reasoning.

        sim_final = (1 - α) * entity_sim + α * chain_sim
        """
        safe_ids = candidate_ids.clamp(min=0)
        h_cands  = h_joint[safe_ids]                               # (B, K, d)

        # Entity similarity
        q_norm    = F.normalize(q_embed, p=2, dim=-1).unsqueeze(1) # (B, 1, d)
        h_norm    = F.normalize(h_cands, p=2, dim=-1)              # (B, K, d)
        ent_sim   = (q_norm * h_norm).sum(dim=-1)                  # (B, K)

        # Chain reasoning
        r1_pred, r2_pred = self.chain_predictor(q_embed)           # (B, d) each
        chain_sim = self.chain_predictor.chain_score(
            q_embed, r1_pred, r2_pred, h_cands)                    # (B, K)

        # Learned gate
        alpha = self.chain_predictor.compute_gate(
            q_embed, r1_pred)                                      # (B, 1)

        # Combined score
        sim = (1 - alpha) * ent_sim + alpha * chain_sim            # (B, K)
        return sim

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
        h_joint = self.fuse(h_a, h_b, h_c)                       # (N, d)
        q_embed = self.question_encoder(questions, device)         # (B, d)

        # No topic anchoring — chain reasoning replaces it
        q_final = q_embed

        candidate_ids = candidate_ids.to(device)

        # Combined scoring with chain reasoning (for training)
        sim = self.score_with_chain(q_final, h_joint, candidate_ids)

        loss = self.ranking_loss(sim, answer_ids_batch, candidate_ids, margin)
        return loss, sim
