# src/server.py — Federated server
# Concatenation aggregation + optional topic anchoring + precomputed candidates

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

from config import BERT_MODEL, MLP_HIDDEN_DIMS, MLP_DROPOUT, KGE_EMBED_DIM, USE_TOPIC_ANCHORING


class QuestionEncoder(nn.Module):
    """
    BERT [CLS] → MLP → q_embed ∈ ℝ^(3d)
    Output dimension 3d matches h_joint = [h_A || h_B || h_C].
    BERT is frozen; only MLP is trained (for speed).
    """

    def __init__(self, embed_dim, hidden_dims=None, dropout=0.1):
        super().__init__()
        output_dim = 3 * embed_dim

        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        self.bert      = BertModel.from_pretrained(BERT_MODEL)

        # Freeze BERT — only MLP trains
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
        with torch.no_grad():                          # BERT frozen → no grad
            out = self.bert(**enc)
        cls = out.last_hidden_state[:, 0, :]           # (B, 768)
        return self.mlp(cls)                           # (B, 3d)


class FedVServer(nn.Module):
    """
    Central server for FedV-KGQA (3 silos).

    Aggregation:
        h_joint(e) = [h_A(e) || h_B(e) || h_C(e)] ∈ ℝ^(3d)
        One row per entity. Each silo contributes d dims.

    Topic anchoring (optional, controlled by USE_TOPIC_ANCHORING):
        q_final = q_embed + h_joint[topic_id]
        Grounds the query in KG space. Only useful after KB enrichment
        gives persons strong embeddings. Can be disabled via config.

    Candidate scoring:
        Score only precomputed 2-hop neighbors of topic entity
        instead of all N entities → ~100x faster than full scoring.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim        = embed_dim
        self.question_encoder = QuestionEncoder(embed_dim)

    # ── Fusion ────────────────────────────────────────────────────────────────

    def fuse(self, h_a, h_b, h_c):
        """
        Entity-wise concatenation across 3 silos.
        Each entity gets contributions from all 3 silos.

        h_a, h_b, h_c : (N, d)
        returns        : (N, 3d)
        """
        return torch.cat([h_a, h_b, h_c], dim=-1)

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score_candidates(self, q_embed, h_joint, candidate_ids):
        """
        Cosine similarity against precomputed candidate entities only.

        q_embed       : (B, 3d)
        h_joint       : (N, 3d)
        candidate_ids : (B, K)  — -1 = padding

        returns sim   : (B, K)
        """
        safe_ids = candidate_ids.clamp(min=0)          # handle -1 padding
        h_cands  = h_joint[safe_ids]                   # (B, K, 3d)
        q_norm   = F.normalize(q_embed, p=2, dim=-1).unsqueeze(1)  # (B, 1, 3d)
        h_norm   = F.normalize(h_cands, p=2, dim=-1)               # (B, K, 3d)
        return (q_norm * h_norm).sum(dim=-1)            # (B, K)

    # ── Loss ──────────────────────────────────────────────────────────────────

    def ranking_loss(self, sim, answer_ids_batch, candidate_ids, margin=1.0):
        """
        Margin ranking loss over candidate entities.

        For each question:
            best_pos  = max score among correct answer candidates
            hard_neg  = max score among incorrect candidates
            loss      = max(0, margin + hard_neg - best_pos)

        sim           : (B, K)
        candidate_ids : (B, K)  — -1 = padding
        """
        device = sim.device
        losses = []

        for i, answer_ids in enumerate(answer_ids_batch):
            cands      = candidate_ids[i]              # (K,)
            valid_mask = cands >= 0                    # exclude padding
            cand_list  = cands[valid_mask].tolist()
            scores     = sim[i][valid_mask]            # (K',)

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

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, questions, topic_ids, h_a, h_b, h_c,
                answer_ids_batch, candidate_ids, device, margin=1.0):
        """
        Full server forward pass.

        questions        : list of B strings
        topic_ids        : LongTensor (B,)
        h_a, h_b, h_c   : FloatTensor (N, d) — silo embeddings
        answer_ids_batch : list of list of int
        candidate_ids    : LongTensor (B, K) — precomputed, -1 = padding
        """
        # Step 1: Fuse — entity-wise concatenation across 3 silos
        h_joint = self.fuse(h_a, h_b, h_c)                     # (N, 3d)

        # Step 2: Encode question with frozen BERT + trainable MLP
        q_embed = self.question_encoder(questions, device)       # (B, 3d)

        # Step 3: Topic anchoring (optional)
        # Adds KG position of topic entity to query vector.
        # Only meaningful after KB enrichment gives persons strong embeddings.
        if USE_TOPIC_ANCHORING:
            topic_emb = h_joint[topic_ids]                       # (B, 3d)
            q_final   = q_embed + topic_emb                      # (B, 3d)
        else:
            q_final = q_embed                                    # (B, 3d)

        # Step 4: Score only precomputed candidates (not all N entities)
        candidate_ids = candidate_ids.to(device)
        sim  = self.score_candidates(q_final, h_joint, candidate_ids)  # (B, K)

        # Step 5: Compute ranking loss
        loss = self.ranking_loss(sim, answer_ids_batch,
                                 candidate_ids, margin)
        return loss, sim
