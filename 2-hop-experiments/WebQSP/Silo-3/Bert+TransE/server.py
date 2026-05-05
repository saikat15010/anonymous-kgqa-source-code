# server.py — Federated server for WebQSP Client3 BERT+TransE (3 silos)
#
# TransE entity embeddings are d-dimensional reals.
# h_joint(e) = [h_A(e) || h_B(e) || h_C(e)]  in R^(3d) = R^768
#
# Encoder: BERT-base-uncased
#   - 12 transformer layers, ~110M params, 768-dim hidden
#   - [CLS] token: last_hidden_state[:, 0, :]
#   - token_type_ids used (unlike RoBERTa)
#   - WordPiece tokeniser — handles Freebase entity surface names well
#
# MLP: 768 → 512 → 768  (matches joint_dim = 3 × 256 = 768)
#
# Client3 has 3 silo optimisers (one per silo entity embedding matrix).
# Gradient splitting: server sends each silo only its d-column slice.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

from config import (BERT_MODEL, BERT_DIM, MLP_HIDDEN_DIMS, MLP_DROPOUT,
                    KGE_EMBED_DIM, USE_TOPIC_ANCHORING)

ENTITY_DIM = KGE_EMBED_DIM          # d = 256  (TransE real embedding)
JOINT_DIM  = 3 * ENTITY_DIM         # 3 silos × 256 = 768


class QuestionEncoder(nn.Module):
    """
    BERT [CLS] → MLP → q_embed in R^(3d=768).  BERT is frozen.

    BERT output = 768-dim (same as joint_dim for Client3+TransE).
    MLP: 768 → 512 → 768.
    """

    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        self.bert      = BertModel.from_pretrained(BERT_MODEL)

        # Freeze BERT — only the MLP is trained during QA fine-tuning
        for param in self.bert.parameters():
            param.requires_grad = False

        # MLP: 768 → 512 → 768
        layers, in_dim = [], BERT_DIM
        for h_dim in MLP_HIDDEN_DIMS:
            layers += [nn.Linear(in_dim, h_dim),
                       nn.ReLU(),
                       nn.Dropout(MLP_DROPOUT)]
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, JOINT_DIM))
        self.mlp = nn.Sequential(*layers)

    def forward(self, questions: list, device: torch.device) -> torch.Tensor:
        enc = self.tokenizer(
            questions, return_tensors="pt", padding=True,
            truncation=True, max_length=64
        ).to(device)
        with torch.no_grad():
            out = self.bert(**enc)
        cls = out.last_hidden_state[:, 0, :]   # (B, 768) — [CLS] token
        return self.mlp(cls)                    # (B, 768)


class FedVServer(nn.Module):
    """
    Central server — WebQSP Client3 BERT+TransE (3 silos).

    Fusion: h_joint(e) = [h_A(e) || h_B(e) || h_C(e)]  in R^(3×256=768)
    Scoring: cosine similarity between q_final and candidate embeddings.
    Topic anchoring: q_final = q_proj + h_joint[topic(Q)]
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim        = embed_dim
        self.question_encoder = QuestionEncoder()

    def fuse(self, h_a: torch.Tensor,
             h_b: torch.Tensor,
             h_c: torch.Tensor) -> torch.Tensor:
        """
        h_a, h_b, h_c : (N, d) each  →  (N, 3d=768)
        Row i of the output is the joint embedding of entity i,
        concatenating its representations from all 3 silos.
        """
        return torch.cat([h_a, h_b, h_c], dim=-1)

    def score_candidates(self, q_embed: torch.Tensor,
                         h_joint: torch.Tensor,
                         candidate_ids: torch.Tensor) -> torch.Tensor:
        """
        q_embed      : (B, 3d)
        h_joint      : (N, 3d)
        candidate_ids: (B, K)  — padded with -1 for invalid entries
        Returns sim  : (B, K)
        """
        safe_ids = candidate_ids.clamp(min=0)
        h_cands  = h_joint[safe_ids]                           # (B, K, 3d)
        q_norm   = F.normalize(q_embed, p=2, dim=-1)          # (B, 3d)
        h_norm   = F.normalize(h_cands, p=2, dim=-1)          # (B, K, 3d)
        return (q_norm.unsqueeze(1) * h_norm).sum(dim=-1)     # (B, K)

    def ranking_loss(self, sim: torch.Tensor,
                     answer_ids_batch: list,
                     candidate_ids: torch.Tensor,
                     margin: float = 1.0) -> torch.Tensor:
        """
        Margin ranking loss: max(0, margin + score_hard_neg - score_best_pos)
        Averaged over the batch.
        """
        device = sim.device
        losses = []
        for i, answer_ids in enumerate(answer_ids_batch):
            cands      = candidate_ids[i]
            valid_mask = cands >= 0
            cand_list  = cands[valid_mask].tolist()
            scores     = sim[i][valid_mask]
            answer_set = set(answer_ids)

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

    def forward(self, questions: list,
                topic_ids: torch.Tensor,
                h_a: torch.Tensor,
                h_b: torch.Tensor,
                h_c: torch.Tensor,
                answer_ids_batch: list,
                candidate_ids: torch.Tensor,
                device: torch.device,
                margin: float = 1.0):
        """Full forward pass: encode → fuse → anchor → score → loss."""
        h_joint  = self.fuse(h_a, h_b, h_c)         # (N, 768)
        q_proj   = self.question_encoder(questions, device)  # (B, 768)

        if USE_TOPIC_ANCHORING:
            q_final = q_proj + h_joint[topic_ids]    # anchor to topic entity
        else:
            q_final = q_proj

        candidate_ids = candidate_ids.to(device)
        sim  = self.score_candidates(q_final, h_joint, candidate_ids)
        loss = self.ranking_loss(sim, answer_ids_batch, candidate_ids, margin)
        return loss, sim
