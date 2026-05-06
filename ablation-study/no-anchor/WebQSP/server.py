# server.py — Federated server for WebQSP Client3 DistilBERT+TransE (3 silos)
#
# TransE entity_dim = d = 256  →  joint_dim = 3 * d = 768
# MLP: 768 → 512 → 768  (same architecture as BERT+TransE)
#
# DistilBERT vs BERT differences implemented here:
#   1. DistilBertTokenizer / DistilBertModel (not BertTokenizer / BertModel)
#   2. NO token_type_ids — DistilBERT does not use segment embeddings.
#      Passing token_type_ids to DistilBERT raises an error.
#   3. [CLS] token is still index 0 in last_hidden_state.
#   4. Same 768-dim output → joint_dim and MLP architecture unchanged.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertModel

from config import (DISTILBERT_MODEL, DISTILBERT_DIM, MLP_HIDDEN_DIMS,
                    MLP_DROPOUT, KGE_EMBED_DIM, USE_TOPIC_ANCHORING)

ENTITY_DIM = KGE_EMBED_DIM        # d = 256
JOINT_DIM  = 3 * ENTITY_DIM       # 3 × 256 = 768


class QuestionEncoder(nn.Module):
    """
    DistilBERT [CLS] → MLP → q_embed in R^(3d=768). DistilBERT frozen.

    Key difference from BERT: no token_type_ids.
    Output dim is 768 — same as BERT-base — so MLP and joint_dim are unchanged.
    MLP: 768 → 512 → 768.
    """

    def __init__(self):
        super().__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained(DISTILBERT_MODEL)
        self.distilbert = DistilBertModel.from_pretrained(DISTILBERT_MODEL)

        # Freeze DistilBERT — only the MLP is trained during QA fine-tuning
        for param in self.distilbert.parameters():
            param.requires_grad = False

        # MLP: 768 → 512 → 768
        layers, in_dim = [], DISTILBERT_DIM
        for h_dim in MLP_HIDDEN_DIMS:
            layers += [nn.Linear(in_dim, h_dim),
                       nn.ReLU(),
                       nn.Dropout(MLP_DROPOUT)]
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, JOINT_DIM))
        self.mlp = nn.Sequential(*layers)

    def forward(self, questions: list, device: torch.device) -> torch.Tensor:
        # DistilBERT tokenizer does not produce token_type_ids —
        # do NOT pass them to the model.
        enc = self.tokenizer(
            questions, return_tensors="pt", padding=True,
            truncation=True, max_length=64
        ).to(device)
        with torch.no_grad():
            out = self.distilbert(**enc)
        # last_hidden_state[:, 0, :] is the [CLS] representation
        cls = out.last_hidden_state[:, 0, :]   # (B, 768)
        return self.mlp(cls)                    # (B, 768)


class FedVServer(nn.Module):
    """
    Central server — WebQSP Client3 DistilBERT+TransE (3 silos).

    Fusion:    h_joint(e) = [h_A(e) || h_B(e) || h_C(e)] in R^(3×256=768)
    Scoring:   cosine similarity between q_final and candidate embeddings.
    Anchoring: q_final = q_proj + h_joint[topic(Q)]
    """

    def __init__(self, embed_dim: int):
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
