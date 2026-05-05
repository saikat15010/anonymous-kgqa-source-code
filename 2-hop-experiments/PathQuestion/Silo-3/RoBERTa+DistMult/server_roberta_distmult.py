# server_roberta_distmult.py — Federated server for RoBERTa + DistMult experiment (PQ2H)
#
# DistMult entity embeddings are d-dimensional (real), same as TransE.
# h_joint = [h_A || h_B || h_C] ∈ ℝ^(3d) = ℝ^768
# QuestionEncoder MLP output dim = 3d = 768  (same as RoBERTa+TransE)
#
# Only difference from server_roberta_transe.py:
#   - Imports from config_roberta_distmult instead of config_roberta_transe
# Everything else — RoBERTa encoder, fusion, scoring, loss — is identical.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel

from config_roberta_distmult import (
    BERT_MODEL, MLP_HIDDEN_DIMS, MLP_DROPOUT,
    KGE_EMBED_DIM, USE_TOPIC_ANCHORING, ROBERTA_DIM
)

JOINT_DIM = 3 * KGE_EMBED_DIM   # 3 × 256 = 768


class QuestionEncoder(nn.Module):
    """
    RoBERTa <s> token → MLP → q_embed ∈ ℝ^(3d)

    RoBERTa is frozen; only the MLP projection head is trained.
    Output dimension 3d = 768 matches h_joint = [h_A || h_B || h_C].
    DistMult entity embeddings are d-dimensional (real), so joint dim = 3d.

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
        return self.mlp(cls)                    # (B, 3d=768)


class FedVServer(nn.Module):
    """
    Central server for RoBERTa + DistMult FedV-KGQA (3 silos, PQ2H).
    h_joint(e) = [h_A(e) || h_B(e) || h_C(e)] ∈ ℝ^(3d)
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim        = embed_dim
        self.question_encoder = QuestionEncoder()

    def fuse(self, h_a, h_b, h_c):
        """
        h_a, h_b, h_c : (N, d)
        returns        : (N, 3d)
        """
        return torch.cat([h_a, h_b, h_c], dim=-1)

    def score_candidates(self, q_embed, h_joint, candidate_ids):
        """
        q_embed       : (B, 3d)
        h_joint       : (N, 3d)
        candidate_ids : (B, K)   (-1 = padding)
        returns sim   : (B, K)
        """
        safe_ids = candidate_ids.clamp(min=0)
        h_cands  = h_joint[safe_ids]                               # (B, K, 3d)
        q_norm   = F.normalize(q_embed, p=2, dim=-1).unsqueeze(1)  # (B, 1, 3d)
        h_norm   = F.normalize(h_cands, p=2, dim=-1)               # (B, K, 3d)
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
        h_joint = self.fuse(h_a, h_b, h_c)                        # (N, 3d)
        q_embed = self.question_encoder(questions, device)         # (B, 3d)

        if USE_TOPIC_ANCHORING:
            q_final = q_embed + h_joint[topic_ids]
        else:
            q_final = q_embed

        candidate_ids = candidate_ids.to(device)
        sim  = self.score_candidates(q_final, h_joint, candidate_ids)
        loss = self.ranking_loss(sim, answer_ids_batch,
                                 candidate_ids, margin)
        return loss, sim
