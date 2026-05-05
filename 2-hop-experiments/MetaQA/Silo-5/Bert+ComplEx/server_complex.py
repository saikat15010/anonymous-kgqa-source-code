# server_complex.py — Federated server for BERT + ComplEx | 5 silos | Client5
#
# Differences from Client3 server_complex.py (3 silos):
#   - fuse() takes 5 args (h_a, h_b, h_c, h_d, h_e)
#   - JOINT_DIM = 5 * ENTITY_DIM = 5 * 512 = 2560  (not 3 * 512 = 1536)
#   - MLP output = 2560
#   - forward() signature takes h_a, h_b, h_c, h_d, h_e
#
# ComplEx entity embeddings are 2d-dim (complex [re|im]):
#   entity_dim = 512 per silo
#   joint_dim  = 2560 for 5 silos
# Everything else identical to Client3.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

from config_complex import (
    BERT_MODEL, MLP_HIDDEN_DIMS, MLP_DROPOUT,
    KGE_EMBED_DIM, USE_TOPIC_ANCHORING, ENTITY_DIM, JOINT_DIM
)


class QuestionEncoder(nn.Module):
    """
    BERT [CLS] → MLP → q_embed ∈ ℝ^2560

    ComplEx entity embeddings ∈ ℝ^512 (= 2×256 [re|im]) per silo.
    Joint: h_joint = [h_A||h_B||h_C||h_D||h_E] ∈ ℝ^2560
    MLP: 768 → 512 → 2560
    BERT is frozen; only MLP trains.
    """

    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        self.bert      = BertModel.from_pretrained(BERT_MODEL)

        for param in self.bert.parameters():
            param.requires_grad = False

        layers, in_dim = [], 768
        for h_dim in MLP_HIDDEN_DIMS:
            layers += [nn.Linear(in_dim, h_dim), nn.ReLU(),
                       nn.Dropout(MLP_DROPOUT)]
            in_dim = h_dim
        layers += [nn.Linear(in_dim, JOINT_DIM)]   # → 2560
        self.mlp = nn.Sequential(*layers)

    def forward(self, questions, device):
        enc = self.tokenizer(
            questions, return_tensors="pt", padding=True,
            truncation=True, max_length=64
        ).to(device)
        with torch.no_grad():
            out = self.bert(**enc)
        cls = out.last_hidden_state[:, 0, :]   # (B, 768)
        return self.mlp(cls)                   # (B, 2560)


class FedVServer(nn.Module):
    """
    Federated server for BERT + ComplEx | 5 silos | Client5.

    Fusion:
        h_joint(e) = [h_A||h_B||h_C||h_D||h_E] ∈ ℝ^2560

    Gradient splitting:
        ∂L/∂h_joint → 5 slices of (N, 512) — one per silo.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim        = embed_dim
        self.question_encoder = QuestionEncoder()

    def fuse(self, h_a, h_b, h_c, h_d, h_e):
        """
        h_a … h_e : (N, 512)
        returns    : (N, 2560)
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
        h_a … h_e : (N, 512) — ComplEx entity embeddings [re|im]
        h_joint   : (N, 2560)
        q_embed   : (B, 2560)
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
