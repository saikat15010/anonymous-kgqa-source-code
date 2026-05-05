# rotate_model.py — RotatE Knowledge Graph Embedding model (Client7 | PQ2H)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RotatE(nn.Module):
    """
    RotatE embedding model.

    Each entity h in C^d stored as 2d reals [Re(h) || Im(h)].
    Each relation r is a phase rotation in [0, 2*pi)^d  (d reals).

    Scoring function:
        phi(h, r, t) = -|| h o r - t ||  (element-wise complex rotation)

    Only entity embeddings (2d reals) are sent to the server.
    Relation embeddings (d reals) stay permanently in the silo.

    joint_dim = 7 * 2d = 14d = 3584 for Client7 (7 silos).
    """

    def __init__(self, num_entities, num_relations, embed_dim, norm=2):
        super().__init__()
        self.embed_dim = embed_dim   # d (complex dim)
        self.norm      = norm

        self.ent_embed = nn.Embedding(num_entities, 2 * embed_dim)
        self.rel_embed = nn.Embedding(num_relations, embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.ent_embed.weight)
        nn.init.uniform_(self.rel_embed.weight, -math.pi, math.pi)

    def _complex_rotate(self, h_ids, r_ids, t_ids):
        h = self.ent_embed(h_ids)
        t = self.ent_embed(t_ids)
        r = self.rel_embed(r_ids)

        d = self.embed_dim
        h_re, h_im = h[:, :d], h[:, d:]
        t_re, t_im = t[:, :d], t[:, d:]

        r_re = torch.cos(r)
        r_im = torch.sin(r)

        rot_re = h_re * r_re - h_im * r_im
        rot_im = h_re * r_im + h_im * r_re

        diff = torch.cat([rot_re - t_re, rot_im - t_im], dim=-1)
        return diff

    def score(self, h_ids, r_ids, t_ids):
        diff = self._complex_rotate(h_ids, r_ids, t_ids)
        return -torch.norm(diff, p=self.norm, dim=-1)

    def margin_ranking_loss(self, h_ids, r_ids, t_pos_ids, t_neg_ids,
                            margin=1.0):
        pos_scores = self.score(h_ids, r_ids, t_pos_ids)

        B, K = t_neg_ids.shape
        h_exp      = h_ids.unsqueeze(1).expand(B, K).reshape(-1)
        r_exp      = r_ids.unsqueeze(1).expand(B, K).reshape(-1)
        t_neg_flat = t_neg_ids.reshape(-1)

        neg_scores = self.score(h_exp, r_exp, t_neg_flat).reshape(B, K)
        hard_neg_scores, _ = neg_scores.max(dim=1)

        return F.relu(margin - pos_scores + hard_neg_scores).mean()

    def get_entity_embeddings(self, entity_ids=None):
        """Returns 2d-dimensional entity embeddings (real + imaginary)."""
        if entity_ids is None:
            return self.ent_embed.weight
        return self.ent_embed(entity_ids)
