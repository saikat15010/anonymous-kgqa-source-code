# complex_model.py — ComplEx Knowledge Graph Embedding model (Client5 | PQ2H)

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplEx(nn.Module):
    """
    ComplEx embedding model.

    Each entity h in C^d stored as 2d reals [Re(h) || Im(h)].
    Each relation r in C^d stored as 2d reals [Re(r) || Im(r)].

    Scoring function (Hermitian dot product):
        phi(h, r, t) = Re(<h, r, conj(t)>)
                     = Re(h)*Re(r)*Re(t)
                     + Re(h)*Im(r)*Im(t)
                     + Im(h)*Re(r)*Im(t)
                     - Im(h)*Im(r)*Re(t)

    Handles symmetric, anti-symmetric, and inverse relation patterns.

    Only entity embeddings (2d reals) are sent to the server.
    Relation embeddings (2d reals) stay permanently in the silo.
    """

    def __init__(self, num_entities, num_relations, embed_dim, norm=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.norm      = norm

        self.ent_embed = nn.Embedding(num_entities, 2 * embed_dim)
        self.rel_embed = nn.Embedding(num_relations, 2 * embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.ent_embed.weight)
        nn.init.xavier_uniform_(self.rel_embed.weight)

    def score(self, h_ids, r_ids, t_ids):
        d = self.embed_dim
        h = self.ent_embed(h_ids)
        r = self.rel_embed(r_ids)
        t = self.ent_embed(t_ids)

        h_re, h_im = h[:, :d], h[:, d:]
        r_re, r_im = r[:, :d], r[:, d:]
        t_re, t_im = t[:, :d], t[:, d:]

        return (h_re * r_re * t_re
                + h_re * r_im * t_im
                + h_im * r_re * t_im
                - h_im * r_im * t_re).sum(dim=-1)

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
