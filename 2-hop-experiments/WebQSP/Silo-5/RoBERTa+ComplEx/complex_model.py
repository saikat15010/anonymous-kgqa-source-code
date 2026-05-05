# complex_model.py — ComplEx Knowledge Graph Embedding model
# WebQSP | Client5 | 5 silos | RoBERTa+ComplEx
#
# KGE training is encoder-agnostic — RoBERTa is not involved here.
#
# Entities h, t in C^d stored as 2d reals [Re(h)||Im(h)].
# Relations r ALSO in C^d stored as 2d reals [Re(r)||Im(r)].
# (Key difference from RotatE: relations are 2d, not d phase angles.)
#
# Scoring: phi(h,r,t) = Re(<h, r, conj(t)>)
#        = Re(h)*Re(r)*Re(t) + Re(h)*Im(r)*Im(t)
#        + Im(h)*Re(r)*Im(t) - Im(h)*Im(r)*Re(t)
#
# entity_dim = 2d = 512.
# joint_dim at server = 5 × 2d = 5 × 512 = 2560  (Client5 — 5 silos).
# Adam weight_decay=1e-6 for KGE phase (unique to ComplEx).
# L2 normalise entity embeddings after every gradient step.

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplEx(nn.Module):
    """
    ComplEx: entities and relations both in C^d stored as 2d reals.

    phi(h,r,t) = Re(<h, r, conj(t)>)
    entity_dim = 2d = 512.
    joint_dim at server = 5 × 2d = 2560  (Client5 — 5 silos).
    """

    def __init__(self, num_entities: int, num_relations: int,
                 embed_dim: int, norm: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.norm      = norm

        self.ent_embed = nn.Embedding(num_entities,  2 * embed_dim)
        self.rel_embed = nn.Embedding(num_relations, 2 * embed_dim)  # 2d NOT d

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.ent_embed.weight)
        nn.init.xavier_uniform_(self.rel_embed.weight)
        with torch.no_grad():
            self.ent_embed.weight.data = F.normalize(
                self.ent_embed.weight.data, p=2, dim=-1)

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

    def margin_ranking_loss(self, h_ids, r_ids, t_pos_ids,
                            t_neg_ids, margin=1.0):
        pos_scores = self.score(h_ids, r_ids, t_pos_ids)
        B, K = t_neg_ids.shape
        h_exp      = h_ids.unsqueeze(1).expand(B, K).reshape(-1)
        r_exp      = r_ids.unsqueeze(1).expand(B, K).reshape(-1)
        t_neg_flat = t_neg_ids.reshape(-1)
        neg_scores = self.score(h_exp, r_exp, t_neg_flat).reshape(B, K)
        hard_neg, _ = neg_scores.max(dim=1)
        return F.relu(margin - pos_scores + hard_neg).mean()

    def get_entity_embeddings(self, entity_ids=None):
        """Returns 2d-dim [Re||Im] embeddings for the server."""
        if entity_ids is None:
            return self.ent_embed.weight
        return self.ent_embed(entity_ids)
