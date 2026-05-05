# src/complex_model.py — ComplEx Knowledge Graph Embedding model

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplEx(nn.Module):
    """
    ComplEx embedding model.

    Scoring function (Hermitian dot product):
        phi(h, r, t) = Re(<h, r, conj(t)>)
                     = Re(sum(h * r * conj(t)))
                     = sum(h_re*r_re*t_re + h_re*r_im*t_im
                           + h_im*r_re*t_im - h_im*r_im*t_re)

    Entities h, t ∈ ℂᵈ  (stored as 2d reals: [re | im])
    Relations r  ∈ ℂᵈ  (stored as 2d reals: [re | im]) — stay in silo

    Key advantage over DistMult: handles anti-symmetric relations
    (e.g. directed_by vs directed) which are critical for the
    OWL inverseOf axioms in this pipeline.

    Only entity embeddings h(e) ∈ ℝ^(2d) are sent to the server.
    Relation embeddings stay permanently in the silo.
    """

    def __init__(self, num_entities, num_relations, embed_dim, norm=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.norm      = norm   # kept for API compatibility

        # Entity embeddings: real + imaginary → (N, 2*embed_dim)
        self.ent_embed = nn.Embedding(num_entities, 2 * embed_dim)

        # Relation embeddings: real + imaginary → (R, 2*embed_dim)
        # NEVER leave the silo
        self.rel_embed = nn.Embedding(num_relations, 2 * embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.ent_embed.weight)
        nn.init.xavier_uniform_(self.rel_embed.weight)

    def score(self, h_ids, r_ids, t_ids):
        """
        Compute ComplEx scores for a batch of triples.

        Score = Re(<h, r, conj(t)>)   (higher = more plausible)

        Args:
            h_ids, r_ids, t_ids : LongTensor (B,)
        Returns:
            scores : FloatTensor (B,)
        """
        d = self.embed_dim

        h = self.ent_embed(h_ids)           # (B, 2d)
        r = self.rel_embed(r_ids)           # (B, 2d)
        t = self.ent_embed(t_ids)           # (B, 2d)

        h_re, h_im = h[:, :d], h[:, d:]    # (B, d) each
        r_re, r_im = r[:, :d], r[:, d:]
        t_re, t_im = t[:, :d], t[:, d:]

        # Re(<h, r, conj(t)>) = Re(h * r * conj(t))
        # conj(t) = t_re - i*t_im
        score = (h_re * r_re * t_re
               + h_re * r_im * t_im
               + h_im * r_re * t_im
               - h_im * r_im * t_re).sum(dim=-1)   # (B,)

        return score

    def margin_ranking_loss(self, h_ids, r_ids, t_pos_ids, t_neg_ids,
                            margin=1.0):
        """
        Margin ranking loss:
            L = max(0, γ - score(h,r,t_pos) + score(h,r,t_neg_hard))

        Args:
            h_ids, r_ids, t_pos_ids : LongTensor (B,)
            t_neg_ids               : LongTensor (B, K)
            margin                  : float γ
        Returns:
            scalar loss
        """
        pos_scores = self.score(h_ids, r_ids, t_pos_ids)   # (B,)

        B, K = t_neg_ids.shape
        h_exp      = h_ids.unsqueeze(1).expand(B, K).reshape(-1)
        r_exp      = r_ids.unsqueeze(1).expand(B, K).reshape(-1)
        t_neg_flat = t_neg_ids.reshape(-1)

        neg_scores = self.score(h_exp, r_exp, t_neg_flat).reshape(B, K)
        hard_neg_scores, _ = neg_scores.max(dim=1)

        loss = F.relu(margin - pos_scores + hard_neg_scores).mean()
        return loss

    def get_entity_embeddings(self, entity_ids=None):
        """
        Return entity embeddings (2d real) to send to the server.
        If entity_ids is None, return all entity embeddings.

        Returns:
            h : FloatTensor (N, 2*embed_dim)
        """
        if entity_ids is None:
            return self.ent_embed.weight
        return self.ent_embed(entity_ids)
