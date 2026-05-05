# src/rotate_model.py — RotatE Knowledge Graph Embedding model

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RotatE(nn.Module):
    """
    RotatE embedding model.

    Scoring function:
        phi(h, r, t) = -|| h ∘ r - t ||   (in complex space)

    Entities h, t ∈ ℂᵈ  (represented as 2d real vectors)
    Relations r ∈ ℂᵈ    with |r_i| = 1  (unit modulus constraint)
                         i.e. r_i = e^(i*theta_i)

    Each relation is a rotation in complex space.
    Superior to TransE/DistMult for anti-symmetric and
    compositional relation patterns — important for OWL chain axioms.

    Only entity embeddings h(e) ∈ ℝ^(2d) are sent to the server.
    Relation embeddings r ∈ ℝᵈ (phases) stay permanently in the silo.

    embed_dim here is the COMPLEX dimension d.
    Real storage: entities use 2*embed_dim, relations use embed_dim.
    The server receives 2*embed_dim per entity — same total as other
    models if you set embed_dim = KGE_EMBED_DIM // 2.
    For fair comparison we keep embed_dim = KGE_EMBED_DIM and accept
    that entity vectors are 2*embed_dim — consistent across all silos.
    """

    def __init__(self, num_entities, num_relations, embed_dim, norm=2):
        super().__init__()
        self.embed_dim  = embed_dim
        self.norm       = norm   # kept for API compatibility

        # Entity embeddings in complex space: (N, 2*embed_dim)
        # First embed_dim dims = real part, last embed_dim = imaginary part
        self.ent_embed = nn.Embedding(num_entities, 2 * embed_dim)

        # Relation phase angles: (R, embed_dim)  — NEVER leave the silo
        self.rel_embed = nn.Embedding(num_relations, embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.ent_embed.weight)
        # Init relation phases uniformly in [-pi, pi]
        nn.init.uniform_(self.rel_embed.weight, -math.pi, math.pi)

    def _complex_mul(self, h_re, h_im, r_re, r_im):
        """Complex element-wise multiplication: (h_re + i*h_im)(r_re + i*r_im)"""
        return h_re * r_re - h_im * r_im, h_re * r_im + h_im * r_re

    def score(self, h_ids, r_ids, t_ids):
        """
        Compute RotatE scores for a batch of triples.

        Score = -|| h ∘ r - t ||_2   (higher = more plausible)

        Args:
            h_ids, r_ids, t_ids : LongTensor (B,)
        Returns:
            scores : FloatTensor (B,)
        """
        d = self.embed_dim

        # Entity embeddings split into real/imaginary
        h = self.ent_embed(h_ids)          # (B, 2d)
        t = self.ent_embed(t_ids)          # (B, 2d)
        h_re, h_im = h[:, :d], h[:, d:]   # (B, d) each
        t_re, t_im = t[:, :d], t[:, d:]

        # Relation as unit complex numbers: r = e^(i*theta)
        phase = self.rel_embed(r_ids)      # (B, d)
        r_re  = torch.cos(phase)           # (B, d)
        r_im  = torch.sin(phase)           # (B, d)

        # Rotation: h ∘ r
        hr_re, hr_im = self._complex_mul(h_re, h_im, r_re, r_im)

        # Distance: || h ∘ r - t ||
        diff_re = hr_re - t_re             # (B, d)
        diff_im = hr_im - t_im             # (B, d)
        dist = torch.sqrt(diff_re ** 2 + diff_im ** 2 + 1e-8).sum(dim=-1)  # (B,)

        return -dist   # higher = more plausible

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
