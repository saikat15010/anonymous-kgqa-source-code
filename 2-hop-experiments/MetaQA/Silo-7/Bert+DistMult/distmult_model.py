# src/distmult_model.py — DistMult Knowledge Graph Embedding model

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistMult(nn.Module):
    """
    DistMult embedding model.

    Scoring function:
        φ(h, r, t) = <h, r, t> = sum(h * r * t)

    A relation r is modelled as a diagonal matrix (element-wise scaling).
    Efficient and effective for symmetric relations.

    Only entity embeddings h(e) ∈ ℝᵈ are sent to the server.
    Relation embeddings r ∈ ℝᵈ stay permanently in the silo.

    Key difference from TransE:
        - No norm constraint on entity embeddings
        - Bilinear scoring instead of translation-based
        - Better for one-to-many and symmetric relations
    """

    def __init__(self, num_entities, num_relations, embed_dim, norm=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.norm      = norm   # kept for API compatibility with TransE

        # Entity embeddings — sent to server during federated training
        self.ent_embed = nn.Embedding(num_entities, embed_dim)

        # Relation embeddings — NEVER leave the silo
        self.rel_embed = nn.Embedding(num_relations, embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.ent_embed.weight)
        nn.init.xavier_uniform_(self.rel_embed.weight)

    def score(self, h_ids, r_ids, t_ids):
        """
        Compute DistMult scores for a batch of triples.

        Score = sum(h * r * t)   (higher = more plausible)

        Args:
            h_ids, r_ids, t_ids : LongTensor (B,)
        Returns:
            scores : FloatTensor (B,)
        """
        h = self.ent_embed(h_ids)   # (B, d)
        r = self.rel_embed(r_ids)   # (B, d)
        t = self.ent_embed(t_ids)   # (B, d)

        return (h * r * t).sum(dim=-1)   # (B,)

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
        hard_neg_scores, _ = neg_scores.max(dim=1)   # hardest negative

        loss = F.relu(margin - pos_scores + hard_neg_scores).mean()
        return loss

    def get_entity_embeddings(self, entity_ids=None):
        """
        Return entity embeddings to send to the server.
        If entity_ids is None, return all entity embeddings.

        Returns:
            h : FloatTensor (N, d)
        """
        if entity_ids is None:
            return self.ent_embed.weight
        return self.ent_embed(entity_ids)
