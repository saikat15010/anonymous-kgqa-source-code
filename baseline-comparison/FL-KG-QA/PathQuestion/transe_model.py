# transe_model.py — TransE Knowledge Graph Embedding model

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransE(nn.Module):
    """
    TransE embedding model.

    Scoring function:
        φ(h, r, t) = -|| h + r - t ||_p

    The idea: a relation r is modelled as a translation in embedding space,
    so the correct tail t should be close to h + r.

    Higher score = more plausible triple (we negate the distance).

    Only entity embeddings h(e) ∈ ℝᵈ are sent to the server.
    Relation embeddings r ∈ ℝᵈ stay permanently in the silo.
    """

    def __init__(self, num_entities, num_relations, embed_dim, norm=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.norm      = norm

        # Entity embeddings — sent to server during federated training
        self.ent_embed = nn.Embedding(num_entities, embed_dim)

        # Relation embeddings — NEVER leave the silo
        self.rel_embed = nn.Embedding(num_relations, embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.ent_embed.weight)
        nn.init.xavier_uniform_(self.rel_embed.weight)
        # Normalize entity embeddings to unit sphere (TransE convention)
        with torch.no_grad():
            self.ent_embed.weight.data = F.normalize(
                self.ent_embed.weight.data, p=2, dim=-1
            )

    def score(self, h_ids, r_ids, t_ids):
        """
        Compute TransE scores for a batch of triples.

        Score = -|| h + r - t ||_p   (higher = more plausible)

        Args:
            h_ids, r_ids, t_ids : LongTensor (B,)
        Returns:
            scores : FloatTensor (B,)
        """
        h = self.ent_embed(h_ids)   # (B, d)
        r = self.rel_embed(r_ids)   # (B, d)
        t = self.ent_embed(t_ids)   # (B, d)

        score = -torch.norm(h + r - t, p=self.norm, dim=-1)   # (B,)
        return score

    def margin_ranking_loss(self, h_ids, r_ids, t_pos_ids, t_neg_ids, margin=1.0):
        """
        Margin ranking loss:
            L = max(0, γ - score(h,r,t_pos) + score(h,r,t_neg_hard))
              = max(0, γ + ||h+r-t_pos|| - ||h+r-t_neg_hard||)

        Args:
            h_ids, r_ids, t_pos_ids : LongTensor (B,)
            t_neg_ids               : LongTensor (B, K)
            margin                  : float γ
        Returns:
            scalar loss
        """
        pos_scores = self.score(h_ids, r_ids, t_pos_ids)   # (B,)

        B, K = t_neg_ids.shape
        h_exp = h_ids.unsqueeze(1).expand(B, K).reshape(-1)
        r_exp = r_ids.unsqueeze(1).expand(B, K).reshape(-1)
        t_neg_flat = t_neg_ids.reshape(-1)

        neg_scores = self.score(h_exp, r_exp, t_neg_flat).reshape(B, K)  # (B, K)
        hard_neg_scores, _ = neg_scores.max(dim=1)   # (B,) hardest negatives

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
