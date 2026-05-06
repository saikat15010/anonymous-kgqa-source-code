# transe_model.py — TransE Knowledge Graph Embedding model
# WebQSP | Client3 | 3 silos | BERT+TransE
#
# TransE scoring: phi(h, r, t) = -|| h + r - t ||_2
# Entity embeddings are d-dimensional reals (NOT complex).
# joint_dim = 3 * d = 768 for Client3 (3 silos).
#
# Only entity embeddings are transmitted to the server.
# Relation embeddings stay permanently in each silo.

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransE(nn.Module):
    """
    TransE embedding model.

    Scoring function:
        phi(h, r, t) = -|| h + r - t ||_p

    A true triple (h, r, t) satisfies t ≈ h + r in embedding space.
    Training minimises a margin ranking loss over sampled negatives.

    Entity embeddings (d-dim real) are sent to the server after training.
    Relation embeddings (d-dim real) remain in the silo at all times.

    For Client3 with d=256:
        entity_dim = 256
        joint_dim  = 3 × 256 = 768
    """

    def __init__(self, num_entities: int, num_relations: int,
                 embed_dim: int, norm: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.norm      = norm

        self.ent_embed = nn.Embedding(num_entities, embed_dim)
        self.rel_embed = nn.Embedding(num_relations, embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.ent_embed.weight)
        nn.init.xavier_uniform_(self.rel_embed.weight)
        # L2-normalise entity embeddings at initialisation
        with torch.no_grad():
            self.ent_embed.weight.data = F.normalize(
                self.ent_embed.weight.data, p=2, dim=-1)

    def score(self, h_ids: torch.Tensor, r_ids: torch.Tensor,
              t_ids: torch.Tensor) -> torch.Tensor:
        """Returns TransE score (negative norm) for each triple."""
        h = self.ent_embed(h_ids)   # (B, d)
        r = self.rel_embed(r_ids)   # (B, d)
        t = self.ent_embed(t_ids)   # (B, d)
        return -torch.norm(h + r - t, p=self.norm, dim=-1)

    def margin_ranking_loss(self, h_ids: torch.Tensor,
                            r_ids: torch.Tensor,
                            t_pos_ids: torch.Tensor,
                            t_neg_ids: torch.Tensor,
                            margin: float = 1.0) -> torch.Tensor:
        """
        Hard-negative margin ranking loss.

        For each positive triple we take the hardest negative in the
        current batch (max negative score) and apply:
            L = mean( max(0, margin + score_neg - score_pos) )
        """
        pos_scores = self.score(h_ids, r_ids, t_pos_ids)   # (B,)

        B, K = t_neg_ids.shape
        h_exp      = h_ids.unsqueeze(1).expand(B, K).reshape(-1)
        r_exp      = r_ids.unsqueeze(1).expand(B, K).reshape(-1)
        t_neg_flat = t_neg_ids.reshape(-1)

        neg_scores = self.score(h_exp, r_exp, t_neg_flat).reshape(B, K)
        hard_neg_scores, _ = neg_scores.max(dim=1)   # (B,)

        return F.relu(margin - pos_scores + hard_neg_scores).mean()

    def get_entity_embeddings(self,
                              entity_ids: torch.Tensor = None
                              ) -> torch.Tensor:
        """
        Returns entity embeddings for sharing with the server.
        If entity_ids is None, returns the full embedding matrix.
        """
        if entity_ids is None:
            return self.ent_embed.weight
        return self.ent_embed(entity_ids)
