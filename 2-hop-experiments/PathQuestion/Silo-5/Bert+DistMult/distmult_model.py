# distmult_model.py — DistMult Knowledge Graph Embedding model (Client5 | PQ2H)

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistMult(nn.Module):
    """
    DistMult embedding model.

    Scoring function:
        phi(h, r, t) = <h, r, t> = sum(h * r * t)

    Entity embeddings are d-dimensional (real), same as TransE.
    joint_dim = 5 * d = 1280 for Client5 (5 silos).

    Only entity embeddings h(e) in R^d are sent to the server.
    Relation embeddings r in R^d stay permanently in the silo.
    """

    def __init__(self, num_entities, num_relations, embed_dim, norm=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.norm      = norm   # kept for API compatibility with TransE

        self.ent_embed = nn.Embedding(num_entities, embed_dim)
        self.rel_embed = nn.Embedding(num_relations, embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.ent_embed.weight)
        nn.init.xavier_uniform_(self.rel_embed.weight)

    def score(self, h_ids, r_ids, t_ids):
        h = self.ent_embed(h_ids)
        r = self.rel_embed(r_ids)
        t = self.ent_embed(t_ids)
        return (h * r * t).sum(dim=-1)

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
        if entity_ids is None:
            return self.ent_embed.weight
        return self.ent_embed(entity_ids)
