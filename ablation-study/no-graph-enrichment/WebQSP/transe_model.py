# transe_model.py — TransE Knowledge Graph Embedding model
# WebQSP | Client3 | 3 silos | DistilBERT+TransE
#
# Identical to BERT+TransE transe_model.py — KGE training is encoder-agnostic.
# TransE scoring: phi(h, r, t) = -|| h + r - t ||_2
# entity_dim = d = 256,  joint_dim = 3 * d = 768  for Client3.
# L2 normalise entity embeddings after every gradient step.
#
# Only entity embeddings transmitted to server.
# Relation embeddings stay permanently in each silo.

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransE(nn.Module):
    """
    TransE embedding model.

    Scoring: phi(h, r, t) = -|| h + r - t ||_p

    entity_dim = d = 256,  joint_dim = 3 × 256 = 768  (Client3).
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
        with torch.no_grad():
            self.ent_embed.weight.data = F.normalize(
                self.ent_embed.weight.data, p=2, dim=-1)

    def score(self, h_ids, r_ids, t_ids):
        h = self.ent_embed(h_ids)
        r = self.rel_embed(r_ids)
        t = self.ent_embed(t_ids)
        return -torch.norm(h + r - t, p=self.norm, dim=-1)

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
        if entity_ids is None:
            return self.ent_embed.weight
        return self.ent_embed(entity_ids)
