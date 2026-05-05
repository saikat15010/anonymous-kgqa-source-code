# src/server_roberta_transe.py — Federated server for RoBERTa + TransE experiment
#
# Differences from server.py (Bert+TransE):
#   - RobertaTokenizer + RobertaModel instead of BertTokenizer + BertModel
#   - RoBERTa does NOT use token_type_ids — tokenizer call omits them automatically
#   - Imports from config_roberta_transe
#
# Differences from server_distilbert.py (DistilBert+TransE):
#   - RoBERTa has 12 layers (vs DistilBERT's 6) — same depth as BERT-base
#   - RoBERTa has ~125M params (BERT-base: ~110M, DistilBERT: ~66M)
#   - RoBERTa uses BPE tokenizer (no [CLS]/[SEP] special tokens, uses <s>/</s>)
#   - RoBERTa output: last_hidden_state[:, 0, :] = <s> token = equivalent of [CLS]
#   - RoBERTa hidden size = 768 → h_joint dim, MLP dims, JOINT_DIM all unchanged
#
# Everything else — fusion, cosine scoring, margin ranking loss,
# topic anchoring, gradient splitting — is identical to all other experiments.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel

from config_roberta_transe import (
    BERT_MODEL, MLP_HIDDEN_DIMS, MLP_DROPOUT,
    KGE_EMBED_DIM, USE_TOPIC_ANCHORING, ROBERTA_DIM
)

JOINT_DIM = 3 * KGE_EMBED_DIM   # 3 × 256 = 768


class QuestionEncoder(nn.Module):
    """
    RoBERTa <s> token → MLP → q_embed ∈ ℝ^(3d)

    RoBERTa (Robustly Optimised BERT Pre-training Approach):
      - 12 transformer layers    (same as BERT-base)
      - ~125M parameters         (BERT-base ~110M, DistilBERT ~66M)
      - 768-dim hidden output    (identical to BERT-base → no architecture change)
      - Trained on 160GB text    (BERT: 16GB)
      - Larger batches (8K vs 256), longer sequences, more steps
      - Byte-Pair Encoding (BPE) tokenizer — no token_type_ids, no NSP
      - Typically outperforms BERT-base on GLUE, SQuAD, and QA benchmarks

    Key implementation note:
      - RoBERTa tokenizer does NOT produce token_type_ids
      - The first token <s> (BOS) serves as the sequence representation,
        equivalent to BERT's [CLS] token
      - Access via: out.last_hidden_state[:, 0, :]

    RoBERTa is frozen; only the MLP projection head is trained.
    The output dimension 3d = 768 matches h_joint = [h_A || h_B || h_C].
    """

    def __init__(self):
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(BERT_MODEL)
        self.roberta   = RobertaModel.from_pretrained(BERT_MODEL)

        # Freeze RoBERTa — only the MLP head trains during federated QA
        for param in self.roberta.parameters():
            param.requires_grad = False

        # MLP: 768 → 512 → 768  (matches BERT and DistilBERT versions exactly)
        layers, in_dim = [], ROBERTA_DIM   # 768
        for h_dim in MLP_HIDDEN_DIMS:
            layers += [nn.Linear(in_dim, h_dim), nn.ReLU(),
                       nn.Dropout(MLP_DROPOUT)]
            in_dim = h_dim
        layers += [nn.Linear(in_dim, JOINT_DIM)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, questions, device):
        """
        Args:
            questions : list[str]  — raw question strings
            device    : torch.device
        Returns:
            q_embed   : (B, 3d) = (B, 768)
        """
        enc = self.tokenizer(
            questions, return_tensors="pt", padding=True,
            truncation=True, max_length=64
        ).to(device)
        with torch.no_grad():
            # RoBERTa returns BaseModelOutputWithPoolingAndCrossAttentions
            # last_hidden_state[:, 0, :] = <s> token representation
            # (pooler_output also available but <s> token is standard for RoBERTa)
            out = self.roberta(**enc)
        cls = out.last_hidden_state[:, 0, :]    # (B, 768)
        return self.mlp(cls)                    # (B, 3d)


class FedVServer(nn.Module):
    """
    Federated server for RoBERTa + TransE experiment (3 silos, 2-hop).

    h_joint(e) = [h_A(e) || h_B(e) || h_C(e)] ∈ ℝ^(3d) = ℝ^768

    TransE entity embeddings are d-dim real vectors. The joint embedding
    space is ℝ^768 — identical to Bert+TransE and DistilBert+TransE.
    The only architectural change is the question encoder: RoBERTa instead
    of BERT or DistilBERT.

    Gradient splitting at concat boundary:
        ∂L/∂h_joint → [∂L/∂h_A || ∂L/∂h_B || ∂L/∂h_C]
        each of size (N, 256) — sent to respective silo.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim        = embed_dim
        self.question_encoder = QuestionEncoder()

    # ── Fusion ────────────────────────────────────────────────────────────────

    def fuse(self, h_a, h_b, h_c):
        """
        Concatenate per-silo TransE entity embeddings.
        h_a, h_b, h_c : (N, d) = (N, 256)
        returns        : (N, 3d) = (N, 768)
        """
        return torch.cat([h_a, h_b, h_c], dim=-1)

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score_candidates(self, q_embed, h_joint, candidate_ids):
        """
        Cosine similarity between question vector and candidate entities.

        sim(e) = cos(q_final, h_joint(e))

        q_embed       : (B, 3d)
        h_joint       : (N, 3d)
        candidate_ids : (B, K)   — -1 = padding
        returns sim   : (B, K)
        """
        safe_ids = candidate_ids.clamp(min=0)
        h_cands  = h_joint[safe_ids]                                # (B, K, 3d)
        q_norm   = F.normalize(q_embed, p=2, dim=-1).unsqueeze(1)  # (B, 1, 3d)
        h_norm   = F.normalize(h_cands, p=2, dim=-1)               # (B, K, 3d)
        return (q_norm * h_norm).sum(dim=-1)                        # (B, K)

    # ── Loss ──────────────────────────────────────────────────────────────────

    def ranking_loss(self, sim, answer_ids_batch, candidate_ids, margin=1.0):
        """
        Margin ranking loss with hardest negative mining:
            L = max(0, γ + sim(e⁻) − sim(e⁺))
        """
        device = sim.device
        losses = []

        for i, answer_ids in enumerate(answer_ids_batch):
            cands      = candidate_ids[i]
            valid_mask = cands >= 0
            cand_list  = cands[valid_mask].tolist()
            scores     = sim[i][valid_mask]

            answer_set  = set(answer_ids)
            pos_indices = [j for j, c in enumerate(cand_list)
                           if c in answer_set]
            if not pos_indices:
                continue

            best_pos = scores[pos_indices].max()

            neg_mask = torch.ones(len(cand_list), dtype=torch.bool, device=device)
            for j in pos_indices:
                neg_mask[j] = False
            if neg_mask.sum() == 0:
                continue

            hard_neg = scores[neg_mask].max()
            losses.append(F.relu(margin + hard_neg - best_pos))

        if not losses:
            return torch.tensor(0.0, requires_grad=True, device=device)
        return torch.stack(losses).mean()

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, questions, topic_ids, h_a, h_b, h_c,
                answer_ids_batch, candidate_ids, device, margin=1.0):
        """
        Full forward pass for one training batch.

        h_a, h_b, h_c : (N, d)  = (N, 256) — TransE entity embeddings
        h_joint        : (N, 3d) = (N, 768) — after fusion
        q_embed        : (B, 3d) = (B, 768) — from RoBERTa + MLP

        Returns:
            loss : scalar margin ranking loss
            sim  : (B, K) cosine similarities over candidate set
        """
        h_joint = self.fuse(h_a, h_b, h_c)                         # (N, 768)
        q_embed = self.question_encoder(questions, device)          # (B, 768)

        if USE_TOPIC_ANCHORING:
            q_final = q_embed + h_joint[topic_ids]
        else:
            q_final = q_embed

        candidate_ids = candidate_ids.to(device)
        sim  = self.score_candidates(q_final, h_joint, candidate_ids)
        loss = self.ranking_loss(sim, answer_ids_batch, candidate_ids, margin)
        return loss, sim
