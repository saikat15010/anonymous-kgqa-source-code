# config_roberta_transe.py — Hyperparameters for FedV-KGQA (TransE + RoBERTa)
# All parameters identical to Bert+TransE / DistilBert+TransE for fair comparison.
# Changes vs Bert+TransE:
#   - CHECKPOINT_DIR → RoBERTa+TransE/models
#   - BERT_MODEL     → roberta-base
#   - ROBERTA_DIM    = 768 (same hidden size as BERT-base)
# h_joint = [h_A || h_B || h_C] ∈ ℝ^(3d) = ℝ^768 — unchanged.
# MLP dims and all QA/KGE hyperparameters are identical.
# Absolute paths used to avoid BASE_DIR resolution issues.

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR       = "/home/islamm9/ISWC/Dataset/Client3/data"
CHECKPOINT_DIR = "/home/islamm9/ISWC/Client3/RoBERTa+TransE/models"

KB_ORIGINAL = DATA_DIR + "/kb.txt"

SILO_A_KB = DATA_DIR + "/silos/kb_silo_a.txt"
SILO_B_KB = DATA_DIR + "/silos/kb_silo_b.txt"
SILO_C_KB = DATA_DIR + "/silos/kb_silo_c.txt"

QA_TRAIN  = DATA_DIR + "/qa/2-hop/qa_train.txt"
QA_DEV    = DATA_DIR + "/qa/2-hop/qa_dev.txt"
QA_TEST   = DATA_DIR + "/qa/2-hop/qa_test.txt"

# ── TransE KGE ─────────────────────────────────────────────────────────────────
# All values identical to all other experiments for fair comparison.
# TransE scoring: φ(h, r, t) = −||h + r − t||₂  (translation-based)
# Entity embeddings ∈ ℝᵈ  (real-valued)
KGE_EMBED_DIM   = 256
KGE_MARGIN      = 1.0
KGE_NORM        = 2
KGE_LR          = 1e-3
KGE_EPOCHS      = 100
KGE_BATCH_SIZE  = 512
KGE_NEG_SAMPLES = 10

# ── Question Encoder (RoBERTa + MLP) ───────────────────────────────────────────
# RoBERTa-base specs:
#   - 12 transformer layers  (same as BERT-base)
#   - ~125M parameters       (BERT-base: ~110M — slightly larger due to no NSP head)
#   - 768-dim hidden output  (same as BERT-base → MLP dims unchanged)
#   - Trained with more data, longer, larger batches, no NSP objective
#   - Uses Byte-Pair Encoding (BPE) tokenizer — no token_type_ids
#   - Typically outperforms BERT-base on downstream NLP tasks
#
# h_joint = [h_A || h_B || h_C] ∈ ℝ^(3d) = ℝ^768 — unchanged from BERT version
BERT_MODEL    = "roberta-base"
ROBERTA_DIM   = 768          # hidden size — same as BERT-base
MLP_HIDDEN_DIMS = [768, 512]   # 768 → 512 → 3d=768
MLP_DROPOUT   = 0.1

# ── Federated QA Training ──────────────────────────────────────────────────────
QA_LR           = 1e-4
QA_EPOCHS       = 100
QA_BATCH_SIZE   = 64
QA_MARGIN       = 1.0

# ── Topic anchoring ────────────────────────────────────────────────────────────
# q_final = q_embed + h_joint[topic_id]
USE_TOPIC_ANCHORING = True

# ── Candidate filtering ────────────────────────────────────────────────────────
MAX_NEIGHBORS      = 100
CANDIDATE_HOP1_CAP = 50
CANDIDATE_HOP2_CAP = 20

# ── General ────────────────────────────────────────────────────────────────────
SEED   = 42
DEVICE = "cuda:0"
