# config_distilbert_distmult.py — Hyperparameters for FedV-KGQA (DistMult + DistilBERT)
# All parameters identical to Bert+TransE / Bert+DistMult for fair comparison.
# Only change from Bert+DistMult: BERT_MODEL → distilbert-base-uncased.
# Absolute paths used to avoid BASE_DIR resolution issues.

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR       = "/home/islamm9/ISWC/Dataset/Client3/data"
CHECKPOINT_DIR = "/home/islamm9/ISWC/Client3/DistilBert+DistMult/models"

KB_ORIGINAL = DATA_DIR + "/kb.txt"

SILO_A_KB = DATA_DIR + "/silos/kb_silo_a.txt"
SILO_B_KB = DATA_DIR + "/silos/kb_silo_b.txt"
SILO_C_KB = DATA_DIR + "/silos/kb_silo_c.txt"

QA_TRAIN  = DATA_DIR + "/qa/2-hop/qa_train.txt"
QA_DEV    = DATA_DIR + "/qa/2-hop/qa_dev.txt"
QA_TEST   = DATA_DIR + "/qa/2-hop/qa_test.txt"

# ── DistMult KGE ───────────────────────────────────────────────────────────────
# Identical to all other experiments for fair comparison.
# DistMult scoring: φ(h, r, t) = <h, r, t> = sum(h * r * t)
# Entity embeddings ∈ ℝᵈ  (real-valued, same dim as TransE)
KGE_EMBED_DIM   = 256
KGE_MARGIN      = 1.0
KGE_NORM        = 2       # kept for API compatibility, unused by DistMult scoring
KGE_LR          = 1e-3
KGE_EPOCHS      = 100
KGE_BATCH_SIZE  = 512
KGE_NEG_SAMPLES = 10

# ── Question Encoder (DistilBERT + MLP) ────────────────────────────────────────
# DistilBERT: 6 layers, 66M params, 768-dim hidden (same as BERT-base output dim)
# h_joint = [h_A || h_B || h_C] ∈ ℝ^(3d) = ℝ^768 — same as Bert+DistMult
BERT_MODEL      = "distilbert-base-uncased"
DISTILBERT_DIM  = 768          # hidden size — same as BERT-base
MLP_HIDDEN_DIMS = [768, 512]   # 768 → 512 → 3d=768
MLP_DROPOUT     = 0.1

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
