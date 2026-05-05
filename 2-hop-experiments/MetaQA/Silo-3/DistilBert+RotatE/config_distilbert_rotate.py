# config_distilbert_rotate.py — Hyperparameters for FedV-KGQA (RotatE + DistilBERT)
# All parameters identical to all other experiments for fair comparison.
# Only changes vs Bert+RotatE:
#   - CHECKPOINT_DIR points to DistilBert+RotatE/models
#   - BERT_MODEL = "distilbert-base-uncased"
# Only changes vs DistilBert+DistMult:
#   - RotatE entity embeddings are 2*KGE_EMBED_DIM (complex representation)
#   - h_joint ∈ ℝ^(6d) = ℝ^1536  (not 3d=768)
# Absolute paths used to avoid BASE_DIR resolution issues.

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR       = "/home/islamm9/ISWC/Dataset/Client3/data"
CHECKPOINT_DIR = "/home/islamm9/ISWC/Client3/DistilBert+RotatE/models"

KB_ORIGINAL = DATA_DIR + "/kb.txt"

SILO_A_KB = DATA_DIR + "/silos/kb_silo_a.txt"
SILO_B_KB = DATA_DIR + "/silos/kb_silo_b.txt"
SILO_C_KB = DATA_DIR + "/silos/kb_silo_c.txt"

QA_TRAIN  = DATA_DIR + "/qa/2-hop/qa_train.txt"
QA_DEV    = DATA_DIR + "/qa/2-hop/qa_dev.txt"
QA_TEST   = DATA_DIR + "/qa/2-hop/qa_test.txt"

# ── RotatE KGE ─────────────────────────────────────────────────────────────────
# Scoring: φ(h, r, t) = -||h ∘ r - t||  (complex element-wise rotation)
# Entities h, t ∈ ℂᵈ  → stored as 2d real floats per entity
# Relations r ∈ ℂᵈ    → stored as d phase angles, unit modulus constraint
#
# IMPORTANT: entity embedding dim sent to server = 2 * KGE_EMBED_DIM = 512
#            h_joint = [h_A || h_B || h_C] ∈ ℝ^(3 × 2d) = ℝ^1536
#            MLP output and q_embed are also 1536-dim
KGE_EMBED_DIM   = 256     # complex dim d; real entity storage = 2d = 512 per silo
KGE_MARGIN      = 1.0
KGE_NORM        = 2       # kept for API compatibility, unused by RotatE scoring
KGE_LR          = 1e-3
KGE_EPOCHS      = 100
KGE_BATCH_SIZE  = 512
KGE_NEG_SAMPLES = 10

# Derived dimensions — used by server and evaluate modules
ENTITY_DIM = 2 * KGE_EMBED_DIM   # 512  (real storage per silo per entity)
JOINT_DIM  = 3 * ENTITY_DIM      # 1536 (full h_joint dimension)

# ── Question Encoder (DistilBERT + MLP) ────────────────────────────────────────
# DistilBERT: 6 layers, 66M params, 768-dim hidden output (same as BERT-base)
# MLP projects: 768 → 512 → 1536  (output matches JOINT_DIM)
BERT_MODEL      = "distilbert-base-uncased"
DISTILBERT_DIM  = 768
MLP_HIDDEN_DIMS = [768, 512]   # intermediate dims; final layer → JOINT_DIM
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
DEVICE = "cuda:1"