# config.py — All hyperparameters for FedV-KGQA
# Dataset : WebQSP | Client3 | 3 silos | 1-2 hop | Freebase
# KGE     : DistMult
# Encoder : DistilBERT (distilbert-base-uncased)
#
# DistilBERT vs BERT:
#   - 6 transformer layers (vs 12), ~66M params (vs ~110M)
#   - Same 768-dim hidden output → same joint_dim, same MLP
#   - No token_type_ids — never pass them to DistilBERT
#
# DistMult vs TransE:
#   - Scoring: sum(h * r * t)  NOT  -||h + r - t||
#   - No L2 normalisation of entity embeddings
#   - entity_dim = d = 256, joint_dim = 3d = 768  (same as TransE)
#
# QA_EPOCHS = 100 (same as DistilBERT+TransE)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR       = "/home/islamm9/ISWC/Dataset/WebQSP/data/Client3"
CHECKPOINT_DIR = "/home/islamm9/ISWC/WebQSP/Client3/DistilBert+DistMult/models"

# ── Silo KBs (3 silos — OWL-enriched) ─────────────────────────────────────────
SILO_A_KB = DATA_DIR + "/silos/kb_silo_a_enriched.txt"
SILO_B_KB = DATA_DIR + "/silos/kb_silo_b_enriched.txt"
SILO_C_KB = DATA_DIR + "/silos/kb_silo_c_enriched.txt"

QA_TRAIN  = DATA_DIR + "/qa/qa_train.txt"
QA_DEV    = DATA_DIR + "/qa/qa_dev.txt"
QA_TEST   = DATA_DIR + "/qa/qa_test.txt"

# ── DistMult KGE ───────────────────────────────────────────────────────────────
# DistMult entity embeddings are d-dimensional reals (same dim as TransE).
# NOTE: DistMult does NOT use KGE_NORM — kept for API compatibility only.
KGE_EMBED_DIM   = 256
KGE_MARGIN      = 1.0
KGE_NORM        = 2       # unused by DistMult; kept for API compatibility
KGE_LR          = 1e-3
KGE_EPOCHS      = 100
KGE_BATCH_SIZE  = 512
KGE_NEG_SAMPLES = 10

# ── Question Encoder (DistilBERT + MLP) ────────────────────────────────────────
# distilbert-base-uncased: 6 layers, 768-dim hidden output.
# [CLS] token (index 0) of last_hidden_state used as sequence representation.
# IMPORTANT: DistilBERT does NOT use token_type_ids — never pass them.
# joint_dim = 3 * 256 = 768  →  MLP: 768 → 512 → 768
DISTILBERT_MODEL = "distilbert-base-uncased"
DISTILBERT_DIM   = 768
MLP_HIDDEN_DIMS  = [768, 512]
MLP_DROPOUT      = 0.1

# ── Federated QA Training ──────────────────────────────────────────────────────
QA_LR           = 1e-4
QA_EPOCHS       = 100
QA_BATCH_SIZE   = 64
QA_MARGIN       = 1.0

# ── Topic anchoring ────────────────────────────────────────────────────────────
USE_TOPIC_ANCHORING = True

# ── Candidate filtering ────────────────────────────────────────────────────────
MAX_NEIGHBORS      = 200
CANDIDATE_HOP1_CAP = 100
CANDIDATE_HOP2_CAP = 30

# ── General ────────────────────────────────────────────────────────────────────
SEED   = 42
DEVICE = "cuda:1"
