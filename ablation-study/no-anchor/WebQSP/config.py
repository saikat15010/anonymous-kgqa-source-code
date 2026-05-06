# config.py — All hyperparameters for FedV-KGQA
# Dataset : WebQSP | Client3 | 3 silos | 1-2 hop | Freebase
# KGE     : TransE
# Encoder : DistilBERT (distilbert-base-uncased)
#
# DistilBERT vs BERT:
#   - 6 transformer layers (vs 12 for BERT)
#   - Same 768-dim hidden output → same joint_dim, same MLP architecture
#   - No token_type_ids (DistilBERT does not use segment embeddings)
#   - ~40% fewer parameters, ~60% faster than BERT-base
#
# TransE entity embeddings are d-dimensional reals.
# entity_dim = d = 256  →  joint_dim = 3 * d = 768
# DistilBERT output = 768-dim  →  MLP: 768 → 512 → 768
#
# QA_EPOCHS = 100 (increased from 50 used in BERT experiments)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR       = "/home/islamm9/ISWC/Dataset/WebQSP/data/Client3"
CHECKPOINT_DIR = "/home/islamm9/ISWC/Ablation/WebQSP/DistilBert+TransE_no_anchor/models"

# ── Silo KBs (3 silos — OWL-enriched) ─────────────────────────────────────────
SILO_A_KB = DATA_DIR + "/silos/kb_silo_a_enriched.txt"
SILO_B_KB = DATA_DIR + "/silos/kb_silo_b_enriched.txt"
SILO_C_KB = DATA_DIR + "/silos/kb_silo_c_enriched.txt"

QA_TRAIN  = DATA_DIR + "/qa/qa_train.txt"
QA_DEV    = DATA_DIR + "/qa/qa_dev.txt"
QA_TEST   = DATA_DIR + "/qa/qa_test.txt"

# ── TransE KGE ─────────────────────────────────────────────────────────────────
KGE_EMBED_DIM   = 256     # entity_dim = d = 256
KGE_MARGIN      = 1.0
KGE_NORM        = 2
KGE_LR          = 1e-3
KGE_EPOCHS      = 100
KGE_BATCH_SIZE  = 512
KGE_NEG_SAMPLES = 10

# ── Question Encoder (DistilBERT + MLP) ────────────────────────────────────────
# distilbert-base-uncased: 6 layers, ~66M params, 768-dim hidden output.
# [CLS] token (index 0) used as the sequence representation.
# IMPORTANT: DistilBERT does NOT use token_type_ids — never pass them.
# joint_dim = 3 * 256 = 768 → MLP output must be 768 (same as BERT+TransE).
DISTILBERT_MODEL = "distilbert-base-uncased"
DISTILBERT_DIM   = 768        # hidden size — same as BERT-base
MLP_HIDDEN_DIMS  = [768, 512]
MLP_DROPOUT      = 0.1

# ── Federated QA Training ──────────────────────────────────────────────────────
QA_LR           = 1e-4
QA_EPOCHS       = 100     # increased from 50 (BERT experiments) to 100
QA_BATCH_SIZE   = 64
QA_MARGIN       = 1.0

# ── Topic anchoring ────────────────────────────────────────────────────────────
USE_TOPIC_ANCHORING = False

# ── Candidate filtering ────────────────────────────────────────────────────────
MAX_NEIGHBORS      = 200
CANDIDATE_HOP1_CAP = 100
CANDIDATE_HOP2_CAP = 30

# ── General ────────────────────────────────────────────────────────────────────
SEED   = 42
DEVICE = "cuda:1"
