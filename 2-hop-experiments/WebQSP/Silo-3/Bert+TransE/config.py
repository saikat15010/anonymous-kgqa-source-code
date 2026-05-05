# config.py — All hyperparameters for FedV-KGQA
# Dataset : WebQSP | Client3 | 3 silos | 1-2 hop | Freebase
# KGE     : TransE
# Encoder : BERT (bert-base-uncased)
#
# TransE entity embeddings are d-dimensional (real).
# entity_dim = d = 256  →  joint_dim = 3 * d = 768
# BERT output = 768-dim  →  MLP: 768 → 512 → 768
#
# WebQSP-specific candidate settings are larger than MetaQA/PQ2H
# because the KB has 985K entities and high-degree hub nodes.

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR       = "/home/islamm9/ISWC/Dataset/WebQSP/data/Client3"
CHECKPOINT_DIR = "/home/islamm9/ISWC/WebQSP/Client3/Bert+TransE/models"

# ── Silo KBs (3 silos) ─────────────────────────────────────────────────────────
#   Silo A — People, Medicine & Biology
#             people.*, biology.*, medicine.*, celebrities.*, fictional_universe.*
#   Silo B — Places & Organisations
#             location.*, geography.*, organization.*, government.*,
#             education.*, business.*, finance.*, architecture.*,
#             transportation.*, military.*, law.*, cvg.*
#   Silo C — Arts, Sports & Entertainment  (catch-all)
#             film.*, tv.*, music.*, sports.*, book.*, award.*, olympics.*, etc.
SILO_A_KB = DATA_DIR + "/silos/kb_silo_a_enriched.txt"
SILO_B_KB = DATA_DIR + "/silos/kb_silo_b_enriched.txt"
SILO_C_KB = DATA_DIR + "/silos/kb_silo_c_enriched.txt"

QA_TRAIN  = DATA_DIR + "/qa/qa_train.txt"
QA_DEV    = DATA_DIR + "/qa/qa_dev.txt"
QA_TEST   = DATA_DIR + "/qa/qa_test.txt"

# ── TransE KGE ─────────────────────────────────────────────────────────────────
# All values identical to MetaQA and PQ2H experiments for fair comparison.
# TransE entity embeddings are d-dimensional reals (NOT complex).
KGE_EMBED_DIM   = 256     # entity_dim = d = 256
KGE_MARGIN      = 1.0
KGE_NORM        = 2       # L2 norm in TransE scoring
KGE_LR          = 1e-3
KGE_EPOCHS      = 100
KGE_BATCH_SIZE  = 512
KGE_NEG_SAMPLES = 10

# ── Question Encoder (BERT + MLP) ───────────────────────────────────────────────
# BERT-base-uncased: 12 transformer layers, ~110M parameters, 768-dim output.
# [CLS] token (index 0) used as the sequence representation.
# joint_dim = 3 * 256 = 768  →  MLP output must match this exactly.
BERT_MODEL      = "bert-base-uncased"
BERT_DIM        = 768
MLP_HIDDEN_DIMS = [768, 512]
MLP_DROPOUT     = 0.1

# ── Federated QA Training ──────────────────────────────────────────────────────
QA_LR           = 1e-4
QA_EPOCHS       = 100
QA_BATCH_SIZE   = 64
QA_MARGIN       = 1.0

# ── Topic anchoring ────────────────────────────────────────────────────────────
USE_TOPIC_ANCHORING = True

# ── Candidate filtering ────────────────────────────────────────────────────────
# WebQSP has 985K entities and high-degree hub nodes (e.g. major cities with
# 1000+ neighbours). Caps are increased vs MetaQA/PQ2H to maintain good
# answer recall while keeping candidate set manageable.
MAX_NEIGHBORS      = 200   # MetaQA/PQ2H used 100
CANDIDATE_HOP1_CAP = 100   # MetaQA/PQ2H used 50
CANDIDATE_HOP2_CAP = 30    # MetaQA/PQ2H used 20

# ── General ────────────────────────────────────────────────────────────────────
SEED   = 42
DEVICE = "cuda:0"
