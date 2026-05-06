# config_roberta_transe.py — Hyperparameters for FedV-KGQA
# RoBERTa + TransE | 3 silos | 3-hop | MetaQA | with KB enrichment
#
# BASED ON: 2-hop RoBERTa+TransE config
# CHANGES FOR 3-HOP:
#   1. QA paths point to qa/3-hop/ instead of qa/2-hop/
#   2. CANDIDATE_HOP3_CAP added for 3-hop candidate expansion
#   3. NUM_HOPS = 3
#   4. CHECKPOINT_DIR → 3-hop project folder
#   5. QA_EPOCHS = 30 (for fair comparison with other 3-hop experiments)
#
# Absolute paths to avoid BASE_DIR resolution issues.

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR       = "/home/islamm9/ISWC/Dataset/MetaQA/Client3/data"
CHECKPOINT_DIR = "/home/islamm9/ISWC/3-hop/MetaQA/RoBERTa+TransE/models"

KB_ORIGINAL = DATA_DIR + "/kb.txt"

SILO_A_KB = DATA_DIR + "/silos/kb_silo_a.txt"
SILO_B_KB = DATA_DIR + "/silos/kb_silo_b.txt"
SILO_C_KB = DATA_DIR + "/silos/kb_silo_c.txt"

# QA files — 3-hop
QA_TRAIN  = DATA_DIR + "/qa/3-hop/qa_train.txt"
QA_DEV    = DATA_DIR + "/qa/3-hop/qa_dev.txt"
QA_TEST   = DATA_DIR + "/qa/3-hop/qa_test.txt"

# ── TransE KGE ─────────────────────────────────────────────────────────────────
KGE_EMBED_DIM   = 256
KGE_MARGIN      = 1.0
KGE_NORM        = 2
KGE_LR          = 1e-3
KGE_EPOCHS      = 100
KGE_BATCH_SIZE  = 512
KGE_NEG_SAMPLES = 10

# ── Question Encoder (RoBERTa + MLP) ───────────────────────────────────────────
BERT_MODEL      = "roberta-base"
ROBERTA_DIM     = 768
MLP_HIDDEN_DIMS = [768, 512]
MLP_DROPOUT     = 0.1

# ── Federated QA Training ──────────────────────────────────────────────────────
QA_LR           = 1e-4
QA_EPOCHS       = 30
QA_BATCH_SIZE   = 64
QA_MARGIN       = 1.0

# ── Topic anchoring ────────────────────────────────────────────────────────────
USE_TOPIC_ANCHORING = True

# ── Candidate filtering ────────────────────────────────────────────────────────
NUM_HOPS           = 3
MAX_NEIGHBORS      = 100
CANDIDATE_HOP1_CAP = 50
CANDIDATE_HOP2_CAP = 20
CANDIDATE_HOP3_CAP = 10

# ── General ────────────────────────────────────────────────────────────────────
SEED   = 42
DEVICE = "cuda:0"
