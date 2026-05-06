# config.py — All hyperparameters for FedV-KGQA
# TransE + BERT | 3 silos | 3-hop | MetaQA | with KB enrichment
#
# BASED ON: 2-hop Bert+TransE config.py
# CHANGES FOR 3-HOP:
#   1. QA paths point to qa/3-hop/ instead of qa/2-hop/
#   2. CANDIDATE_HOP3_CAP added for 3-hop candidate expansion
#   3. NUM_HOPS = 3 (used by dataset.py to select 2-hop or 3-hop expansion)
#   4. CHECKPOINT_DIR points to a separate 3-hop model folder
#
# KB and silo files are UNCHANGED — same OWL-enriched KB as 2-hop.
# KGE checkpoints from 2-hop can be REUSED (same KB → same embeddings).

# ── Paths ──────────────────────────────────────────────────────────────────────
# Absolute paths to avoid BASE_DIR resolution issues.
DATA_DIR       = "/home/islamm9/ISWC/Dataset/MetaQA/Client3/data"
CHECKPOINT_DIR = "/home/islamm9/ISWC/3-hop/MetaQA/Bert+TransE/models"

# Original KB
KB_ORIGINAL    = DATA_DIR + "/kb.txt"

# Enriched silo KBs (SAME as 2-hop — no change)
SILO_A_KB      = DATA_DIR + "/silos/kb_silo_a.txt"
SILO_B_KB      = DATA_DIR + "/silos/kb_silo_b.txt"
SILO_C_KB      = DATA_DIR + "/silos/kb_silo_c.txt"

# QA files — 3-hop (CHANGED from 2-hop)
QA_TRAIN       = DATA_DIR + "/qa/3-hop/qa_train.txt"
QA_DEV         = DATA_DIR + "/qa/3-hop/qa_dev.txt"
QA_TEST        = DATA_DIR + "/qa/3-hop/qa_test.txt"

# ── TransE KGE ─────────────────────────────────────────────────────────────────
# ALL identical to 2-hop for fair comparison
KGE_EMBED_DIM   = 256
KGE_MARGIN      = 1.0
KGE_NORM        = 2
KGE_LR          = 1e-3
KGE_EPOCHS      = 100
KGE_BATCH_SIZE  = 512
KGE_NEG_SAMPLES = 10

# ── Question Encoder (BERT + MLP) ──────────────────────────────────────────────
BERT_MODEL      = "bert-base-uncased"
MLP_HIDDEN_DIMS = [768, 512]
MLP_DROPOUT     = 0.1

# ── Federated QA Training ──────────────────────────────────────────────────────
QA_LR           = 1e-4
QA_EPOCHS       = 30
QA_BATCH_SIZE   = 64
QA_MARGIN       = 1.0

# ── Topic anchoring flag ───────────────────────────────────────────────────────
USE_TOPIC_ANCHORING = True

# ── Candidate filtering ────────────────────────────────────────────────────────
# 3-hop requires expanding one more hop than 2-hop.
# Caps are tighter per hop to keep candidate sets manageable.
NUM_HOPS           = 3          # NEW: controls hop depth in precompute_candidates
MAX_NEIGHBORS      = 100
CANDIDATE_HOP1_CAP = 50         # max hop-1 neighbors to expand from
CANDIDATE_HOP2_CAP = 20         # max hop-2 neighbors per hop-1 node
CANDIDATE_HOP3_CAP = 10         # NEW: max hop-3 neighbors per hop-2 node

# ── General ────────────────────────────────────────────────────────────────────
SEED   = 42
DEVICE = "cuda"
