# config.py — All hyperparameters for FedV-KGQA
# TransE + BERT | 3 silos | 2-hop | with KB enrichment

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR       = os.path.join(BASE_DIR, "data")

# Original KB
KB_ORIGINAL    = os.path.join(DATA_DIR, "kb.txt")

# Enriched silo KBs (output of enrich_kb.py + split_kb.py)
SILO_A_KB      = os.path.join(DATA_DIR, "silos", "kb_silo_a.txt")
SILO_B_KB      = os.path.join(DATA_DIR, "silos", "kb_silo_b.txt")
SILO_C_KB      = os.path.join(DATA_DIR, "silos", "kb_silo_c.txt")

# QA files
QA_TRAIN       = os.path.join(DATA_DIR, "qa", "2-hop", "qa_train.txt")
QA_DEV         = os.path.join(DATA_DIR, "qa", "2-hop", "qa_dev.txt")
QA_TEST        = os.path.join(DATA_DIR, "qa", "2-hop", "qa_test.txt")

# Model checkpoints
CHECKPOINT_DIR = os.path.join(BASE_DIR, "models")

# ── TransE KGE ─────────────────────────────────────────────────────────────────
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
QA_EPOCHS       = 100
QA_BATCH_SIZE   = 64
QA_MARGIN       = 1.0

# ── Topic anchoring flag ───────────────────────────────────────────────────────
# True  → q_final = q_embed + h_joint[topic_id]
# False → q_final = q_embed  (use topic only for candidate filtering)
USE_TOPIC_ANCHORING = True

# ── Candidate filtering ────────────────────────────────────────────────────────
MAX_NEIGHBORS      = 100   # max neighbors per entity in index
CANDIDATE_HOP1_CAP = 50    # max hop-1 neighbors to expand from
CANDIDATE_HOP2_CAP = 20    # max hop-2 neighbors per hop-1 node

# ── General ────────────────────────────────────────────────────────────────────
SEED   = 42
DEVICE = "cuda"
