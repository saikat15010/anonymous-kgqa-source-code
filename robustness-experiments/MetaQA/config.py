# config.py — Noise Robustness Experiment
# TransE + BERT | 5 silos | 2-hop | MetaQA | Client5
#
# Same config as the original Client5 Bert+TransE experiment.
# Noise is injected at evaluation time — no retraining needed.

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = "/home/islamm9/ISWC/Dataset/MetaQA/Client5/data"
CHECKPOINT_DIR = "/home/islamm9/ISWC/MetaQA/Client5/Bert+TransE/models"

KB_ORIGINAL = DATA_DIR + "/kb.txt"

SILO_A_KB = DATA_DIR + "/silos/kb_silo_a.txt"
SILO_B_KB = DATA_DIR + "/silos/kb_silo_b.txt"
SILO_C_KB = DATA_DIR + "/silos/kb_silo_c.txt"
SILO_D_KB = DATA_DIR + "/silos/kb_silo_d.txt"
SILO_E_KB = DATA_DIR + "/silos/kb_silo_e.txt"

QA_TRAIN  = DATA_DIR + "/qa/2-hop/qa_train.txt"
QA_DEV    = DATA_DIR + "/qa/2-hop/qa_dev.txt"
QA_TEST   = DATA_DIR + "/qa/2-hop/qa_test.txt"

# ── TransE KGE ─────────────────────────────────────────────────────────────────
KGE_EMBED_DIM   = 256
KGE_MARGIN      = 1.0
KGE_NORM        = 2
KGE_LR          = 1e-3
KGE_EPOCHS      = 100
KGE_BATCH_SIZE  = 512
KGE_NEG_SAMPLES = 10

# ── Joint embedding dimension ──────────────────────────────────────────────────
JOINT_DIM = 5 * KGE_EMBED_DIM   # 1280

# ── Question Encoder (BERT + MLP) ──────────────────────────────────────────────
BERT_MODEL      = "bert-base-uncased"
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
MAX_NEIGHBORS      = 100
CANDIDATE_HOP1_CAP = 50
CANDIDATE_HOP2_CAP = 20

# ── General ────────────────────────────────────────────────────────────────────
SEED   = 42
DEVICE = "cuda:0"

# ── Noise Robustness Settings ──────────────────────────────────────────────────
NOISE_SIGMAS = [0.0, 0.01, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30]
# σ=0.0 is the baseline (no noise)
