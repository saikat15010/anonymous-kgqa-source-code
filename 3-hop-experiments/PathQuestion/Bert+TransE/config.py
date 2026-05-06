# config.py — All hyperparameters for FedV-KGQA (PQ3H)
# TransE + BERT | 3 silos | 3-hop | Freebase13-enriched KB
#
# Same KB and silo split as PQ2H. Only QA files differ (3-hop paths).
# Candidate filtering extended to 3 hops.

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR       = "/home/islamm9/ISWC/Dataset/PQ3H/data/Client3"
CHECKPOINT_DIR = "/home/islamm9/ISWC/3-hop/PathQuestion/Bert+TransE/models"

SILO_A_KB      = DATA_DIR + "/silos/kb_silo_a.txt"
SILO_B_KB      = DATA_DIR + "/silos/kb_silo_b.txt"
SILO_C_KB      = DATA_DIR + "/silos/kb_silo_c.txt"

QA_TRAIN       = DATA_DIR + "/qa/3-hop/qa_train.txt"
QA_DEV         = DATA_DIR + "/qa/3-hop/qa_dev.txt"
QA_TEST        = DATA_DIR + "/qa/3-hop/qa_test.txt"

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
USE_TOPIC_ANCHORING = True

# ── Candidate filtering ────────────────────────────────────────────────────────
NUM_HOPS           = 3
MAX_NEIGHBORS      = 150
CANDIDATE_HOP1_CAP = 50
CANDIDATE_HOP2_CAP = 20
CANDIDATE_HOP3_CAP = 20

# ── General ────────────────────────────────────────────────────────────────────
SEED   = 42
DEVICE = "cuda:0"
