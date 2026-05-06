# config_distilbert_transe.py — Hyperparameters for FedV-KGQA (PQ2H)
# TransE + DistilBERT | 3 silos | 2-hop | Freebase13-enriched KB
# All parameters identical to Bert+TransE (PQ2H) for fair comparison.
# Only change: BERT_MODEL swapped to distilbert-base-uncased.

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR       = "/home/islamm9/ISWC/Dataset/PQ2H/data/Client3"
CHECKPOINT_DIR = "/home/islamm9/ISWC/Ablation/PathQuestion/DistilBert+TransE_no_anchor/models"

SILO_A_KB      = DATA_DIR + "/silos/kb_silo_a.txt"
SILO_B_KB      = DATA_DIR + "/silos/kb_silo_b.txt"
SILO_C_KB      = DATA_DIR + "/silos/kb_silo_c.txt"

QA_TRAIN       = DATA_DIR + "/qa/2-hop/qa_train.txt"
QA_DEV         = DATA_DIR + "/qa/2-hop/qa_dev.txt"
QA_TEST        = DATA_DIR + "/qa/2-hop/qa_test.txt"

# ── TransE KGE ─────────────────────────────────────────────────────────────────
# All values identical to Bert+TransE (PQ2H) for fair comparison
KGE_EMBED_DIM   = 256
KGE_MARGIN      = 1.0
KGE_NORM        = 2
KGE_LR          = 1e-3
KGE_EPOCHS      = 100
KGE_BATCH_SIZE  = 512
KGE_NEG_SAMPLES = 10

# ── Question Encoder (DistilBERT + MLP) ────────────────────────────────────────
# DistilBERT output dim = 768 (same as BERT-base) — MLP dims unchanged
BERT_MODEL      = "distilbert-base-uncased"
DISTILBERT_DIM  = 768   # DistilBERT hidden size (same as BERT-base)
MLP_HIDDEN_DIMS = [768, 512]
MLP_DROPOUT     = 0.1

# ── Federated QA Training ──────────────────────────────────────────────────────
QA_LR           = 1e-4
QA_EPOCHS       = 100
QA_BATCH_SIZE   = 64
QA_MARGIN       = 1.0

# ── Topic anchoring flag ───────────────────────────────────────────────────────
USE_TOPIC_ANCHORING = False

# ── Candidate filtering ────────────────────────────────────────────────────────
MAX_NEIGHBORS      = 100
CANDIDATE_HOP1_CAP = 50
CANDIDATE_HOP2_CAP = 20

# ── General ────────────────────────────────────────────────────────────────────
SEED   = 42
DEVICE = "cuda:1"
