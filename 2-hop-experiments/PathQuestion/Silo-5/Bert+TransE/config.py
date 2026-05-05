# config.py — All hyperparameters for FedV-KGQA (PQ2H | Client5)
# TransE + BERT | 5 silos | 2-hop | Freebase13-enriched KB
# All parameters identical to Client3/Bert+TransE for fair comparison.
# Key difference from Client3: 5 silos → joint_dim = 5d = 1280 (not 3d = 768).

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR       = "/home/islamm9/ISWC/Dataset/PQ2H/data/Client5"
CHECKPOINT_DIR = "/home/islamm9/ISWC/PQ2H/Client5/Bert+TransE/models"

# ── Silo KBs (5 silos) ─────────────────────────────────────────────────────────
#   Silo A — Family      : parents, children, spouse
#   Silo B — Demographics: gender, nationality
#   Silo C — Identity    : ethnicity, religion, cause_of_death
#   Silo D — Occupation  : profession, institution
#   Silo E — Places      : place_of_birth, place_of_death, location
SILO_A_KB = DATA_DIR + "/silos/kb_silo_a.txt"
SILO_B_KB = DATA_DIR + "/silos/kb_silo_b.txt"
SILO_C_KB = DATA_DIR + "/silos/kb_silo_c.txt"
SILO_D_KB = DATA_DIR + "/silos/kb_silo_d.txt"
SILO_E_KB = DATA_DIR + "/silos/kb_silo_e.txt"

QA_TRAIN  = DATA_DIR + "/qa/2-hop/qa_train.txt"
QA_DEV    = DATA_DIR + "/qa/2-hop/qa_dev.txt"
QA_TEST   = DATA_DIR + "/qa/2-hop/qa_test.txt"

# ── TransE KGE ─────────────────────────────────────────────────────────────────
# All values identical to Client3 for fair comparison.
KGE_EMBED_DIM   = 256
KGE_MARGIN      = 1.0
KGE_NORM        = 2
KGE_LR          = 1e-3
KGE_EPOCHS      = 100
KGE_BATCH_SIZE  = 512
KGE_NEG_SAMPLES = 10

# ── Question Encoder (BERT + MLP) ──────────────────────────────────────────────
# BERT output = 768. MLP projects to joint_dim = 5d = 1280.
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
MAX_NEIGHBORS      = 100
CANDIDATE_HOP1_CAP = 50
CANDIDATE_HOP2_CAP = 20

# ── General ────────────────────────────────────────────────────────────────────
SEED   = 42
DEVICE = "cuda:1"
