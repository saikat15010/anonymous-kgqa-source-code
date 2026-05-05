# config.py — All hyperparameters for FedV-KGQA (PQ2H | Client7)
# TransE + BERT | 7 silos | 2-hop | Freebase13-enriched KB
#
# TransE entity embeddings are d-dimensional (real).
# entity_dim = d = 256  ->  joint_dim = 7 * d = 7d = 1792

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR       = "/home/islamm9/ISWC/Dataset/PQ2H/data/Client7"
CHECKPOINT_DIR = "/home/islamm9/ISWC/PQ2H/Client7/Bert+TransE/models"

# ── Silo KBs (7 silos) ─────────────────────────────────────────────────────────
#   Silo A — Marriage    : spouse
#   Silo B — Lineage     : parents, children
#   Silo C — Vitals      : gender, cause_of_death
#   Silo D — Identity    : ethnicity, religion
#   Silo E — Citizenship : nationality
#   Silo F — Occupation  : profession, institution
#   Silo G — Places      : place_of_birth, place_of_death, location
SILO_A_KB = DATA_DIR + "/silos/kb_silo_a.txt"
SILO_B_KB = DATA_DIR + "/silos/kb_silo_b.txt"
SILO_C_KB = DATA_DIR + "/silos/kb_silo_c.txt"
SILO_D_KB = DATA_DIR + "/silos/kb_silo_d.txt"
SILO_E_KB = DATA_DIR + "/silos/kb_silo_e.txt"
SILO_F_KB = DATA_DIR + "/silos/kb_silo_f.txt"
SILO_G_KB = DATA_DIR + "/silos/kb_silo_g.txt"

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

# ── Question Encoder (BERT + MLP) ──────────────────────────────────────────────
# BERT output = 768. MLP projects to joint_dim = 7 * d = 1792.
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
