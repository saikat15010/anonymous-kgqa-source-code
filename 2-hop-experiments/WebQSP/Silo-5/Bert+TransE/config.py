# config.py — All hyperparameters for FedV-KGQA
# Dataset : WebQSP | Client5 | 5 silos | 1-2 hop | Freebase
# KGE     : TransE
# Encoder : BERT (bert-base-uncased)
#
# Client5 has 5 silos (vs 3 in Client3):
#   Silo A — People & Biography
#   Silo B — Medicine, Biology & Science
#   Silo C — Places & Geography
#   Silo D — Organisations & Society
#   Silo E — Arts, Sports & Entertainment
#
# TransE entity_dim = d = 256
# joint_dim = 5 * d = 1280  (5 silos × 256)   ← KEY difference from Client3
# BERT output = 768-dim  →  MLP: 768 → 512 → 1280

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR       = "/home/islamm9/ISWC/Dataset/WebQSP/data/Client5"
CHECKPOINT_DIR = "/home/islamm9/ISWC/WebQSP/Client5/Bert+TransE/models"

# ── Silo KBs (5 silos — OWL-enriched) ─────────────────────────────────────────
SILO_A_KB = DATA_DIR + "/silos/kb_silo_a_enriched.txt"
SILO_B_KB = DATA_DIR + "/silos/kb_silo_b_enriched.txt"
SILO_C_KB = DATA_DIR + "/silos/kb_silo_c_enriched.txt"
SILO_D_KB = DATA_DIR + "/silos/kb_silo_d_enriched.txt"
SILO_E_KB = DATA_DIR + "/silos/kb_silo_e_enriched.txt"

QA_TRAIN  = DATA_DIR + "/qa/qa_train.txt"
QA_DEV    = DATA_DIR + "/qa/qa_dev.txt"
QA_TEST   = DATA_DIR + "/qa/qa_test.txt"

# ── TransE KGE ─────────────────────────────────────────────────────────────────
KGE_EMBED_DIM   = 256
KGE_MARGIN      = 1.0
KGE_NORM        = 2
KGE_LR          = 1e-3
KGE_EPOCHS      = 100
KGE_BATCH_SIZE  = 512
KGE_NEG_SAMPLES = 10

# ── Question Encoder (BERT + MLP) ──────────────────────────────────────────────
# bert-base-uncased: 12 layers, ~110M params, 768-dim hidden output.
# token_type_ids ARE used by BERT — always pass them.
# joint_dim = 5 * 256 = 1280  →  MLP: 768 → 512 → 1280
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
MAX_NEIGHBORS      = 200
CANDIDATE_HOP1_CAP = 100
CANDIDATE_HOP2_CAP = 30

# ── General ────────────────────────────────────────────────────────────────────
SEED   = 42
DEVICE = "cuda:1"
