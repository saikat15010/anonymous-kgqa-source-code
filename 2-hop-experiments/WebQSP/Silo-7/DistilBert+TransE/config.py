# config.py — All hyperparameters for FedV-KGQA
# Dataset : WebQSP | Client7 | 7 silos | 1-2 hop | Freebase
# KGE     : TransE
# Encoder : DistilBERT (distilbert-base-uncased)
#
# Client7 has 7 silos:
#   Silo A — People
#   Silo B — Medicine & Science
#   Silo C — Places
#   Silo D — Government, Law & Military
#   Silo E — Organisations, Business & Education
#   Silo F — Film & TV
#   Silo G — Music, Sports & Books
#
# DistilBERT vs BERT:
#   - 6 transformer layers (vs 12), ~66M params (vs ~110M)
#   - Same 768-dim hidden output → same joint_dim and MLP architecture
#   - NO token_type_ids — never pass them to DistilBERT
#
# TransE entity_dim = d = 256
# joint_dim = 7 * d = 1792  (7 silos × 256)
# DistilBERT output = 768-dim  →  MLP: 768 → 512 → 1792

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR       = "/home/islamm9/ISWC/Dataset/WebQSP/data/Client7"
CHECKPOINT_DIR = "/home/islamm9/ISWC/WebQSP/Client7/DistilBert+TransE/models"

# ── Silo KBs (7 silos — OWL-enriched) ─────────────────────────────────────────
SILO_A_KB = DATA_DIR + "/silos/kb_silo_a_enriched.txt"
SILO_B_KB = DATA_DIR + "/silos/kb_silo_b_enriched.txt"
SILO_C_KB = DATA_DIR + "/silos/kb_silo_c_enriched.txt"
SILO_D_KB = DATA_DIR + "/silos/kb_silo_d_enriched.txt"
SILO_E_KB = DATA_DIR + "/silos/kb_silo_e_enriched.txt"
SILO_F_KB = DATA_DIR + "/silos/kb_silo_f_enriched.txt"
SILO_G_KB = DATA_DIR + "/silos/kb_silo_g_enriched.txt"

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

# ── Question Encoder (DistilBERT + MLP) ────────────────────────────────────────
# distilbert-base-uncased: 6 layers, 768-dim hidden output.
# [CLS] token (index 0) used as sequence representation.
# IMPORTANT: DistilBERT does NOT use token_type_ids — never pass them.
# joint_dim = 7 * 256 = 1792  →  MLP: 768 → 512 → 1792
DISTILBERT_MODEL = "distilbert-base-uncased"
DISTILBERT_DIM   = 768
MLP_HIDDEN_DIMS  = [768, 512]
MLP_DROPOUT      = 0.1

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
DEVICE = "cuda:0"
