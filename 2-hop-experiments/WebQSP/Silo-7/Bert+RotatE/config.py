# config.py — All hyperparameters for FedV-KGQA
# Dataset : WebQSP | Client7 | 7 silos | 1-2 hop | Freebase
# KGE     : RotatE
# Encoder : BERT (bert-base-uncased)
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
# RotatE entity_dim = 2d = 512  (complex Re||Im stored as reals)
# joint_dim = 7 * 2d = 7 * 512 = 3584   ← 7 silos × 512
# BERT output = 768-dim  →  MLP: 768 → 512 → 3584
#
# Dimension summary across clients:
#   Client3: joint_dim = 3 × 512 = 1536
#   Client5: joint_dim = 5 × 512 = 2560
#   Client7: joint_dim = 7 × 512 = 3584  ← this file

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR       = "/home/islamm9/ISWC/Dataset/WebQSP/data/Client7"
CHECKPOINT_DIR = "/home/islamm9/ISWC/WebQSP/Client7/Bert+RotatE/models"

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

# ── RotatE KGE ─────────────────────────────────────────────────────────────────
# KGE_EMBED_DIM = d. Real storage per entity = 2d = 512.
# Relations are d phase angles (NOT 2d like ComplEx).
KGE_EMBED_DIM   = 128     
KGE_MARGIN      = 1.0
KGE_NORM        = 2
KGE_LR          = 1e-3
KGE_EPOCHS      = 100
KGE_BATCH_SIZE  = 512
KGE_NEG_SAMPLES = 10

# ── Question Encoder (BERT + MLP) ──────────────────────────────────────────────
# bert-base-uncased: 12 layers, ~110M params, 768-dim hidden output.
# token_type_ids ARE used by BERT — always pass them.
# joint_dim = 7 * 2d = 3584  →  MLP: 768 → 512 → 3584
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
