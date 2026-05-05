# config.py — All hyperparameters for FedV-KGQA
# Dataset : WebQSP | Client5 | 5 silos | 1-2 hop | Freebase
# KGE     : ComplEx
# Encoder : BERT (bert-base-uncased)
#
# Client5 has 5 silos:
#   Silo A — People & Biography
#   Silo B — Medicine, Biology & Science
#   Silo C — Places & Geography
#   Silo D — Organisations & Society
#   Silo E — Arts, Sports & Entertainment
#
# ComplEx entity_dim = 2d = 512  (complex Re||Im stored as reals)
# joint_dim = 5 * 2d = 5 * 512 = 2560   ← 5 silos × 512
# BERT output = 768-dim  →  MLP: 768 → 512 → 2560
#
# ComplEx vs RotatE (both entity_dim=512, joint_dim=2560 in Client5):
#   - Relation dim: ComplEx=2d=512, RotatE=d=256 (phase angles)
#   - Scoring: Re(<h,r,conj(t)>) vs -||h∘e^ir - t||
#   - Adam weight_decay=1e-6 for KGE phase only (unique to ComplEx)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR       = "/home/islamm9/ISWC/Dataset/WebQSP/data/Client5"
CHECKPOINT_DIR = "/home/islamm9/ISWC/WebQSP/Client5/Bert+ComplEx/models"

# ── Silo KBs (5 silos — OWL-enriched) ─────────────────────────────────────────
SILO_A_KB = DATA_DIR + "/silos/kb_silo_a_enriched.txt"
SILO_B_KB = DATA_DIR + "/silos/kb_silo_b_enriched.txt"
SILO_C_KB = DATA_DIR + "/silos/kb_silo_c_enriched.txt"
SILO_D_KB = DATA_DIR + "/silos/kb_silo_d_enriched.txt"
SILO_E_KB = DATA_DIR + "/silos/kb_silo_e_enriched.txt"

QA_TRAIN  = DATA_DIR + "/qa/qa_train.txt"
QA_DEV    = DATA_DIR + "/qa/qa_dev.txt"
QA_TEST   = DATA_DIR + "/qa/qa_test.txt"

# ── ComplEx KGE ────────────────────────────────────────────────────────────────
# KGE_EMBED_DIM = d. Real storage: 2d per entity AND 2d per relation.
KGE_EMBED_DIM   = 128     
KGE_MARGIN      = 1.0
KGE_NORM        = 2       # kept for API compatibility
KGE_LR          = 1e-3
KGE_EPOCHS      = 100
KGE_BATCH_SIZE  = 512
KGE_NEG_SAMPLES = 10

# ── Question Encoder (BERT + MLP) ──────────────────────────────────────────────
# bert-base-uncased: 12 layers, ~110M params, 768-dim hidden output.
# token_type_ids ARE used by BERT — always pass them.
# joint_dim = 5 * 2d = 2560  →  MLP: 768 → 512 → 2560
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
DEVICE = "cuda:0"
