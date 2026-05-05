# config.py — All hyperparameters for FedV-KGQA
# Dataset : WebQSP | Client3 | 3 silos | 1-2 hop | Freebase
# KGE     : ComplEx
# Encoder : RoBERTa (roberta-base)
#
# RoBERTa: 12 layers, ~125M params, 768-dim output.
# NO token_type_ids — RoBERTa does not use segment embeddings.
# Uses RobertaTokenizer / RobertaModel.
#
# ComplEx entity embeddings are 2d-dimensional (complex: Re||Im stored as reals).
# entity_dim = 2d = 512  →  joint_dim = 3 * 2d = 1536
# Relation embeddings are ALSO 2d (unlike RotatE which uses d phase angles).
# RoBERTa output = 768-dim  →  MLP: 768 → 512 → 1536
#
# ComplEx vs RotatE (both entity_dim=512, joint_dim=1536):
#   - Relation dim: ComplEx=2d=512, RotatE=d=256
#   - Scoring: Re(<h,r,conj(t)>) vs -||h∘e^ir - t||
#   - Adam weight_decay=1e-6 for KGE phase only (unique to ComplEx)
#
# QA_EPOCHS = 100

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR       = "/home/islamm9/ISWC/Dataset/WebQSP/data/Client3"
CHECKPOINT_DIR = "/home/islamm9/ISWC/WebQSP/Client3/RoBERTa+ComplEx/models"

# ── Silo KBs (3 silos — OWL-enriched) ─────────────────────────────────────────
SILO_A_KB = DATA_DIR + "/silos/kb_silo_a_enriched.txt"
SILO_B_KB = DATA_DIR + "/silos/kb_silo_b_enriched.txt"
SILO_C_KB = DATA_DIR + "/silos/kb_silo_c_enriched.txt"

QA_TRAIN  = DATA_DIR + "/qa/qa_train.txt"
QA_DEV    = DATA_DIR + "/qa/qa_dev.txt"
QA_TEST   = DATA_DIR + "/qa/qa_test.txt"

# ── ComplEx KGE ────────────────────────────────────────────────────────────────
# KGE_EMBED_DIM = d. Real storage: 2d per entity AND 2d per relation.
KGE_EMBED_DIM   = 256     # complex dim d; real storage = 2d = 512
KGE_MARGIN      = 1.0
KGE_NORM        = 2       # kept for API compatibility
KGE_LR          = 1e-3
KGE_EPOCHS      = 100
KGE_BATCH_SIZE  = 512
KGE_NEG_SAMPLES = 10

# ── Question Encoder (RoBERTa + MLP) ───────────────────────────────────────────
# roberta-base: 12 layers, ~125M params, 768-dim hidden output.
# [CLS] token (index 0) used as sequence representation.
# IMPORTANT: RoBERTa does NOT use token_type_ids — never pass them.
# joint_dim = 3 * 2d = 1536  →  MLP: 768 → 512 → 1536
ROBERTA_MODEL   = "roberta-base"
ROBERTA_DIM     = 768
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
