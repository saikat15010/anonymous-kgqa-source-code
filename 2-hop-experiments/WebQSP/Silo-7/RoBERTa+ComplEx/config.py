# config.py — All hyperparameters for FedV-KGQA
# Dataset : WebQSP | Client7 | 7 silos | 1-2 hop | Freebase
# KGE     : ComplEx
# Encoder : RoBERTa (roberta-base)
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
# RoBERTa: 12 layers, ~125M params, 768-dim output.
# NO token_type_ids — never pass them to RoBERTa.
# Uses RobertaTokenizer / RobertaModel.
#
# ComplEx entity_dim = 2d = 512  (complex Re||Im stored as reals)
# joint_dim = 7 * 2d = 7 * 512 = 3584   ← 7 silos × 512
# RoBERTa output = 768-dim  →  MLP: 768 → 512 → 3584
#
# ComplEx vs RotatE (both entity_dim=512, joint_dim=3584 in Client7):
#   - Relation dim: ComplEx=2d=512, RotatE=d=256 (phase angles)
#   - Scoring: Re(<h,r,conj(t)>) vs -||h∘e^ir - t||
#   - Adam weight_decay=1e-6 for KGE phase only (unique to ComplEx)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR       = "/home/islamm9/ISWC/Dataset/WebQSP/data/Client7"
CHECKPOINT_DIR = "/home/islamm9/ISWC/WebQSP/Client7/RoBERTa+ComplEx/models"

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

# ── ComplEx KGE ────────────────────────────────────────────────────────────────
# KGE_EMBED_DIM = d. Real storage: 2d per entity AND 2d per relation.
KGE_EMBED_DIM   = 128     
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
# joint_dim = 7 * 2d = 3584  →  MLP: 768 → 512 → 3584
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
