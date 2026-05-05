# config.py — All hyperparameters for FedV-KGQA
# Dataset : WebQSP | Client3 | 3 silos | 1-2 hop | Freebase
# KGE     : TransE
# Encoder : RoBERTa (roberta-base)
#
# RoBERTa vs BERT:
#   - Same architecture: 12 layers, ~125M params, 768-dim output
#   - NO token_type_ids — RoBERTa does not use segment embeddings
#   - Uses Byte-Pair Encoding (BPE) tokeniser vs WordPiece
#   - Trained with more data and longer sequences
#   - Uses RobertaTokenizer / RobertaModel (NOT BertTokenizer / BertModel)
#
# RoBERTa vs DistilBERT:
#   - 12 layers (DistilBERT has 6)
#   - ~125M params (DistilBERT has ~66M)
#   - Both have no token_type_ids
#
# TransE entity embeddings are d-dimensional reals (same as BERT+TransE).
# entity_dim = d = 256  →  joint_dim = 3 * d = 768
# RoBERTa output = 768-dim  →  MLP: 768 → 512 → 768
#
# QA_EPOCHS = 100

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR       = "/home/islamm9/ISWC/Dataset/WebQSP/data/Client3"
CHECKPOINT_DIR = "/home/islamm9/ISWC/WebQSP/Client3/RoBERTa+TransE/models"

# ── Silo KBs (3 silos — OWL-enriched) ─────────────────────────────────────────
SILO_A_KB = DATA_DIR + "/silos/kb_silo_a_enriched.txt"
SILO_B_KB = DATA_DIR + "/silos/kb_silo_b_enriched.txt"
SILO_C_KB = DATA_DIR + "/silos/kb_silo_c_enriched.txt"

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

# ── Question Encoder (RoBERTa + MLP) ───────────────────────────────────────────
# roberta-base: 12 layers, ~125M params, 768-dim hidden output.
# [CLS] token (index 0) of last_hidden_state used as sequence representation.
# IMPORTANT: RoBERTa does NOT use token_type_ids — never pass them.
# joint_dim = 3 * 256 = 768  →  MLP: 768 → 512 → 768
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
DEVICE = "cuda:0"
