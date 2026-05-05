# config.py — All hyperparameters for FedV-KGQA (PQ2H | Client7)
# RotatE + RoBERTa | 7 silos | 2-hop | Freebase13-enriched KB
#
# RotatE entity embeddings are 2*KGE_EMBED_DIM (complex representation).
# entity_dim = 2d = 512  ->  joint_dim = 7 * 2d = 14d = 3584
# RoBERTa output = 768-dim (same as BERT-base) -> MLP dims unchanged.
# Key RoBERTa difference: no token_type_ids; <s> token as CLS equivalent.

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR       = "/home/islamm9/ISWC/Dataset/PQ2H/data/Client7"
CHECKPOINT_DIR = "/home/islamm9/ISWC/PQ2H/Client7/RoBERTa+RotatE/models"

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

# ── RotatE KGE ─────────────────────────────────────────────────────────────────
# All values identical to all other Client7 experiments for fair comparison.
KGE_EMBED_DIM   = 256     # complex dim d; real storage = 2*256 = 512 per entity
KGE_MARGIN      = 1.0
KGE_NORM        = 2
KGE_LR          = 1e-3
KGE_EPOCHS      = 100
KGE_BATCH_SIZE  = 512
KGE_NEG_SAMPLES = 10

# ── Question Encoder (RoBERTa + MLP) ───────────────────────────────────────────
# RoBERTa-base specs:
#   - 12 transformer layers, ~125M parameters
#   - 768-dim hidden output (same as BERT-base -> MLP dims unchanged)
#   - No token_type_ids; BPE tokenizer; <s> token = sequence representation
# MLP output must match joint_dim = 7 * 2d = 3584.
BERT_MODEL      = "roberta-base"
ROBERTA_DIM     = 768
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
