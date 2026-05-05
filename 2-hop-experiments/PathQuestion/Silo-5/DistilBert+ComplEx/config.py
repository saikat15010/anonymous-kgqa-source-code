# config.py — All hyperparameters for FedV-KGQA (PQ2H | Client5)
# ComplEx + DistilBERT | 5 silos | 2-hop | Freebase13-enriched KB
#
# ComplEx entity embeddings are 2*KGE_EMBED_DIM (complex representation).
# entity_dim = 2d = 512  ->  joint_dim = 5 * 2d = 10d = 2560
# DistilBERT output = 768-dim (same as BERT-base) -> MLP dims unchanged.

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR       = "/home/islamm9/ISWC/Dataset/PQ2H/data/Client5"
CHECKPOINT_DIR = "/home/islamm9/ISWC/PQ2H/Client5/DistilBert+ComplEx/models"

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

# ── ComplEx KGE ────────────────────────────────────────────────────────────────
# All values identical to all other Client5 experiments for fair comparison.
# ComplEx entity AND relation embeddings are 2*KGE_EMBED_DIM.
KGE_EMBED_DIM   = 256     # complex dim d; real storage = 2*256 = 512 per entity
KGE_MARGIN      = 1.0
KGE_NORM        = 2       # kept for API compatibility
KGE_LR          = 1e-3
KGE_EPOCHS      = 100
KGE_BATCH_SIZE  = 512
KGE_NEG_SAMPLES = 10

# ── Question Encoder (DistilBERT + MLP) ────────────────────────────────────────
# DistilBERT output = 768-dim (same as BERT-base) -> MLP dims unchanged.
# MLP output must match joint_dim = 5 * 2d = 2560.
BERT_MODEL      = "distilbert-base-uncased"
DISTILBERT_DIM  = 768
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
