# config_distmult.py — All hyperparameters for FedV-KGQA
# DistMult + BERT | 7 silos | 2-hop | Client7

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR       = "/home/islamm9/ISWC/Dataset/Client7/data"
CHECKPOINT_DIR = "/home/islamm9/ISWC/Client7/Bert+DistMult/models"

KB_ORIGINAL = DATA_DIR + "/kb.txt"

SILO_A_KB = DATA_DIR + "/silos/kb_silo_a.txt"   # Directors
SILO_B_KB = DATA_DIR + "/silos/kb_silo_b.txt"   # Writers
SILO_C_KB = DATA_DIR + "/silos/kb_silo_c.txt"   # Cast
SILO_D_KB = DATA_DIR + "/silos/kb_silo_d.txt"   # Tags
SILO_E_KB = DATA_DIR + "/silos/kb_silo_e.txt"   # Temporal
SILO_F_KB = DATA_DIR + "/silos/kb_silo_f.txt"   # Genre/Language
SILO_G_KB = DATA_DIR + "/silos/kb_silo_g.txt"   # Ratings

QA_TRAIN  = DATA_DIR + "/qa/2-hop/qa_train.txt"
QA_DEV    = DATA_DIR + "/qa/2-hop/qa_dev.txt"
QA_TEST   = DATA_DIR + "/qa/2-hop/qa_test.txt"

# ── DistMult KGE ───────────────────────────────────────────────────────────────
# Scoring: φ(h, r, t) = <h, r, t> = Σ h_i · r_i · t_i  (bilinear diagonal)
# Entity embeddings ∈ ℝᵈ  (real-valued, same dim as TransE)
# All hyperparameters identical to all other experiments for fair comparison.
KGE_EMBED_DIM   = 256
KGE_MARGIN      = 1.0
KGE_NORM        = 2       # kept for API compatibility, unused by DistMult
KGE_LR          = 1e-3
KGE_EPOCHS      = 100
KGE_BATCH_SIZE  = 512
KGE_NEG_SAMPLES = 10

# ── Joint embedding dimension ──────────────────────────────────────────────────
# h_joint = [h_A||h_B||h_C||h_D||h_E||h_F||h_G] ∈ ℝ^(7d) = ℝ^1792
# Client3: 3×256=768  |  Client5: 5×256=1280  |  Client7: 7×256=1792
JOINT_DIM = 7 * KGE_EMBED_DIM   # 1792

# ── Question Encoder (BERT + MLP) ──────────────────────────────────────────────
# MLP: 768 → 512 → 1792  (output = JOINT_DIM)
BERT_MODEL      = "bert-base-uncased"
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
MAX_NEIGHBORS      = 100
CANDIDATE_HOP1_CAP = 50
CANDIDATE_HOP2_CAP = 20

# ── General ────────────────────────────────────────────────────────────────────
SEED   = 42
DEVICE = "cuda"
