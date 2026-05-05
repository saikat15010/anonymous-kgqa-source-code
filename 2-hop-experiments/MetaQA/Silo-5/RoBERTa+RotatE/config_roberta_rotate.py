# config_roberta_rotate.py — All hyperparameters for FedV-KGQA
# RotatE + RoBERTa | 5 silos | 2-hop | Client5

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR       = "/home/islamm9/ISWC/Dataset/Client5/data"
CHECKPOINT_DIR = "/home/islamm9/ISWC/Client5/RoBERTa+RotatE/models"

KB_ORIGINAL = DATA_DIR + "/kb.txt"

SILO_A_KB = DATA_DIR + "/silos/kb_silo_a.txt"
SILO_B_KB = DATA_DIR + "/silos/kb_silo_b.txt"
SILO_C_KB = DATA_DIR + "/silos/kb_silo_c.txt"
SILO_D_KB = DATA_DIR + "/silos/kb_silo_d.txt"
SILO_E_KB = DATA_DIR + "/silos/kb_silo_e.txt"

QA_TRAIN  = DATA_DIR + "/qa/2-hop/qa_train.txt"
QA_DEV    = DATA_DIR + "/qa/2-hop/qa_dev.txt"
QA_TEST   = DATA_DIR + "/qa/2-hop/qa_test.txt"

# ── RotatE KGE ─────────────────────────────────────────────────────────────────
# Scoring: φ(h, r, t) = −||h ∘ r − t||  (complex rotation distance)
# Entities h, t ∈ ℂᵈ → stored as 2d real floats [re|im]
# Relations r ∈ ℂᵈ → unit modulus via cos/sin — stay in silo
# RotatE: plain Adam (no weight_decay), no post-step entity L2-normalisation
KGE_EMBED_DIM   = 256     # complex dim d; real entity storage = 2*256 = 512
KGE_MARGIN      = 1.0
KGE_NORM        = 2       # kept for API compatibility
KGE_LR          = 1e-3
KGE_EPOCHS      = 100
KGE_BATCH_SIZE  = 512
KGE_NEG_SAMPLES = 10

# ── Joint embedding dimension ──────────────────────────────────────────────────
# RotatE entity dim per silo = 2 * KGE_EMBED_DIM = 512
# h_joint = [h_A || h_B || h_C || h_D || h_E] ∈ ℝ^(5 × 2d) = ℝ^2560
# Client3 used 3 × 2d = 1536; Client5 uses 5 × 2d = 2560
ENTITY_DIM = 2 * KGE_EMBED_DIM   # 512
JOINT_DIM  = 5 * ENTITY_DIM      # 2560

# ── Question Encoder (RoBERTa + MLP) ───────────────────────────────────────────
# RoBERTa-base: 12 layers, ~125M params, 768-dim output, BPE tokenizer
# No token_type_ids; last_hidden_state[:, 0, :] = <s> token
# MLP: 768 → 512 → 2560  (output = JOINT_DIM)
BERT_MODEL      = "roberta-base"
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
MAX_NEIGHBORS      = 100
CANDIDATE_HOP1_CAP = 50
CANDIDATE_HOP2_CAP = 20

# ── General ────────────────────────────────────────────────────────────────────
SEED   = 42
DEVICE = "cuda"
