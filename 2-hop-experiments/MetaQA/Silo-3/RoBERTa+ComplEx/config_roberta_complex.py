# config_roberta_complex.py — Hyperparameters for FedV-KGQA (ComplEx + RoBERTa)
# All parameters identical to all other experiments for fair comparison.
# Changes vs Bert+ComplEx         : CHECKPOINT_DIR, BERT_MODEL → roberta-base
# Changes vs DistilBert+ComplEx   : BERT_MODEL → roberta-base (larger, stronger)
# Changes vs RoBERTa+TransE/DM    : ComplEx uses 2d entity embeddings →
#                                   ENTITY_DIM=512, JOINT_DIM=1536
# Absolute paths used to avoid BASE_DIR resolution issues.

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR       = "/home/islamm9/ISWC/Dataset/Client3/data"
CHECKPOINT_DIR = "/home/islamm9/ISWC/Client3/RoBERTa+ComplEx/models"

KB_ORIGINAL = DATA_DIR + "/kb.txt"

SILO_A_KB = DATA_DIR + "/silos/kb_silo_a.txt"
SILO_B_KB = DATA_DIR + "/silos/kb_silo_b.txt"
SILO_C_KB = DATA_DIR + "/silos/kb_silo_c.txt"

QA_TRAIN  = DATA_DIR + "/qa/2-hop/qa_train.txt"
QA_DEV    = DATA_DIR + "/qa/2-hop/qa_dev.txt"
QA_TEST   = DATA_DIR + "/qa/2-hop/qa_test.txt"

# ── ComplEx KGE ────────────────────────────────────────────────────────────────
# Scoring (Hermitian dot product):
#   φ(h, r, t) = Re(<h, r, conj(t)>)
#              = Σ h_re·r_re·t_re + h_re·r_im·t_im
#                  + h_im·r_re·t_im − h_im·r_im·t_re
#
# Entities h, t ∈ ℂᵈ  → stored as 2d real floats [re | im]
# Relations r   ∈ ℂᵈ  → stored as 2d real floats [re | im]  — stay in silo
#
# IMPORTANT: entity embedding dim sent to server = 2 * KGE_EMBED_DIM = 512
#            h_joint = [h_A || h_B || h_C] ∈ ℝ^(3 × 2d) = ℝ^1536
#            MLP output and q_embed are also 1536-dim
#
# ComplEx-specific: weight_decay=1e-6 on the KGE Adam optimizer (standard)
KGE_EMBED_DIM    = 256     # complex dim d; real entity storage = 2d = 512 per silo
KGE_MARGIN       = 1.0
KGE_NORM         = 2       # kept for API compatibility, unused by ComplEx scoring
KGE_LR           = 1e-3
KGE_EPOCHS       = 100
KGE_BATCH_SIZE   = 512
KGE_NEG_SAMPLES  = 10
KGE_WEIGHT_DECAY = 1e-6    # L2 regularisation on KGE Adam — standard for ComplEx

# Derived dimensions — used by server and evaluate modules
ENTITY_DIM = 2 * KGE_EMBED_DIM   # 512  (real storage per silo per entity)
JOINT_DIM  = 3 * ENTITY_DIM      # 1536 (full h_joint dimension)

# ── Question Encoder (RoBERTa + MLP) ───────────────────────────────────────────
# RoBERTa-base specs:
#   - 12 transformer layers   (same as BERT-base)
#   - ~125M parameters        (BERT-base: ~110M, DistilBERT: ~66M)
#   - 768-dim hidden output   (same as BERT-base → MLP input dim unchanged)
#   - BPE tokenizer, no token_type_ids, <s> token as sequence representation
#
# MLP projects: 768 → 512 → 1536  (output = JOINT_DIM = 6d)
BERT_MODEL      = "roberta-base"
ROBERTA_DIM     = 768          # hidden size — same as BERT-base
MLP_HIDDEN_DIMS = [768, 512]   # intermediate dims; final layer → JOINT_DIM
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
DEVICE = "cuda:1"
