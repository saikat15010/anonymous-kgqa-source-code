# config.py — Local-Only Baseline (NO federation, NO fusion)
# DistilBERT + TransE | 3 silos | 2-hop | MetaQA | Client3
#
# Each silo answers independently using only its own embeddings.
# The silo with the most triples involving the topic entity is chosen.
# No cross-silo information sharing.

DATA_DIR       = "/home/islamm9/ISWC/Dataset/MetaQA/Client3/data"
CHECKPOINT_DIR = "/home/islamm9/ISWC/Ablation/LocalOnly/MetaQA/models"

# Use the SAME KGE checkpoints from your federated experiment
# (no retraining — silos trained independently in both settings)
FEDERATED_CHECKPOINT_DIR = "/home/islamm9/ISWC/MetaQA/Client3/DistilBert+TransE/models"

SILO_A_KB = DATA_DIR + "/silos/kb_silo_a.txt"
SILO_B_KB = DATA_DIR + "/silos/kb_silo_b.txt"
SILO_C_KB = DATA_DIR + "/silos/kb_silo_c.txt"

QA_TRAIN  = DATA_DIR + "/qa/2-hop/qa_train.txt"
QA_DEV    = DATA_DIR + "/qa/2-hop/qa_dev.txt"
QA_TEST   = DATA_DIR + "/qa/2-hop/qa_test.txt"

KB_DELIMITER = "|"

KGE_EMBED_DIM   = 256
KGE_NORM        = 2

BERT_MODEL      = "distilbert-base-uncased"
DISTILBERT_DIM  = 768
MLP_HIDDEN_DIMS = [768, 512]
MLP_DROPOUT     = 0.1

# joint_dim = d = 256 (single silo, no fusion)
JOINT_DIM = KGE_EMBED_DIM

QA_LR           = 1e-4
QA_EPOCHS       = 100
QA_BATCH_SIZE   = 64
QA_MARGIN       = 1.0

USE_TOPIC_ANCHORING = True

MAX_NEIGHBORS      = 100
CANDIDATE_HOP1_CAP = 50
CANDIDATE_HOP2_CAP = 20

SEED   = 42
DEVICE = "cuda"
