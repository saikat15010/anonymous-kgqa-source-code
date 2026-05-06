# config.py — Centralized Baseline (NO federation)
# DistilBERT + TransE | 1 merged KB | 2-hop | PQ2H

DATA_DIR       = "/home/islamm9/ISWC/Dataset/PQ2H/data/Client3"
CHECKPOINT_DIR = "/home/islamm9/ISWC/Ablation/Centralized/PQ2H/models"

SILO_KB_PATHS = [
    DATA_DIR + "/silos/kb_silo_a.txt",
    DATA_DIR + "/silos/kb_silo_b.txt",
    DATA_DIR + "/silos/kb_silo_c.txt",
]

QA_TRAIN  = DATA_DIR + "/qa/2-hop/qa_train.txt"
QA_DEV    = DATA_DIR + "/qa/2-hop/qa_dev.txt"
QA_TEST   = DATA_DIR + "/qa/2-hop/qa_test.txt"

KB_DELIMITER = "\t"

KGE_EMBED_DIM   = 256
KGE_MARGIN      = 1.0
KGE_NORM        = 2
KGE_LR          = 1e-3
KGE_EPOCHS      = 100
KGE_BATCH_SIZE  = 512
KGE_NEG_SAMPLES = 10

BERT_MODEL      = "distilbert-base-uncased"
DISTILBERT_DIM  = 768
MLP_HIDDEN_DIMS = [768, 512]
MLP_DROPOUT     = 0.1
JOINT_DIM       = KGE_EMBED_DIM   # 256

QA_LR           = 1e-4
QA_EPOCHS       = 100
QA_BATCH_SIZE   = 64
QA_MARGIN       = 1.0

USE_TOPIC_ANCHORING = True

MAX_NEIGHBORS      = 100
CANDIDATE_HOP1_CAP = 50
CANDIDATE_HOP2_CAP = 20

SEED   = 42
DEVICE = "cuda:0"