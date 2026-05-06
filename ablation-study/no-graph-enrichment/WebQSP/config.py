# config.py — Ablation: NO OWL enrichment
# DistilBERT + TransE | 3 silos | 1-2 hop | WebQSP | Client3
# Silos from unenriched KB (kb_silo_a.txt instead of kb_silo_a_enriched.txt)

DATA_DIR       = "/home/islamm9/ISWC/Dataset/WebQSP/data/Client3"
CHECKPOINT_DIR = "/home/islamm9/ISWC/Ablation/WebQSP/DistilBert+TransE_no_owl/models"

# Unenriched silo KBs (no _enriched suffix)
SILO_A_KB = DATA_DIR + "/silos/kb_silo_a.txt"
SILO_B_KB = DATA_DIR + "/silos/kb_silo_b.txt"
SILO_C_KB = DATA_DIR + "/silos/kb_silo_c.txt"

QA_TRAIN  = DATA_DIR + "/qa/qa_train.txt"
QA_DEV    = DATA_DIR + "/qa/qa_dev.txt"
QA_TEST   = DATA_DIR + "/qa/qa_test.txt"

KGE_EMBED_DIM   = 256
KGE_MARGIN      = 1.0
KGE_NORM        = 2
KGE_LR          = 1e-3
KGE_EPOCHS      = 100
KGE_BATCH_SIZE  = 512
KGE_NEG_SAMPLES = 10

BERT_MODEL      = "distilbert-base-uncased"
DISTILBERT_MODEL = BERT_MODEL
DISTILBERT_DIM  = 768
MLP_HIDDEN_DIMS = [768, 512]
MLP_DROPOUT     = 0.1

QA_LR           = 1e-4
QA_EPOCHS       = 100
QA_BATCH_SIZE   = 64
QA_MARGIN       = 1.0

USE_TOPIC_ANCHORING = True

MAX_NEIGHBORS      = 20
CANDIDATE_HOP1_CAP = 10
CANDIDATE_HOP2_CAP = 5

SEED   = 42
DEVICE = "cuda:1"