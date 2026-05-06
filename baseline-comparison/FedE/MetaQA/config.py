# config.py — Adapted FedE (VFL) for MetaQA
# FedAvg on entity embeddings, avg pooling, no anchoring

import os

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR       = os.path.join(BASE_DIR, "data")

KB_ORIGINAL    = os.path.join(DATA_DIR, "kb.txt")

SILO_A_KB      = os.path.join(DATA_DIR, "silos", "kb_silo_a.txt")
SILO_B_KB      = os.path.join(DATA_DIR, "silos", "kb_silo_b.txt")
SILO_C_KB      = os.path.join(DATA_DIR, "silos", "kb_silo_c.txt")

QA_TRAIN       = os.path.join(DATA_DIR, "qa", "2-hop", "qa_train.txt")
QA_DEV         = os.path.join(DATA_DIR, "qa", "2-hop", "qa_dev.txt")
QA_TEST        = os.path.join(DATA_DIR, "qa", "2-hop", "qa_test.txt")

CHECKPOINT_DIR = os.path.join(BASE_DIR, "models")

KGE_EMBED_DIM   = 256
KGE_MARGIN      = 1.0
KGE_NORM        = 2
KGE_LR          = 1e-3
KGE_EPOCHS      = 100
KGE_BATCH_SIZE  = 512
KGE_NEG_SAMPLES = 10

BERT_MODEL      = "bert-base-uncased"
MLP_HIDDEN_DIMS = [768, 512]
MLP_DROPOUT     = 0.1

QA_LR           = 1e-4
QA_EPOCHS       = 100
QA_BATCH_SIZE   = 64
QA_MARGIN       = 1.0

USE_TOPIC_ANCHORING = False  # FedE has no anchoring

MAX_NEIGHBORS      = 100
CANDIDATE_HOP1_CAP = 50
CANDIDATE_HOP2_CAP = 20

SEED   = 42
DEVICE = "cuda:0"
