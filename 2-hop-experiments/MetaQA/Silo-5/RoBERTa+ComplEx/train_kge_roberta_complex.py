# train_kge_roberta_complex.py — Train ComplEx on 5 silos | RoBERTa | Client5
#
# Differences from Client3 (3 silos):
#   - Imports SILO_D_KB, SILO_E_KB, KGE_WEIGHT_DECAY from config_roberta_complex
#   - Builds shared entity index from 5 silos
#   - Trains 5 silos sequentially: silo_a through silo_e
# KGE training is encoder-agnostic — RoBERTa is not involved here.
#
# ComplEx: Adam with weight_decay=1e-6; L2-normalise entity embeddings each step.

import os, json, random, time
from datetime import datetime
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from config_roberta_complex import (
    SILO_A_KB, SILO_B_KB, SILO_C_KB, SILO_D_KB, SILO_E_KB,
    CHECKPOINT_DIR,
    KGE_EMBED_DIM, KGE_MARGIN, KGE_NORM, KGE_LR, KGE_WEIGHT_DECAY,
    KGE_EPOCHS, KGE_BATCH_SIZE, KGE_NEG_SAMPLES, SEED, DEVICE,
    ENTITY_DIM, JOINT_DIM
)
from dataset import build_index, build_shared_entity_index, KGEDataset
from complex_model import ComplEx


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_negatives(batch_size, num_entities, num_neg, device):
    return torch.randint(0, num_entities, (batch_size, num_neg), device=device)


class KGELogger:
    def __init__(self, log_dir, config_dict):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir     = log_dir
        self.txt_path    = os.path.join(log_dir, "kge_train_log.txt")
        self.json_path   = os.path.join(log_dir, "kge_train_log.json")
        self.run_start   = time.time()
        self._silo_start = None
        self._cur        = None

        self.data = {
            "run_id"    : os.path.basename(log_dir),
            "kge_model" : "ComplEx",
            "qa_encoder": "RoBERTa",
            "num_silos" : 5,
            "config"    : config_dict,
            "silos"     : {}
        }

        with open(self.txt_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("  FedV-KGQA  —  KGE Training Log  [ComplEx | 5 silos | Client5]\n")
            f.write(f"  Run ID : {self.data['run_id']}\n")
            f.write(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")
            f.write("Config:\n")
            for k, v in config_dict.items():
                f.write(f"  {k:<28} = {v}\n")
            f.write("\n")
        self._flush()
        print(f"KGE logs → {log_dir}")

    def start_silo(self, silo_name, num_entities, num_relations, num_triples):
        self._cur        = silo_name
        self._silo_start = time.time()
        self.data["silos"][silo_name] = {
            "num_entities"     : num_entities,
            "num_relations"    : num_relations,
            "num_triples"      : num_triples,
            "entity_embed_dim" : ENTITY_DIM,
            "epochs"           : [],
            "final_loss"       : None,
            "training_time_sec": None
        }
        with open(self.txt_path, "a") as f:
            f.write("=" * 70 + "\n")
            f.write(f"  Silo : {silo_name.upper()}\n")
            f.write(f"  Entities       : {num_entities:>10,}\n")
            f.write(f"  Relations      : {num_relations:>10,}\n")
            f.write(f"  Triples        : {num_triples:>10,}\n")
            f.write(f"  Entity emb dim : {ENTITY_DIM:>10,}  (2×{KGE_EMBED_DIM} [re|im])\n")
            f.write("-" * 70 + "\n")
            f.write(f"  {'Epoch':<8} {'Avg Loss':<16} {'Elapsed':>10}\n")
            f.write("-" * 70 + "\n")
        self._flush()

    def log_epoch(self, epoch, avg_loss):
        elapsed = time.time() - self._silo_start
        entry = {"epoch": epoch, "avg_loss": round(avg_loss, 6),
                 "elapsed_sec": round(elapsed, 1)}
        self.data["silos"][self._cur]["epochs"].append(entry)
        self.data["silos"][self._cur]["final_loss"] = round(avg_loss, 6)
        with open(self.txt_path, "a") as f:
            f.write(f"  {epoch:<8} {avg_loss:<16.6f} {elapsed:>9.1f}s\n")
        self._flush()

    def finish_silo(self, silo_name, ckpt_path):
        elapsed = time.time() - self._silo_start
        self.data["silos"][silo_name]["training_time_sec"] = round(elapsed, 1)
        with open(self.txt_path, "a") as f:
            f.write(f"\n  Checkpoint → {ckpt_path}\n")
            f.write(f"  Silo training time: {elapsed:.1f}s\n\n")
        self._flush()

    def finish_all(self):
        total = time.time() - self.run_start
        with open(self.txt_path, "a") as f:
            f.write("=" * 70 + "\n")
            f.write("  All 5 silos complete.\n")
            f.write(f"  Total time : {total/60:.1f} min\n")
            f.write(f"  Finished   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n")
        self._flush()
        print(f"\n  KGE logs saved:\n    {self.txt_path}\n    {self.json_path}")

    def _flush(self):
        with open(self.json_path, "w") as f:
            json.dump(self.data, f, indent=2)


def train_silo(silo_name, kb_path, shared_entity2id, device, logger):
    print(f"\n{'='*60}")
    print(f"  Training ComplEx — {silo_name}  [Client5 | RoBERTa | 5 silos]")
    print(f"  Entity dim: {ENTITY_DIM} (= 2×{KGE_EMBED_DIM})  weight_decay={KGE_WEIGHT_DECAY}")
    print(f"{'='*60}")

    triples, _, relation2id = build_index(kb_path)
    dataset = KGEDataset(triples, shared_entity2id, relation2id)
    loader  = DataLoader(dataset, batch_size=KGE_BATCH_SIZE,
                         shuffle=True, num_workers=4, pin_memory=True)

    num_entities  = len(shared_entity2id)
    num_relations = len(relation2id)
    num_triples   = len(dataset)

    logger.start_silo(silo_name, num_entities, num_relations, num_triples)

    model     = ComplEx(num_entities, num_relations, KGE_EMBED_DIM,
                        norm=KGE_NORM).to(device)
    # ComplEx standard: Adam with L2 weight decay
    optimizer = optim.Adam(model.parameters(), lr=KGE_LR,
                           weight_decay=KGE_WEIGHT_DECAY)

    for epoch in range(1, KGE_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for h, r, t_pos in loader:
            h, r, t_pos = h.to(device), r.to(device), t_pos.to(device)
            t_neg = sample_negatives(h.size(0), num_entities,
                                     KGE_NEG_SAMPLES, device)

            optimizer.zero_grad()
            loss = model.margin_ranking_loss(h, r, t_pos, t_neg,
                                             margin=KGE_MARGIN)
            loss.backward()
            optimizer.step()

            # ComplEx: L2-normalise entity embeddings each step
            with torch.no_grad():
                model.ent_embed.weight.data = F.normalize(
                    model.ent_embed.weight.data, p=2, dim=-1)

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        logger.log_epoch(epoch, avg_loss)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4}/{KGE_EPOCHS}  |  Loss: {avg_loss:.4f}",
                  flush=True)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{silo_name}_complex.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "entity2id"       : shared_entity2id,
        "relation2id"     : relation2id,
        "embed_dim"       : KGE_EMBED_DIM,
        "entity_dim"      : ENTITY_DIM,
        "num_entities"    : num_entities,
        "num_relations"   : num_relations,
    }, ckpt_path)
    print(f"  Saved → {ckpt_path}")
    logger.finish_silo(silo_name, ckpt_path)


def main():
    set_seed(SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"ComplEx entity dim per silo : {ENTITY_DIM}  (= 2×{KGE_EMBED_DIM})")
    print(f"Joint embedding dim (5×2d)  : {JOINT_DIM}")
    print(f"KGE weight decay            : {KGE_WEIGHT_DECAY}")

    run_id  = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(CHECKPOINT_DIR),
                           "logs_roberta_complex", run_id)
    os.makedirs(log_dir, exist_ok=True)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(os.path.join(CHECKPOINT_DIR, "current_run_id.txt"), "w") as f:
        f.write(run_id)

    config_dict = {
        "kge_model"       : "ComplEx",
        "qa_encoder"      : "RoBERTa",
        "num_silos"       : 5,
        "kge_embed_dim"   : KGE_EMBED_DIM,
        "entity_dim"      : ENTITY_DIM,
        "joint_dim"       : JOINT_DIM,
        "kge_lr"          : KGE_LR,
        "kge_weight_decay": KGE_WEIGHT_DECAY,
        "kge_epochs"      : KGE_EPOCHS,
        "kge_batch_size"  : KGE_BATCH_SIZE,
        "kge_neg_samples" : KGE_NEG_SAMPLES,
        "kge_margin"      : KGE_MARGIN,
        "seed"            : SEED,
        "device"          : str(device),
    }
    logger = KGELogger(log_dir, config_dict)

    # Build shared entity index from ALL 5 silos
    _, e2id_a, _ = build_index(SILO_A_KB)
    _, e2id_b, _ = build_index(SILO_B_KB)
    _, e2id_c, _ = build_index(SILO_C_KB)
    _, e2id_d, _ = build_index(SILO_D_KB)
    _, e2id_e, _ = build_index(SILO_E_KB)
    shared_entity2id = build_shared_entity_index(
        e2id_a, e2id_b, e2id_c, e2id_d, e2id_e)
    print(f"\nShared entity vocabulary: {len(shared_entity2id):,}")

    with open(os.path.join(CHECKPOINT_DIR, "shared_entity2id.json"), "w") as f:
        json.dump(shared_entity2id, f)

    train_silo("silo_a", SILO_A_KB, shared_entity2id, device, logger)
    train_silo("silo_b", SILO_B_KB, shared_entity2id, device, logger)
    train_silo("silo_c", SILO_C_KB, shared_entity2id, device, logger)
    train_silo("silo_d", SILO_D_KB, shared_entity2id, device, logger)
    train_silo("silo_e", SILO_E_KB, shared_entity2id, device, logger)

    logger.finish_all()


if __name__ == "__main__":
    main()
