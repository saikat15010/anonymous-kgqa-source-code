# train_fedv.py — Adapted FedE (VFL) Training
#
# Only change from FedV-KGQA: calls fedavg_entity_embeddings() after each
# training epoch to average entity embeddings across all 3 silos.

import os, json, random, time
from datetime import datetime
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from config import (
    SILO_A_KB, SILO_B_KB, SILO_C_KB,
    CHECKPOINT_DIR, QA_TRAIN, QA_DEV, QA_TEST,
    KGE_EMBED_DIM, KGE_NORM, QA_LR, QA_EPOCHS,
    QA_BATCH_SIZE, QA_MARGIN, SEED, DEVICE,
    MAX_NEIGHBORS, CANDIDATE_HOP1_CAP, CANDIDATE_HOP2_CAP
)
from dataset import (QADataset, build_neighbor_index, precompute_candidates)
from transe_model import TransE
from server import FedVServer, fedavg_entity_embeddings  # ← import FedAvg
from evaluate import evaluate, print_results


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FedVLogger:
    def __init__(self, log_dir, config_dict):
        os.makedirs(log_dir, exist_ok=True)
        self.txt_path  = os.path.join(log_dir, "fedv_train_log.txt")
        self.json_path = os.path.join(log_dir, "fedv_train_log.json")
        self.run_start = time.time()
        self.data = {
            "run_id": os.path.basename(log_dir),
            "config": config_dict,
            "train_loss": [], "dev_metrics": [], "test_metrics": None,
        }
        with open(self.txt_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("  Adapted FedE (VFL) — Federated QA Training Log\n")
            f.write(f"  Run ID : {self.data['run_id']}\n")
            f.write(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")
            for k, v in config_dict.items():
                f.write(f"  {k:<26} = {v}\n")
            f.write("\n" + "-" * 70 + "\n")
        self._flush()

    def log_train_epoch(self, epoch, avg_loss):
        elapsed = time.time() - self.run_start
        self.data["train_loss"].append({
            "epoch": epoch, "avg_loss": round(avg_loss, 6),
            "elapsed_sec": round(elapsed, 1)
        })
        with open(self.txt_path, "a") as f:
            f.write(f"  {epoch:<8} {avg_loss:<16.6f} {elapsed:>9.1f}s\n")
        self._flush()

    def log_dev(self, epoch, train_avg_loss, overall, per_type):
        entry = {
            "epoch": epoch, "train_avg_loss": round(train_avg_loss, 6),
            "overall": {k: round(v, 6) for k, v in overall.items()},
            "per_type": {t: {k: round(v, 6) for k, v in m.items()}
                         for t, m in per_type.items()},
        }
        self.data["dev_metrics"].append(entry)
        with open(self.txt_path, "a") as f:
            f.write(f"\n  DEV Epoch {epoch} | Loss: {train_avg_loss:.6f}\n")
            f.write(f"  MRR: {overall['mrr']:.4f} H@1: {overall['hits@1']:.4f} "
                    f"H@3: {overall['hits@3']:.4f} H@5: {overall['hits@5']:.4f} "
                    f"H@10: {overall['hits@10']:.4f}\n")
        self._flush()

    def log_test(self, best_epoch, overall, per_type):
        self.data["test_metrics"] = {
            "best_epoch": best_epoch,
            "overall": {k: round(v, 6) for k, v in overall.items()},
            "per_type": {t: {k: round(v, 6) for k, v in m.items()}
                         for t, m in per_type.items()},
        }
        with open(self.txt_path, "a") as f:
            f.write("=" * 70 + "\n")
            f.write(f"  FINAL TEST (best epoch {best_epoch})\n")
            f.write(f"  MRR: {overall['mrr']:.4f} H@1: {overall['hits@1']:.4f} "
                    f"H@3: {overall['hits@3']:.4f} H@5: {overall['hits@5']:.4f} "
                    f"H@10: {overall['hits@10']:.4f}\n")
            f.write("=" * 70 + "\n")
        self._flush()
        print(f"\n  Logs: {self.txt_path}\n        {self.json_path}")

    def _flush(self):
        with open(self.json_path, "w") as f:
            json.dump(self.data, f, indent=2)


def make_collate_fn(candidate_map):
    def collate_fn(batch):
        questions  = [item[0] for item in batch]
        topic_ids  = torch.tensor([item[1] for item in batch], dtype=torch.long)
        answer_ids = [item[2] for item in batch]
        cand_tensors = [candidate_map[item[1]] for item in batch]
        max_k        = max(t.shape[0] for t in cand_tensors)
        padded       = torch.full((len(batch), max_k), -1, dtype=torch.long)
        for i, t in enumerate(cand_tensors):
            padded[i, :t.shape[0]] = t
        return questions, topic_ids, answer_ids, padded
    return collate_fn


def load_transe(silo_name, device):
    ckpt = torch.load(
        os.path.join(CHECKPOINT_DIR, f"{silo_name}_transe.pt"),
        map_location=device)
    model = TransE(
        num_entities=ckpt["num_entities"],
        num_relations=ckpt["num_relations"],
        embed_dim=ckpt["embed_dim"], norm=KGE_NORM
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded {silo_name}")
    return model


def train():
    set_seed(SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_id_path = os.path.join(CHECKPOINT_DIR, "current_run_id.txt")
    if os.path.exists(run_id_path):
        with open(run_id_path) as f:
            run_id = f.read().strip()
    else:
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    log_dir = os.path.join(os.path.dirname(CHECKPOINT_DIR), "logs", run_id)
    config_dict = {
        "baseline": "Adapted FedE (VFL)",
        "kge_embed_dim": KGE_EMBED_DIM, "qa_lr": QA_LR,
        "qa_epochs": QA_EPOCHS, "qa_batch_size": QA_BATCH_SIZE,
        "qa_margin": QA_MARGIN, "seed": SEED, "device": str(device),
    }
    logger = FedVLogger(log_dir, config_dict)

    with open(os.path.join(CHECKPOINT_DIR, "shared_entity2id.json")) as f:
        shared_entity2id = json.load(f)
    print(f"Shared entities: {len(shared_entity2id):,}")

    print("\nBuilding neighbor index...")
    neighbor_index = build_neighbor_index(
        [SILO_A_KB, SILO_B_KB, SILO_C_KB],
        shared_entity2id, max_neighbors=MAX_NEIGHBORS)
    candidate_map = precompute_candidates(
        list(shared_entity2id.values()), neighbor_index,
        hop1_cap=CANDIDATE_HOP1_CAP, hop2_cap=CANDIDATE_HOP2_CAP)

    print("\nLoading pretrained KGE models...")
    model_a = load_transe("silo_a", device)
    model_b = load_transe("silo_b", device)
    model_c = load_transe("silo_c", device)

    print("\nBuilding QA datasets...")
    train_ds = QADataset(QA_TRAIN, shared_entity2id)
    dev_ds   = QADataset(QA_DEV,   shared_entity2id)
    test_ds  = QADataset(QA_TEST,  shared_entity2id)
    print(f"  Train: {len(train_ds):,} | Dev: {len(dev_ds):,} | Test: {len(test_ds):,}")

    collate      = make_collate_fn(candidate_map)
    train_loader = DataLoader(train_ds, batch_size=QA_BATCH_SIZE,
                              shuffle=True,  collate_fn=collate, num_workers=0)
    dev_loader   = DataLoader(dev_ds,   batch_size=QA_BATCH_SIZE,
                              shuffle=False, collate_fn=collate, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=QA_BATCH_SIZE,
                              shuffle=False, collate_fn=collate, num_workers=0)

    server           = FedVServer(embed_dim=KGE_EMBED_DIM).to(device)
    server_optimizer = optim.Adam(server.question_encoder.mlp.parameters(), lr=QA_LR)
    optim_a = optim.Adam(model_a.ent_embed.parameters(), lr=QA_LR)
    optim_b = optim.Adam(model_b.ent_embed.parameters(), lr=QA_LR)
    optim_c = optim.Adam(model_c.ent_embed.parameters(), lr=QA_LR)

    best_hits1 = 0.0
    best_epoch = 0
    n_batches  = len(train_loader)

    print(f"\nAdapted FedE training — {QA_EPOCHS} epochs\n")

    for epoch in range(1, QA_EPOCHS + 1):
        server.train()
        model_a.train(); model_b.train(); model_c.train()
        total_loss = 0.0

        for batch_idx, (questions, topic_ids,
                        answer_ids_batch, candidate_ids) in \
                enumerate(train_loader, 1):

            topic_ids     = topic_ids.to(device)
            candidate_ids = candidate_ids.to(device)

            server_optimizer.zero_grad()
            optim_a.zero_grad(); optim_b.zero_grad(); optim_c.zero_grad()

            loss, _ = server(
                questions, topic_ids,
                model_a.ent_embed.weight,
                model_b.ent_embed.weight,
                model_c.ent_embed.weight,
                answer_ids_batch, candidate_ids,
                device, margin=QA_MARGIN)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                server.question_encoder.mlp.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model_a.ent_embed.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model_b.ent_embed.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model_c.ent_embed.parameters(), 1.0)

            server_optimizer.step()
            optim_a.step(); optim_b.step(); optim_c.step()

            with torch.no_grad():
                for m in [model_a, model_b, model_c]:
                    m.ent_embed.weight.data = F.normalize(
                        m.ent_embed.weight.data, p=2, dim=-1)

            total_loss += loss.item()

            if batch_idx % 200 == 0 or batch_idx == n_batches:
                print(f"  Epoch {epoch}/{QA_EPOCHS} | "
                      f"Batch {batch_idx}/{n_batches} | "
                      f"Loss: {total_loss/batch_idx:.4f}", flush=True)

        # ══════════════════════════════════════════════════════════════════════
        # KEY CHANGE: FedAvg on entity embeddings after each epoch
        # ══════════════════════════════════════════════════════════════════════
        fedavg_entity_embeddings(model_a, model_b, model_c)

        avg_loss = total_loss / n_batches
        logger.log_train_epoch(epoch, avg_loss)

        if epoch % 2 == 0 or epoch == QA_EPOCHS:
            print(f"\nEvaluating epoch {epoch}...", flush=True)
            dev_results, dev_type_results = evaluate(
                server, model_a, model_b, model_c,
                dev_loader, device, per_type=True)
            print_results("Dev", dev_results, dev_type_results)
            logger.log_dev(epoch, avg_loss, dev_results, dev_type_results)

            if dev_results["hits@1"] > best_hits1:
                best_hits1 = dev_results["hits@1"]
                best_epoch = epoch
                _save(server, model_a, model_b, model_c, "best")
                print(f"  ★ New best Hits@1: {best_hits1:.4f} (epoch {best_epoch})")
        else:
            print(f"Epoch {epoch:>3}/{QA_EPOCHS} | Train Loss: {avg_loss:.4f}")

    print(f"\nLoading best model (epoch {best_epoch})...")
    _load(server, model_a, model_b, model_c, "best", device)
    test_results, test_type_results = evaluate(
        server, model_a, model_b, model_c,
        test_loader, device, per_type=True)
    print("\n" + "=" * 60)
    print("  FINAL TEST — Adapted FedE (VFL)")
    print("=" * 60)
    print_results("Test", test_results, test_type_results)
    logger.log_test(best_epoch, test_results, test_type_results)


def _save(server, ma, mb, mc, tag):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save({
        "server": server.state_dict(),
        "model_a": ma.state_dict(),
        "model_b": mb.state_dict(),
        "model_c": mc.state_dict(),
    }, os.path.join(CHECKPOINT_DIR, f"fedv_{tag}.pt"))


def _load(server, ma, mb, mc, tag, device):
    ckpt = torch.load(
        os.path.join(CHECKPOINT_DIR, f"fedv_{tag}.pt"), map_location=device)
    server.load_state_dict(ckpt["server"])
    ma.load_state_dict(ckpt["model_a"])
    mb.load_state_dict(ckpt["model_b"])
    mc.load_state_dict(ckpt["model_c"])


if __name__ == "__main__":
    train()
