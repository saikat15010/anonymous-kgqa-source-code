# train_fedv.py — Federated QA training | BERT + TransE | 7 silos | Client7
#
# Differences from Client5 train_fedv.py (5 silos):
#   - Imports SILO_F_KB, SILO_G_KB
#   - Loads model_f, model_g
#   - 7 optimizers: optim_a … optim_g
#   - server() forward: h_a … h_g (each (N, 256) TransE embeddings)
#   - Gradient clipping on 7 models
#   - L2 normalisation on 7 models
#   - build_neighbor_index with 7 KB paths
#   - evaluate() with model_a … model_g
#   - _save/_load with model_a … model_g

import os, json, random, time
from datetime import datetime
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from config import (
    SILO_A_KB, SILO_B_KB, SILO_C_KB, SILO_D_KB,
    SILO_E_KB, SILO_F_KB, SILO_G_KB,
    CHECKPOINT_DIR, QA_TRAIN, QA_DEV, QA_TEST,
    KGE_EMBED_DIM, KGE_NORM, QA_LR, QA_EPOCHS,
    QA_BATCH_SIZE, QA_MARGIN, SEED, DEVICE,
    MAX_NEIGHBORS, CANDIDATE_HOP1_CAP, CANDIDATE_HOP2_CAP,
    JOINT_DIM
)
from dataset import QADataset, build_neighbor_index, precompute_candidates
from transe_model import TransE
from server import FedVServer
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
            "run_id"      : os.path.basename(log_dir),
            "kge_model"   : "TransE",
            "qa_encoder"  : "BERT",
            "num_silos"   : 7,
            "config"      : config_dict,
            "train_loss"  : [],
            "dev_metrics" : [],
            "test_metrics": None,
        }

        with open(self.txt_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("  FedV-KGQA  —  Federated QA Training Log"
                    "  [TransE + BERT | 7 silos | Client7]\n")
            f.write(f"  Run ID : {self.data['run_id']}\n")
            f.write(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")
            f.write("Config:\n")
            for k, v in config_dict.items():
                f.write(f"  {k:<26} = {v}\n")
            f.write("\n")
            f.write("-" * 70 + "\n")
            f.write("  TRAINING LOSS  (every epoch)\n")
            f.write("-" * 70 + "\n")
            f.write(f"  {'Epoch':<8} {'Avg Loss':<16} {'Elapsed':>10}\n")
            f.write("-" * 70 + "\n")
        self._flush()
        print(f"FedV logs → {log_dir}")

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
            "epoch"         : epoch,
            "train_avg_loss": round(train_avg_loss, 6),
            "overall"       : {k: round(v, 6) for k, v in overall.items()},
            "per_type"      : {t: {k: round(v, 6) for k, v in m.items()}
                               for t, m in per_type.items()},
        }
        self.data["dev_metrics"].append(entry)
        with open(self.txt_path, "a") as f:
            f.write("\n" + "-" * 70 + "\n")
            f.write(f"  DEV EVALUATION  —  Epoch {epoch}"
                    f"  |  Train Loss: {train_avg_loss:.6f}\n")
            f.write("-" * 70 + "\n")
            f.write(_fmt_overall(overall))
            f.write(_fmt_per_type(per_type))
            f.write("-" * 70 + "\n\n")
        self._flush()

    def log_test(self, best_epoch, overall, per_type):
        self.data["test_metrics"] = {
            "best_epoch": best_epoch,
            "overall"   : {k: round(v, 6) for k, v in overall.items()},
            "per_type"  : {t: {k: round(v, 6) for k, v in m.items()}
                           for t, m in per_type.items()},
        }
        total = time.time() - self.run_start
        with open(self.txt_path, "a") as f:
            f.write("=" * 70 + "\n")
            f.write(f"  FINAL TEST RESULTS"
                    f"  (best checkpoint: epoch {best_epoch})\n")
            f.write("=" * 70 + "\n")
            f.write(_fmt_overall(overall))
            f.write(_fmt_per_type(per_type))
            f.write("=" * 70 + "\n")
            f.write(f"\n  Total training time : {total/3600:.2f} hours\n")
            f.write(f"  Finished            : "
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self._flush()
        print(f"\n  FedV logs saved:\n    {self.txt_path}\n    {self.json_path}")

    def _flush(self):
        with open(self.json_path, "w") as f:
            json.dump(self.data, f, indent=2)


def _fmt_overall(r):
    return (f"  Overall"
            f"  |  MRR: {r['mrr']:.4f}"
            f"  Hits@1: {r['hits@1']:.4f}"
            f"  Hits@3: {r['hits@3']:.4f}"
            f"  Hits@5: {r['hits@5']:.4f}"
            f"  Hits@10: {r['hits@10']:.4f}"
            f"  N: {r['total']}\n")


def _fmt_per_type(per_type):
    lines = "  Per Answer-Type:\n"
    for atype in ["person", "movie", "year", "genre", "language", "unknown"]:
        if atype not in per_type:
            continue
        r = per_type[atype]
        lines += (f"    {atype:<10}"
                  f"  MRR: {r['mrr']:.4f}"
                  f"  Hits@1: {r['hits@1']:.4f}"
                  f"  Hits@3: {r['hits@3']:.4f}"
                  f"  Hits@5: {r['hits@5']:.4f}"
                  f"  Hits@10: {r['hits@10']:.4f}"
                  f"  N: {r['total']}\n")
    return lines


def make_collate_fn(candidate_map):
    def collate_fn(batch):
        questions    = [item[0] for item in batch]
        topic_ids    = torch.tensor([item[1] for item in batch], dtype=torch.long)
        answer_ids   = [item[2] for item in batch]
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
        map_location=device
    )
    model = TransE(
        num_entities  = ckpt["num_entities"],
        num_relations = ckpt["num_relations"],
        embed_dim     = ckpt["embed_dim"],
        norm          = KGE_NORM
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded {silo_name}  |  entities: {ckpt['num_entities']:,}")
    return model


def _save(server, ma, mb, mc, md, me, mf, mg, tag):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save({
        "server" : server.state_dict(),
        "model_a": ma.state_dict(),
        "model_b": mb.state_dict(),
        "model_c": mc.state_dict(),
        "model_d": md.state_dict(),
        "model_e": me.state_dict(),
        "model_f": mf.state_dict(),
        "model_g": mg.state_dict(),
    }, os.path.join(CHECKPOINT_DIR, f"fedv_{tag}.pt"))


def _load(server, ma, mb, mc, md, me, mf, mg, tag, device):
    ckpt = torch.load(
        os.path.join(CHECKPOINT_DIR, f"fedv_{tag}.pt"),
        map_location=device)
    server.load_state_dict(ckpt["server"])
    ma.load_state_dict(ckpt["model_a"])
    mb.load_state_dict(ckpt["model_b"])
    mc.load_state_dict(ckpt["model_c"])
    md.load_state_dict(ckpt["model_d"])
    me.load_state_dict(ckpt["model_e"])
    mf.load_state_dict(ckpt["model_f"])
    mg.load_state_dict(ckpt["model_g"])


def train():
    set_seed(SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"TransE entity dim per silo : {KGE_EMBED_DIM}")
    print(f"Joint embedding dim (7d)   : {JOINT_DIM}")

    run_id_path = os.path.join(CHECKPOINT_DIR, "current_run_id.txt")
    if os.path.exists(run_id_path):
        with open(run_id_path) as f:
            run_id = f.read().strip()
        print(f"Continuing run: {run_id}")
    else:
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        print(f"New run: {run_id}")

    log_dir = os.path.join(os.path.dirname(CHECKPOINT_DIR), "logs", run_id)

    config_dict = {
        "kge_model"          : "TransE",
        "qa_encoder"         : "BERT",
        "num_silos"          : 7,
        "kge_embed_dim"      : KGE_EMBED_DIM,
        "joint_dim"          : JOINT_DIM,
        "qa_lr"              : QA_LR,
        "qa_epochs"          : QA_EPOCHS,
        "qa_batch_size"      : QA_BATCH_SIZE,
        "qa_margin"          : QA_MARGIN,
        "candidate_hop1_cap" : CANDIDATE_HOP1_CAP,
        "candidate_hop2_cap" : CANDIDATE_HOP2_CAP,
        "max_neighbors"      : MAX_NEIGHBORS,
        "seed"               : SEED,
        "device"             : str(device),
    }
    logger = FedVLogger(log_dir, config_dict)

    with open(os.path.join(CHECKPOINT_DIR, "shared_entity2id.json")) as f:
        shared_entity2id = json.load(f)
    print(f"Shared entities: {len(shared_entity2id):,}")

    print("\nBuilding neighbor index (all 7 silo KBs)...")
    neighbor_index = build_neighbor_index(
        [SILO_A_KB, SILO_B_KB, SILO_C_KB, SILO_D_KB,
         SILO_E_KB, SILO_F_KB, SILO_G_KB],
        shared_entity2id, max_neighbors=MAX_NEIGHBORS
    )
    print("Precomputing 2-hop candidate sets...")
    candidate_map = precompute_candidates(
        list(shared_entity2id.values()), neighbor_index,
        hop1_cap=CANDIDATE_HOP1_CAP, hop2_cap=CANDIDATE_HOP2_CAP
    )

    print("\nLoading pretrained TransE models (Phase 1)...")
    model_a = load_transe("silo_a", device)
    model_b = load_transe("silo_b", device)
    model_c = load_transe("silo_c", device)
    model_d = load_transe("silo_d", device)
    model_e = load_transe("silo_e", device)
    model_f = load_transe("silo_f", device)
    model_g = load_transe("silo_g", device)

    print("\nBuilding QA datasets...")
    train_ds = QADataset(QA_TRAIN, shared_entity2id)
    dev_ds   = QADataset(QA_DEV,   shared_entity2id)
    test_ds  = QADataset(QA_TEST,  shared_entity2id)
    print(f"  Train: {len(train_ds):,} | Dev: {len(dev_ds):,} | "
          f"Test: {len(test_ds):,}")

    collate      = make_collate_fn(candidate_map)
    train_loader = DataLoader(train_ds, batch_size=QA_BATCH_SIZE,
                              shuffle=True,  collate_fn=collate, num_workers=0)
    dev_loader   = DataLoader(dev_ds,   batch_size=QA_BATCH_SIZE,
                              shuffle=False, collate_fn=collate, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=QA_BATCH_SIZE,
                              shuffle=False, collate_fn=collate, num_workers=0)

    server           = FedVServer(embed_dim=KGE_EMBED_DIM).to(device)
    server_optimizer = optim.Adam(
        server.question_encoder.mlp.parameters(), lr=QA_LR)
    optim_a = optim.Adam(model_a.ent_embed.parameters(), lr=QA_LR)
    optim_b = optim.Adam(model_b.ent_embed.parameters(), lr=QA_LR)
    optim_c = optim.Adam(model_c.ent_embed.parameters(), lr=QA_LR)
    optim_d = optim.Adam(model_d.ent_embed.parameters(), lr=QA_LR)
    optim_e = optim.Adam(model_e.ent_embed.parameters(), lr=QA_LR)
    optim_f = optim.Adam(model_f.ent_embed.parameters(), lr=QA_LR)
    optim_g = optim.Adam(model_g.ent_embed.parameters(), lr=QA_LR)

    best_hits1 = 0.0
    best_epoch = 0
    n_batches  = len(train_loader)

    print(f"\nFederated QA training (TransE + BERT)"
          f" — {QA_EPOCHS} epochs | {n_batches} batches/epoch\n")

    for epoch in range(1, QA_EPOCHS + 1):
        server.train()
        model_a.train(); model_b.train(); model_c.train()
        model_d.train(); model_e.train(); model_f.train(); model_g.train()
        total_loss = 0.0

        for batch_idx, (questions, topic_ids,
                        answer_ids_batch, candidate_ids) in \
                enumerate(train_loader, 1):

            topic_ids     = topic_ids.to(device)
            candidate_ids = candidate_ids.to(device)

            server_optimizer.zero_grad()
            optim_a.zero_grad(); optim_b.zero_grad(); optim_c.zero_grad()
            optim_d.zero_grad(); optim_e.zero_grad()
            optim_f.zero_grad(); optim_g.zero_grad()

            loss, _ = server(
                questions, topic_ids,
                model_a.ent_embed.weight,    # h_A: (N, 256)
                model_b.ent_embed.weight,    # h_B: (N, 256)
                model_c.ent_embed.weight,    # h_C: (N, 256)
                model_d.ent_embed.weight,    # h_D: (N, 256)
                model_e.ent_embed.weight,    # h_E: (N, 256)
                model_f.ent_embed.weight,    # h_F: (N, 256)
                model_g.ent_embed.weight,    # h_G: (N, 256)
                answer_ids_batch, candidate_ids,
                device, margin=QA_MARGIN
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                server.question_encoder.mlp.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model_a.ent_embed.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model_b.ent_embed.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model_c.ent_embed.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model_d.ent_embed.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model_e.ent_embed.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model_f.ent_embed.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model_g.ent_embed.parameters(), 1.0)

            server_optimizer.step()
            optim_a.step(); optim_b.step(); optim_c.step()
            optim_d.step(); optim_e.step()
            optim_f.step(); optim_g.step()

            # TransE: L2-normalise entity embeddings after each update
            with torch.no_grad():
                for m in [model_a, model_b, model_c, model_d,
                          model_e, model_f, model_g]:
                    m.ent_embed.weight.data = F.normalize(
                        m.ent_embed.weight.data, p=2, dim=-1)

            total_loss += loss.item()

            if batch_idx % 200 == 0 or batch_idx == n_batches:
                print(f"  Epoch {epoch}/{QA_EPOCHS} | "
                      f"Batch {batch_idx}/{n_batches} | "
                      f"Loss: {total_loss/batch_idx:.4f}", flush=True)

        avg_loss = total_loss / n_batches
        logger.log_train_epoch(epoch, avg_loss)

        if epoch % 2 == 0 or epoch == QA_EPOCHS:
            print(f"\nEvaluating epoch {epoch}...", flush=True)
            dev_results, dev_type_results = evaluate(
                server, model_a, model_b, model_c, model_d,
                model_e, model_f, model_g,
                dev_loader, device, per_type=True
            )
            print_results("Dev", dev_results, dev_type_results)
            logger.log_dev(epoch, avg_loss, dev_results, dev_type_results)

            if dev_results["hits@1"] > best_hits1:
                best_hits1 = dev_results["hits@1"]
                best_epoch = epoch
                _save(server, model_a, model_b, model_c, model_d,
                      model_e, model_f, model_g, "best")
                print(f"  ★ New best  Hits@1: {best_hits1:.4f}  "
                      f"(epoch {best_epoch})")
        else:
            print(f"Epoch {epoch:>3}/{QA_EPOCHS}  |  "
                  f"Train Loss: {avg_loss:.4f}")

    print(f"\nLoading best model (epoch {best_epoch})...")
    _load(server, model_a, model_b, model_c, model_d,
          model_e, model_f, model_g, "best", device)

    test_results, test_type_results = evaluate(
        server, model_a, model_b, model_c, model_d,
        model_e, model_f, model_g,
        test_loader, device, per_type=True
    )

    print("\n" + "=" * 65)
    print("  FINAL TEST RESULTS (TransE + BERT | 2-hop | 7 silos | Client7)")
    print("=" * 65)
    print_results("Test", test_results, test_type_results)
    print("=" * 65)

    logger.log_test(best_epoch, test_results, test_type_results)


if __name__ == "__main__":
    train()
