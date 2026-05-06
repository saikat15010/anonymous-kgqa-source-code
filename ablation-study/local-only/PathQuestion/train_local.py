# train_local.py — Local-Only QA training and evaluation
# Each question is routed to the silo with the most triples for its topic entity.
# Each silo has its own QA encoder (shared architecture, separate weights).
# No cross-silo communication, no fusion.
#
# Reuses KGE checkpoints from the federated experiment (KGE is silo-local in both).

import os, json, random, time
from datetime import datetime
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from config import (
    SILO_A_KB, SILO_B_KB, SILO_C_KB,
    CHECKPOINT_DIR, FEDERATED_CHECKPOINT_DIR,
    QA_TRAIN, QA_DEV, QA_TEST,
    KGE_EMBED_DIM, KGE_NORM, QA_LR, QA_EPOCHS,
    QA_BATCH_SIZE, QA_MARGIN, SEED, DEVICE,
    MAX_NEIGHBORS, CANDIDATE_HOP1_CAP, CANDIDATE_HOP2_CAP
)
from dataset import (QADataset, build_neighbor_index_single,
                     count_entity_triples, precompute_candidates)
from transe_model import TransE
from server import LocalServer


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Logger ────────────────────────────────────────────────────────────────────

class LocalLogger:
    def __init__(self, log_dir, config_dict):
        os.makedirs(log_dir, exist_ok=True)
        self.txt_path  = os.path.join(log_dir, "local_train_log.txt")
        self.json_path = os.path.join(log_dir, "local_train_log.json")
        self.run_start = time.time()
        self.data = {
            "run_id": os.path.basename(log_dir), "mode": "local_only",
            "kge_model": "TransE", "qa_encoder": "DistilBERT",
            "config": config_dict, "train_loss": [],
            "dev_metrics": [], "test_metrics": None}
        with open(self.txt_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("  Local-Only QA Training Log"
                    "  [DistilBERT + TransE | No Federation]\n")
            f.write(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\nConfig:\n")
            for k, v in config_dict.items():
                f.write(f"  {k:<26} = {v}\n")
            f.write(f"\n{'─'*70}\n")
            f.write(f"  {'Epoch':<8} {'Avg Loss':<16} {'Elapsed':>10}\n")
            f.write("─" * 70 + "\n")
        self._flush()
        print(f"Logs → {log_dir}")

    def log_train_epoch(self, epoch, avg_loss):
        elapsed = time.time() - self.run_start
        self.data["train_loss"].append(
            {"epoch": epoch, "avg_loss": round(avg_loss, 6),
             "elapsed_sec": round(elapsed, 1)})
        with open(self.txt_path, "a") as f:
            f.write(f"  {epoch:<8} {avg_loss:<16.6f} {elapsed:>9.1f}s\n")
        self._flush()

    def log_dev(self, epoch, train_loss, overall):
        self.data["dev_metrics"].append(
            {"epoch": epoch, "train_avg_loss": round(train_loss, 6),
             "overall": {k: round(v, 6) for k, v in overall.items()}})
        with open(self.txt_path, "a") as f:
            f.write(f"\n{'─'*70}\n")
            f.write(f"  DEV Epoch {epoch} | Loss {train_loss:.6f}\n")
            f.write(f"  MRR:{overall['mrr']:.4f}  H@1:{overall['hits@1']:.4f}"
                    f"  H@3:{overall['hits@3']:.4f}  H@5:{overall['hits@5']:.4f}"
                    f"  H@10:{overall['hits@10']:.4f}  N:{overall['total']}\n")
            f.write(f"{'─'*70}\n\n")
        self._flush()

    def log_test(self, best_epoch, overall):
        total = time.time() - self.run_start
        self.data["test_metrics"] = {
            "best_epoch": best_epoch,
            "overall": {k: round(v, 6) for k, v in overall.items()}}
        with open(self.txt_path, "a") as f:
            f.write("=" * 70 + "\n")
            f.write(f"  FINAL TEST (best epoch {best_epoch})\n")
            f.write(f"  MRR:{overall['mrr']:.4f}  H@1:{overall['hits@1']:.4f}"
                    f"  H@3:{overall['hits@3']:.4f}  H@5:{overall['hits@5']:.4f}"
                    f"  H@10:{overall['hits@10']:.4f}  N:{overall['total']}\n")
            f.write(f"  Total time: {total/3600:.2f} hours\n")
            f.write("=" * 70 + "\n")
        self._flush()
        print(f"  Logs saved → {self.txt_path}")
        print(f"               {self.json_path}")

    def _flush(self):
        with open(self.json_path, "w") as f:
            json.dump(self.data, f, indent=2)


# ── Collate ───────────────────────────────────────────────────────────────────

def make_collate_fn(candidate_maps):
    """
    Collate that uses per-silo candidate maps.
    Each sample has a silo_idx; candidates come from that silo's map.
    """
    def collate_fn(batch):
        questions  = [item[0] for item in batch]
        topic_ids  = torch.tensor([item[1] for item in batch], dtype=torch.long)
        answer_ids = [item[2] for item in batch]
        silo_idxs  = [item[3] for item in batch]

        cand_tensors = []
        for item in batch:
            topic_id = item[1]
            silo_idx = item[3]
            cmap = candidate_maps[silo_idx]
            if topic_id in cmap:
                cand_tensors.append(cmap[topic_id])
            else:
                cand_tensors.append(torch.tensor([topic_id], dtype=torch.long))

        max_k  = max(t.shape[0] for t in cand_tensors)
        padded = torch.full((len(batch), max_k), -1, dtype=torch.long)
        for i, t in enumerate(cand_tensors):
            padded[i, :t.shape[0]] = t

        return questions, topic_ids, answer_ids, padded, silo_idxs
    return collate_fn


# ── Evaluate ──────────────────────────────────────────────────────────────────

class MetricAccumulator:
    def __init__(self):
        self.hits_1 = self.hits_3 = self.hits_5 = self.hits_10 = 0
        self.mrr = 0.0
        self.total = 0

    def update(self, rank):
        self.total += 1
        if rank is not None:
            self.mrr     += 1.0 / rank
            self.hits_1  += int(rank == 1)
            self.hits_3  += int(rank <= 3)
            self.hits_5  += int(rank <= 5)
            self.hits_10 += int(rank <= 10)

    def results(self):
        n = max(self.total, 1)
        return {"mrr": self.mrr/n, "hits@1": self.hits_1/n,
                "hits@3": self.hits_3/n, "hits@5": self.hits_5/n,
                "hits@10": self.hits_10/n, "total": self.total}


def evaluate_local(servers, models, qa_loader, device):
    """
    Evaluate local-only: each question uses its assigned silo's
    server and model independently.
    """
    for s in servers:
        s.eval()
    for m in models:
        m.eval()

    overall = MetricAccumulator()

    with torch.no_grad():
        # Precompute all silo embeddings
        h_silos = [m.get_entity_embeddings().to(device) for m in models]

        for questions, topic_ids, answer_ids_batch, candidate_ids, silo_idxs in qa_loader:
            topic_ids     = topic_ids.to(device)
            candidate_ids = candidate_ids.to(device)

            # Process each sample with its assigned silo
            for i in range(len(questions)):
                s_idx = silo_idxs[i]
                server = servers[s_idx]
                h_silo = h_silos[s_idx]

                q_embed = server.question_encoder([questions[i]], device)
                if server.question_encoder.__class__.__name__:
                    from config import USE_TOPIC_ANCHORING
                    if USE_TOPIC_ANCHORING:
                        q_final = q_embed + h_silo[topic_ids[i:i+1]]
                    else:
                        q_final = q_embed

                cands = candidate_ids[i:i+1]
                sim   = server.score_candidates(q_final, h_silo, cands)

                cand_row = cands[0]
                valid    = cand_row >= 0
                cand_list = cand_row[valid].tolist()
                scores    = sim[0][valid].tolist()
                ranked    = sorted(zip(cand_list, scores),
                                   key=lambda x: x[1], reverse=True)
                answer_set = set(answer_ids_batch[i])
                rank = next((pos+1 for pos, (eid, _) in enumerate(ranked)
                             if eid in answer_set), None)
                overall.update(rank)

    return overall.results()


# ── Load KGE ──────────────────────────────────────────────────────────────────

def load_transe(silo_name, device):
    ckpt = torch.load(
        os.path.join(FEDERATED_CHECKPOINT_DIR, f"{silo_name}_transe.pt"),
        map_location=device)
    model = TransE(
        num_entities=ckpt["num_entities"],
        num_relations=ckpt["num_relations"],
        embed_dim=ckpt["embed_dim"],
        norm=KGE_NORM).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded {silo_name}_transe.pt")
    return model


# ── Main ──────────────────────────────────────────────────────────────────────

def train():
    set_seed(SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Local-Only baseline | entity_dim={KGE_EMBED_DIM} | no fusion")

    run_id  = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(CHECKPOINT_DIR), "logs", run_id)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    config_dict = {
        "mode": "local_only", "kge_model": "TransE",
        "qa_encoder": "DistilBERT", "entity_dim": KGE_EMBED_DIM,
        "qa_lr": QA_LR, "qa_epochs": QA_EPOCHS,
        "qa_batch_size": QA_BATCH_SIZE, "qa_margin": QA_MARGIN,
        "seed": SEED, "device": str(device)}
    logger = LocalLogger(log_dir, config_dict)

    # Load shared entity vocab from federated experiment
    with open(os.path.join(FEDERATED_CHECKPOINT_DIR, "shared_entity2id.json")) as f:
        entity2id = json.load(f)
    print(f"Shared entities: {len(entity2id):,}")

    # Count triples per entity per silo (for topic→silo assignment)
    silo_paths = [SILO_A_KB, SILO_B_KB, SILO_C_KB]
    silo_names = ["silo_a", "silo_b", "silo_c"]

    print("\nCounting entity triples per silo ...")
    silo_counts = [count_entity_triples(p, entity2id) for p in silo_paths]

    # Build per-silo neighbor indices and candidate maps
    print("Building per-silo neighbor indices ...")
    silo_neighbor_indices = [
        build_neighbor_index_single(p, entity2id, max_neighbors=MAX_NEIGHBORS)
        for p in silo_paths]
    silo_candidate_maps = [
        precompute_candidates(
            list(entity2id.values()), nb_idx,
            hop1_cap=CANDIDATE_HOP1_CAP, hop2_cap=CANDIDATE_HOP2_CAP)
        for nb_idx in silo_neighbor_indices]

    # Load KGE models (from federated experiment — same silo-local training)
    print("\nLoading TransE models ...")
    models = [load_transe(name, device) for name in silo_names]

    # Create one LocalServer per silo (shared architecture, separate training)
    servers = [LocalServer(embed_dim=KGE_EMBED_DIM).to(device) for _ in range(3)]

    # Build QA datasets with silo assignment
    print("\nBuilding QA datasets ...")
    train_ds = QADataset(QA_TRAIN, entity2id, silo_counts)
    dev_ds   = QADataset(QA_DEV,   entity2id, silo_counts)
    test_ds  = QADataset(QA_TEST,  entity2id, silo_counts)
    print(f"  Train: {len(train_ds):,} | Dev: {len(dev_ds):,} | "
          f"Test: {len(test_ds):,}")

    # Print silo distribution
    from collections import Counter
    train_dist = Counter(s[3] for s in train_ds.samples)
    print(f"  Train silo distribution: "
          f"A={train_dist.get(0,0)} B={train_dist.get(1,0)} C={train_dist.get(2,0)}")

    collate      = make_collate_fn(silo_candidate_maps)
    train_loader = DataLoader(train_ds, batch_size=QA_BATCH_SIZE,
                              shuffle=True, collate_fn=collate, num_workers=0)
    dev_loader   = DataLoader(dev_ds, batch_size=QA_BATCH_SIZE,
                              shuffle=False, collate_fn=collate, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=QA_BATCH_SIZE,
                              shuffle=False, collate_fn=collate, num_workers=0)

    # Optimizers — one per silo server + one per silo KGE embeddings
    srv_opts = [optim.Adam(s.question_encoder.mlp.parameters(), lr=QA_LR)
                for s in servers]
    kge_opts = [optim.Adam(m.ent_embed.parameters(), lr=QA_LR)
                for m in models]

    best_hits1, best_epoch = 0.0, 0
    n_batches = len(train_loader)
    print(f"\nLocal-Only QA — {QA_EPOCHS} epochs | {n_batches} batches/epoch\n")

    for epoch in range(1, QA_EPOCHS + 1):
        for s in servers:
            s.train()
        for m in models:
            m.train()
        total_loss = 0.0
        batch_count = 0

        for batch_idx, (questions, topic_ids, answer_ids_batch,
                        candidate_ids, silo_idxs) in enumerate(train_loader, 1):
            topic_ids     = topic_ids.to(device)
            candidate_ids = candidate_ids.to(device)

            # Group by silo for efficient training
            silo_groups = {}
            for i in range(len(questions)):
                s_idx = silo_idxs[i]
                if s_idx not in silo_groups:
                    silo_groups[s_idx] = []
                silo_groups[s_idx].append(i)

            batch_loss = 0.0
            for s_idx, indices in silo_groups.items():
                srv_opts[s_idx].zero_grad()
                kge_opts[s_idx].zero_grad()

                q_batch   = [questions[i] for i in indices]
                t_batch   = topic_ids[indices]
                a_batch   = [answer_ids_batch[i] for i in indices]
                c_batch   = candidate_ids[indices]

                loss, _ = servers[s_idx](
                    q_batch, t_batch,
                    models[s_idx].ent_embed.weight,
                    a_batch, c_batch,
                    device, margin=QA_MARGIN)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    servers[s_idx].question_encoder.mlp.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(
                    models[s_idx].ent_embed.parameters(), 1.0)

                srv_opts[s_idx].step()
                kge_opts[s_idx].step()

                with torch.no_grad():
                    models[s_idx].ent_embed.weight.data = F.normalize(
                        models[s_idx].ent_embed.weight.data, p=2, dim=-1)

                batch_loss += loss.item()

            total_loss += batch_loss
            batch_count += 1

            if batch_idx % 200 == 0 or batch_idx == n_batches:
                print(f"  Epoch {epoch}/{QA_EPOCHS} | "
                      f"Batch {batch_idx}/{n_batches} | "
                      f"Loss: {total_loss/batch_count:.4f}", flush=True)

        avg_loss = total_loss / max(batch_count, 1)
        logger.log_train_epoch(epoch, avg_loss)

        if epoch % 2 == 0 or epoch == QA_EPOCHS:
            dev_r = evaluate_local(servers, models, dev_loader, device)
            print(f"\n  Dev (epoch {epoch}): MRR:{dev_r['mrr']:.4f}"
                  f"  H@1:{dev_r['hits@1']:.4f}  H@10:{dev_r['hits@10']:.4f}"
                  f"  N:{dev_r['total']}")
            logger.log_dev(epoch, avg_loss, dev_r)

            if dev_r["hits@1"] > best_hits1:
                best_hits1, best_epoch = dev_r["hits@1"], epoch
                _save(servers, models, "best")
                print(f"  ★ New best Hits@1: {best_hits1:.4f} "
                      f"(epoch {best_epoch})")
        else:
            print(f"Epoch {epoch:>3}/{QA_EPOCHS}  |  Loss: {avg_loss:.4f}")

    print(f"\nLoading best model (epoch {best_epoch}) ...")
    _load(servers, models, "best", device)
    test_r = evaluate_local(servers, models, test_loader, device)

    print("\n" + "=" * 60)
    print("  FINAL TEST  [Local-Only | DistilBERT + TransE]")
    print("=" * 60)
    print(f"  MRR:{test_r['mrr']:.4f}  H@1:{test_r['hits@1']:.4f}"
          f"  H@3:{test_r['hits@3']:.4f}  H@5:{test_r['hits@5']:.4f}"
          f"  H@10:{test_r['hits@10']:.4f}  N:{test_r['total']}")
    print("=" * 60)
    logger.log_test(best_epoch, test_r)


def _save(servers, models, tag):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    state = {}
    for i, (s, m) in enumerate(zip(servers, models)):
        state[f"server_{i}"] = s.state_dict()
        state[f"model_{i}"]  = m.state_dict()
    torch.save(state, os.path.join(CHECKPOINT_DIR, f"local_{tag}.pt"))


def _load(servers, models, tag, device):
    ckpt = torch.load(
        os.path.join(CHECKPOINT_DIR, f"local_{tag}.pt"),
        map_location=device)
    for i, (s, m) in enumerate(zip(servers, models)):
        s.load_state_dict(ckpt[f"server_{i}"])
        m.load_state_dict(ckpt[f"model_{i}"])


if __name__ == "__main__":
    train()
