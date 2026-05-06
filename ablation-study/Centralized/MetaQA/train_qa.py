# train_qa.py — Centralized QA training (NO federation)
# Single TransE + DistilBERT. No silo fusion.

import os, json, random, time
from datetime import datetime
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from config import (
    SILO_KB_PATHS, CHECKPOINT_DIR,
    QA_TRAIN, QA_DEV, QA_TEST,
    KGE_EMBED_DIM, KGE_NORM, QA_LR, QA_EPOCHS,
    QA_BATCH_SIZE, QA_MARGIN, SEED, DEVICE,
    MAX_NEIGHBORS, CANDIDATE_HOP1_CAP, CANDIDATE_HOP2_CAP
)
from dataset import QADataset, build_neighbor_index, precompute_candidates
from transe_model import TransE
from server import CentralizedServer
from evaluate import evaluate, print_results


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_collate_fn(candidate_map):
    def collate_fn(batch):
        questions  = [item[0] for item in batch]
        topic_ids  = torch.tensor([item[1] for item in batch], dtype=torch.long)
        answer_ids = [item[2] for item in batch]
        cand_tensors = [candidate_map[item[1]] for item in batch]
        max_k  = max(t.shape[0] for t in cand_tensors)
        padded = torch.full((len(batch), max_k), -1, dtype=torch.long)
        for i, t in enumerate(cand_tensors):
            padded[i, :t.shape[0]] = t
        return questions, topic_ids, answer_ids, padded
    return collate_fn


def train():
    set_seed(SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Centralized baseline | entity_dim={KGE_EMBED_DIM} | "
          f"joint_dim={KGE_EMBED_DIM} (no fusion)")

    # Load entity vocabulary
    with open(os.path.join(CHECKPOINT_DIR, "entity2id.json")) as f:
        entity2id = json.load(f)
    print(f"Entities: {len(entity2id):,}")

    # Build neighbor index from all KB files
    print("\nBuilding neighbor index ...")
    neighbor_index = build_neighbor_index(
        SILO_KB_PATHS, entity2id, max_neighbors=MAX_NEIGHBORS)
    candidate_map = precompute_candidates(
        list(entity2id.values()), neighbor_index,
        hop1_cap=CANDIDATE_HOP1_CAP, hop2_cap=CANDIDATE_HOP2_CAP)

    # Load pretrained centralized TransE
    print("\nLoading centralized TransE ...")
    ckpt = torch.load(
        os.path.join(CHECKPOINT_DIR, "centralized_transe.pt"),
        map_location=device)
    model = TransE(
        num_entities=ckpt["num_entities"],
        num_relations=ckpt["num_relations"],
        embed_dim=ckpt["embed_dim"],
        norm=KGE_NORM).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded centralized_transe.pt")

    # QA datasets
    print("\nBuilding QA datasets ...")
    train_ds = QADataset(QA_TRAIN, entity2id)
    dev_ds   = QADataset(QA_DEV,   entity2id)
    test_ds  = QADataset(QA_TEST,  entity2id)
    print(f"  Train: {len(train_ds):,} | Dev: {len(dev_ds):,} | "
          f"Test: {len(test_ds):,}")

    collate      = make_collate_fn(candidate_map)
    train_loader = DataLoader(train_ds, batch_size=QA_BATCH_SIZE,
                              shuffle=True, collate_fn=collate, num_workers=0)
    dev_loader   = DataLoader(dev_ds, batch_size=QA_BATCH_SIZE,
                              shuffle=False, collate_fn=collate, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=QA_BATCH_SIZE,
                              shuffle=False, collate_fn=collate, num_workers=0)

    server   = CentralizedServer(embed_dim=KGE_EMBED_DIM).to(device)
    srv_opt  = optim.Adam(server.question_encoder.mlp.parameters(), lr=QA_LR)
    kge_opt  = optim.Adam(model.ent_embed.parameters(), lr=QA_LR)

    best_hits1, best_epoch = 0.0, 0
    n_batches = len(train_loader)
    print(f"\nCentralized QA — {QA_EPOCHS} epochs | "
          f"{n_batches} batches/epoch\n")

    for epoch in range(1, QA_EPOCHS + 1):
        server.train()
        model.train()
        total_loss = 0.0

        for batch_idx, (questions, topic_ids,
                        answer_ids_batch, candidate_ids) in \
                enumerate(train_loader, 1):

            topic_ids     = topic_ids.to(device)
            candidate_ids = candidate_ids.to(device)

            srv_opt.zero_grad()
            kge_opt.zero_grad()

            loss, _ = server(
                questions, topic_ids,
                model.ent_embed.weight,
                answer_ids_batch, candidate_ids,
                device, margin=QA_MARGIN)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                server.question_encoder.mlp.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model.ent_embed.parameters(), 1.0)

            srv_opt.step()
            kge_opt.step()

            with torch.no_grad():
                model.ent_embed.weight.data = F.normalize(
                    model.ent_embed.weight.data, p=2, dim=-1)

            total_loss += loss.item()

            if batch_idx % 200 == 0 or batch_idx == n_batches:
                print(f"  Epoch {epoch}/{QA_EPOCHS} | "
                      f"Batch {batch_idx}/{n_batches} | "
                      f"Loss: {total_loss/batch_idx:.4f}", flush=True)

        avg_loss = total_loss / max(n_batches, 1)

        if epoch % 2 == 0 or epoch == QA_EPOCHS:
            dev_r, dev_t = evaluate(server, model,
                                    dev_loader, device, per_type=True)
            print_results(f"Dev (epoch {epoch})", dev_r, dev_t)

            if dev_r["hits@1"] > best_hits1:
                best_hits1, best_epoch = dev_r["hits@1"], epoch
                _save(server, model, "best")
                print(f"  ★ New best Hits@1: {best_hits1:.4f} "
                      f"(epoch {best_epoch})")
        else:
            print(f"Epoch {epoch:>3}/{QA_EPOCHS}  |  Loss: {avg_loss:.4f}")

    print(f"\nLoading best model (epoch {best_epoch}) ...")
    _load(server, model, "best", device)
    test_r, test_t = evaluate(server, model,
                               test_loader, device, per_type=True)
    print("\n" + "=" * 60)
    print("  FINAL TEST  [Centralized | DistilBERT + TransE]")
    print("=" * 60)
    print_results("Test", test_r, test_t)


def _save(server, model, tag):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save({
        "server": server.state_dict(),
        "model": model.state_dict(),
    }, os.path.join(CHECKPOINT_DIR, f"centralized_{tag}.pt"))


def _load(server, model, tag, device):
    ckpt = torch.load(
        os.path.join(CHECKPOINT_DIR, f"centralized_{tag}.pt"),
        map_location=device)
    server.load_state_dict(ckpt["server"])
    model.load_state_dict(ckpt["model"])


if __name__ == "__main__":
    train()
