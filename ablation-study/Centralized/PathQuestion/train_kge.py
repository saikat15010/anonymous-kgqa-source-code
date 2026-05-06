# train_kge.py — Train single TransE on merged KB (centralized baseline)

import os, json, random, time
from datetime import datetime
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from config import (
    SILO_KB_PATHS, CHECKPOINT_DIR,
    KGE_EMBED_DIM, KGE_MARGIN, KGE_NORM, KGE_LR,
    KGE_EPOCHS, KGE_BATCH_SIZE, KGE_NEG_SAMPLES, SEED, DEVICE
)
from dataset import build_index_from_paths, KGEDataset
from transe_model import TransE


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_negatives(batch_size, num_entities, num_neg, device):
    return torch.randint(0, num_entities, (batch_size, num_neg), device=device)


def main():
    set_seed(SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Centralized TransE — single merged KB")
    print(f"KB files: {len(SILO_KB_PATHS)}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    with open(os.path.join(CHECKPOINT_DIR, "current_run_id.txt"), "w") as f:
        f.write(run_id)

    # Merge all silo KBs
    print("\nMerging all silo KBs into single KB ...")
    triples, entity2id, relation2id = build_index_from_paths(SILO_KB_PATHS)
    print(f"  Total triples:  {len(triples):,}")
    print(f"  Total entities: {len(entity2id):,}")
    print(f"  Total relations:{len(relation2id):,}")

    # Save entity2id
    with open(os.path.join(CHECKPOINT_DIR, "entity2id.json"), "w") as f:
        json.dump(entity2id, f)
    with open(os.path.join(CHECKPOINT_DIR, "relation2id.json"), "w") as f:
        json.dump(relation2id, f)

    dataset = KGEDataset(triples, entity2id, relation2id)
    loader  = DataLoader(dataset, batch_size=KGE_BATCH_SIZE,
                         shuffle=True, num_workers=4, pin_memory=True)

    num_entities  = len(entity2id)
    num_relations = len(relation2id)

    model     = TransE(num_entities, num_relations,
                       KGE_EMBED_DIM, norm=KGE_NORM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=KGE_LR)

    print(f"\nTraining TransE — {KGE_EPOCHS} epochs\n")

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

            with torch.no_grad():
                model.ent_embed.weight.data = F.normalize(
                    model.ent_embed.weight.data, p=2, dim=-1)

            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4}/{KGE_EPOCHS}  |  Loss: {avg_loss:.4f}",
                  flush=True)

    ckpt_path = os.path.join(CHECKPOINT_DIR, "centralized_transe.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "entity2id"       : entity2id,
        "relation2id"     : relation2id,
        "embed_dim"       : KGE_EMBED_DIM,
        "num_entities"    : num_entities,
        "num_relations"   : num_relations,
    }, ckpt_path)
    print(f"\n  Saved → {ckpt_path}")


if __name__ == "__main__":
    main()
