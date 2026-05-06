# eval_noise_robustness.py — Noise Robustness Experiment
# MetaQA | Client5 | BERT + TransE | 5 silos | 2-hop
#
# This script:
#   1. Loads the best federated QA checkpoint (fedv_best.pt)
#   2. For each noise level σ ∈ {0.0, 0.01, 0.05, 0.1}:
#      a. Clones the entity embeddings from all 5 silos
#      b. Adds Gaussian noise N(0, σ) to ALL entity embeddings
#      c. Evaluates on the test set
#   3. Prints a comparison table
#
# No retraining needed — this is evaluation-only.
# The noise simulates perturbation during federated communication
# (e.g., channel noise or differential privacy mechanisms).

import os, json, copy, time
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import (
    SILO_A_KB, SILO_B_KB, SILO_C_KB, SILO_D_KB, SILO_E_KB,
    CHECKPOINT_DIR, QA_TEST,
    KGE_EMBED_DIM, KGE_NORM, QA_BATCH_SIZE,
    SEED, DEVICE, MAX_NEIGHBORS,
    CANDIDATE_HOP1_CAP, CANDIDATE_HOP2_CAP,
    NOISE_SIGMAS
)
from dataset import QADataset, build_neighbor_index, precompute_candidates
from transe_model import TransE
from server import FedVServer
from evaluate import evaluate, print_results


def set_seed(seed):
    import random
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
        embed_dim=ckpt["embed_dim"],
        norm=KGE_NORM).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def load_best_checkpoint(server, models, device):
    """Load the best federated QA checkpoint."""
    ckpt = torch.load(
        os.path.join(CHECKPOINT_DIR, "fedv_best.pt"),
        map_location=device)
    server.load_state_dict(ckpt["server"])
    for name, model in zip(["model_a", "model_b", "model_c",
                             "model_d", "model_e"], models):
        model.load_state_dict(ckpt[name])


def inject_noise(models, sigma, device):
    """
    Add Gaussian noise N(0, σ) to entity embeddings of ALL silos.

    This simulates perturbation during federated communication,
    e.g., channel noise or differential privacy mechanisms.

    After adding noise, embeddings are re-normalized to unit sphere
    (TransE convention) to keep the noise effect realistic.
    """
    with torch.no_grad():
        for model in models:
            noise = torch.randn_like(model.ent_embed.weight) * sigma
            model.ent_embed.weight.data += noise
            # Re-normalize to unit sphere (TransE convention)
            model.ent_embed.weight.data = F.normalize(
                model.ent_embed.weight.data, p=2, dim=-1)


def main():
    set_seed(SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Noise Robustness Experiment — MetaQA | Client5 | BERT+TransE")
    print(f"Noise levels σ: {NOISE_SIGMAS}")
    print("=" * 70)

    # Load shared entity vocabulary
    with open(os.path.join(CHECKPOINT_DIR, "shared_entity2id.json")) as f:
        shared_entity2id = json.load(f)
    print(f"Shared entities: {len(shared_entity2id):,}")

    # Build neighbor index and candidate map (done once)
    print("\nBuilding neighbor index ...")
    neighbor_index = build_neighbor_index(
        [SILO_A_KB, SILO_B_KB, SILO_C_KB, SILO_D_KB, SILO_E_KB],
        shared_entity2id, max_neighbors=MAX_NEIGHBORS)
    candidate_map = precompute_candidates(
        list(shared_entity2id.values()), neighbor_index,
        hop1_cap=CANDIDATE_HOP1_CAP, hop2_cap=CANDIDATE_HOP2_CAP)

    # Build test dataset (done once)
    test_ds = QADataset(QA_TEST, shared_entity2id)
    print(f"Test samples: {len(test_ds):,}")
    collate     = make_collate_fn(candidate_map)
    test_loader = DataLoader(test_ds, batch_size=QA_BATCH_SIZE,
                             shuffle=False, collate_fn=collate, num_workers=0)

    # Load base models (structure only — weights loaded from checkpoint each time)
    model_a = load_transe("silo_a", device)
    model_b = load_transe("silo_b", device)
    model_c = load_transe("silo_c", device)
    model_d = load_transe("silo_d", device)
    model_e = load_transe("silo_e", device)
    models  = [model_a, model_b, model_c, model_d, model_e]

    server = FedVServer(embed_dim=KGE_EMBED_DIM).to(device)

    # ── Run evaluation for each noise level ──────────────────────────────────
    all_results = {}

    for sigma in NOISE_SIGMAS:
        print(f"\n{'='*70}")
        print(f"  Evaluating with σ = {sigma}")
        print(f"{'='*70}")

        # Reload best checkpoint (fresh weights each time)
        load_best_checkpoint(server, models, device)

        # Inject noise (skip for σ=0.0 baseline)
        if sigma > 0:
            inject_noise(models, sigma, device)
            print(f"  Gaussian noise N(0, {sigma}) added to all 5 silos")

        # Evaluate
        test_results, test_type_results = evaluate(
            server, model_a, model_b, model_c, model_d, model_e,
            test_loader, device, per_type=True)

        print_results(f"Test (σ={sigma})", test_results, test_type_results)
        all_results[sigma] = test_results

    # ── Print comparison table ───────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  NOISE ROBUSTNESS — COMPARISON TABLE")
    print("  MetaQA | Client5 | BERT + TransE | 5 silos | 2-hop")
    print("=" * 70)
    print(f"  {'σ':<8} {'MRR':<10} {'Hits@1':<10} {'Hits@3':<10} "
          f"{'Hits@5':<10} {'Hits@10':<10}")
    print("-" * 70)

    baseline = all_results.get(0.0, all_results.get(NOISE_SIGMAS[0]))
    for sigma in NOISE_SIGMAS:
        r = all_results[sigma]
        mrr_drop  = baseline['mrr']    - r['mrr']
        h1_drop   = baseline['hits@1'] - r['hits@1']
        print(f"  {sigma:<8} {r['mrr']:<10.4f} {r['hits@1']:<10.4f} "
              f"{r['hits@3']:<10.4f} {r['hits@5']:<10.4f} "
              f"{r['hits@10']:<10.4f}")

    print("-" * 70)
    print("\n  Degradation from baseline (σ=0.0):")
    print(f"  {'σ':<8} {'ΔMRR':<10} {'ΔHits@1':<10} {'ΔHits@3':<10} "
          f"{'ΔHits@5':<10} {'ΔHits@10':<10}")
    print("-" * 70)
    for sigma in NOISE_SIGMAS:
        if sigma == 0.0:
            continue
        r = all_results[sigma]
        print(f"  {sigma:<8} {baseline['mrr']-r['mrr']:<+10.4f} "
              f"{baseline['hits@1']-r['hits@1']:<+10.4f} "
              f"{baseline['hits@3']-r['hits@3']:<+10.4f} "
              f"{baseline['hits@5']-r['hits@5']:<+10.4f} "
              f"{baseline['hits@10']-r['hits@10']:<+10.4f}")
    print("=" * 70)

    # ── Save results to JSON ─────────────────────────────────────────────────
    output = {
        "experiment"  : "noise_robustness",
        "dataset"     : "MetaQA",
        "client"      : "Client5",
        "kge_model"   : "TransE",
        "qa_encoder"  : "BERT",
        "num_silos"   : 5,
        "noise_type"  : "Gaussian N(0, σ)",
        "noise_target": "all entity embeddings (all 5 silos)",
        "timestamp"   : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "results"     : {}
    }
    for sigma in NOISE_SIGMAS:
        r = all_results[sigma]
        output["results"][str(sigma)] = {
            k: round(v, 6) for k, v in r.items()
        }

    out_path = os.path.join(
        os.path.dirname(CHECKPOINT_DIR),
        "noise_robustness_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
