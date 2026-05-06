# evaluate.py — Evaluation for Adapted EmbedKGQA (VFL)
#
# Key difference: ranks ALL entities globally (not just candidates).
# This faithfully reflects EmbedKGQA's design where every entity
# is a potential answer.

import torch
from collections import defaultdict


def detect_answer_type(question):
    q = question.lower()
    if any(w in q for w in ["year", "when", "release date", "release year"]):
        return "year"
    if any(w in q for w in ["language", "spoken", "languages"]):
        return "language"
    if any(w in q for w in ["genre", "type", "kind", "category", "types"]):
        return "genre"
    if any(w in q for w in ["who", "director", "actor", "actress",
                             "screenwriter", "writer", "starred", "co-star",
                             "person", "directed by", "written by"]):
        return "person"
    if any(w in q for w in ["movie", "film", "films", "movies", "same"]):
        return "movie"
    return "unknown"


class MetricAccumulator:
    def __init__(self):
        self.hits_1 = self.hits_3 = self.hits_5 = self.hits_10 = 0
        self.mrr    = 0.0
        self.total  = 0

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
        return {
            "mrr"    : self.mrr    / n,
            "hits@1" : self.hits_1 / n,
            "hits@3" : self.hits_3 / n,
            "hits@5" : self.hits_5 / n,
            "hits@10": self.hits_10 / n,
            "total"  : self.total,
        }


def evaluate(server, model_a, model_b, model_c, qa_loader, device,
             per_type=True):
    """
    Evaluate by ranking ALL entities globally (EmbedKGQA style).
    No candidate filtering — the rank is computed over the full entity set.
    """
    server.eval()
    model_a.eval(); model_b.eval(); model_c.eval()

    overall   = MetricAccumulator()
    type_accs = defaultdict(MetricAccumulator)

    with torch.no_grad():
        h_a = model_a.get_entity_embeddings().to(device)
        h_b = model_b.get_entity_embeddings().to(device)
        h_c = model_c.get_entity_embeddings().to(device)
        h_joint = server.fuse(h_a, h_b, h_c)  # (N, d) — avg pooled

        for questions, topic_ids, answer_ids_batch, candidate_ids in qa_loader:
            q_embed = server.question_encoder(questions, device)  # (B, d)

            # No topic anchoring
            q_final = q_embed

            # Score ALL entities globally (not just candidates)
            scores_all = server.score_all_entities(q_final, h_joint)  # (B, N)

            for i, answer_ids in enumerate(answer_ids_batch):
                if not answer_ids:
                    overall.update(None)
                    continue

                scores = scores_all[i]  # (N,)
                answer_set = set(answer_ids)

                # Sort all entities by score (descending)
                _, sorted_indices = torch.sort(scores, descending=True)
                sorted_indices = sorted_indices.cpu().tolist()

                # Find rank of first correct answer
                rank = None
                for pos, eid in enumerate(sorted_indices):
                    if eid in answer_set:
                        rank = pos + 1
                        break

                overall.update(rank)

                if per_type:
                    atype = detect_answer_type(questions[i])
                    type_accs[atype].update(rank)

    per_type_results = {t: acc.results() for t, acc in type_accs.items()} \
                       if per_type else {}
    return overall.results(), per_type_results


def print_results(split_name, overall, per_type=None):
    def _row(label, r, indent="  "):
        return (f"{indent}[{label}]"
                f"  MRR: {r['mrr']:.4f}"
                f"  |  Hits@1: {r['hits@1']:.4f}"
                f"  |  Hits@3: {r['hits@3']:.4f}"
                f"  |  Hits@5: {r['hits@5']:.4f}"
                f"  |  Hits@10: {r['hits@10']:.4f}"
                f"  |  N: {r['total']}")
    print(f"\n{'─'*70}")
    print(f"  {split_name} Results")
    print(f"{'─'*70}")
    print(_row("Overall   ", overall))
    if per_type:
        print(f"\n  Per Answer-Type Breakdown:")
        for atype in ["person", "movie", "year", "genre", "language", "unknown"]:
            if atype in per_type:
                print(_row(f"{atype:<10}", per_type[atype], indent="    "))
    print(f"{'─'*70}")