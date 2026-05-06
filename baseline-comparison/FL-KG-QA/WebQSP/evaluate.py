# evaluate.py — Evaluation for Adapted FL-KG-QA (VFL)
# Uses average pooling (d-dim), no topic anchoring.
# Candidates are 1-hop only (controlled by config CANDIDATE_HOP2_CAP=0).

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
            topic_ids     = topic_ids.to(device)
            candidate_ids = candidate_ids.to(device)

            q_embed = server.question_encoder(questions, device)

            # No topic anchoring
            q_final = q_embed

            sim = server.score_candidates(q_final, h_joint, candidate_ids)

            for i, answer_ids in enumerate(answer_ids_batch):
                cands     = candidate_ids[i]
                valid     = cands >= 0
                cand_list = cands[valid].tolist()
                scores    = sim[i][valid].tolist()

                ranked     = sorted(zip(cand_list, scores),
                                    key=lambda x: x[1], reverse=True)
                answer_set = set(answer_ids)

                rank = None
                for pos, (eid, _) in enumerate(ranked):
                    if eid in answer_set:
                        rank = pos + 1
                        break

                overall.update(rank)
                if per_type:
                    type_accs[detect_answer_type(questions[i])].update(rank)

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
