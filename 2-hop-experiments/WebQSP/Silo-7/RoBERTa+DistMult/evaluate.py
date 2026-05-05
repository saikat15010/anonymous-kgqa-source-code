# evaluate.py — Evaluation for WebQSP Client7 RoBERTa+DistMult (7 silos)
# DistMult: (N, d=256) embeddings; h_joint = (N, 7d=1792).

import torch
from collections import defaultdict
from config import USE_TOPIC_ANCHORING


def detect_answer_type(question):
    q = question.lower()
    if any(w in q for w in ["born", "birthplace", "birth place",
                             "hometown", "where was", "place of birth"]):
        return "place_birth"
    if any(w in q for w in ["nationality", "citizen", "from", "country"]):
        return "nationality"
    if any(w in q for w in ["profession", "occupation", "job", "career"]):
        return "profession"
    if any(w in q for w in ["spouse", "married", "husband", "wife",
                             "child", "children", "son", "daughter",
                             "parent", "father", "mother"]):
        return "person"
    if any(w in q for w in ["died", "death", "cause of death", "killed"]):
        return "cause_death"
    if any(w in q for w in ["capital", "located", "location",
                             "where is", "state", "city", "county"]):
        return "location"
    if any(w in q for w in ["president", "leader", "prime minister",
                             "director", "directed by", "actor", "actress",
                             "cast", "star"]):
        return "person"
    if any(w in q for w in ["founded", "established", "headquarters"]):
        return "organization"
    if any(w in q for w in ["language", "speak", "official language"]):
        return "language"
    if any(w in q for w in ["genre", "type of", "kind of"]):
        return "genre"
    if any(w in q for w in ["award", "win", "won", "prize"]):
        return "award"
    if any(w in q for w in ["team", "play for", "club"]):
        return "team"
    if any(w in q for w in ["album", "song", "music", "record"]):
        return "music"
    return "other"


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


def evaluate(server, model_a, model_b, model_c, model_d,
             model_e, model_f, model_g,
             qa_loader, device, per_type=True):
    """Evaluate with 7 silo DistMult models."""
    server.eval()
    for m in [model_a, model_b, model_c, model_d, model_e, model_f, model_g]:
        m.eval()

    overall   = MetricAccumulator()
    type_accs = defaultdict(MetricAccumulator)

    with torch.no_grad():
        h_a = model_a.get_entity_embeddings().to(device)
        h_b = model_b.get_entity_embeddings().to(device)
        h_c = model_c.get_entity_embeddings().to(device)
        h_d = model_d.get_entity_embeddings().to(device)
        h_e = model_e.get_entity_embeddings().to(device)
        h_f = model_f.get_entity_embeddings().to(device)
        h_g = model_g.get_entity_embeddings().to(device)
        h_joint = server.fuse(h_a, h_b, h_c, h_d, h_e, h_f, h_g)  # (N, 1792)

        for questions, topic_ids, answer_ids_batch, candidate_ids in qa_loader:
            topic_ids     = topic_ids.to(device)
            candidate_ids = candidate_ids.to(device)
            q_proj  = server.question_encoder(questions, device)
            q_final = q_proj + h_joint[topic_ids] if USE_TOPIC_ANCHORING \
                      else q_proj
            sim = server.score_candidates(q_final, h_joint, candidate_ids)

            for i, answer_ids in enumerate(answer_ids_batch):
                cands     = candidate_ids[i]
                valid     = cands >= 0
                ranked    = sorted(zip(cands[valid].tolist(),
                                       sim[i][valid].tolist()),
                                   key=lambda x: x[1], reverse=True)
                answer_set = set(answer_ids)
                rank = next((pos + 1 for pos, (eid, _) in enumerate(ranked)
                             if eid in answer_set), None)
                overall.update(rank)
                if per_type:
                    type_accs[detect_answer_type(questions[i])].update(rank)

    per_type_results = {t: acc.results()
                        for t, acc in type_accs.items()} if per_type else {}
    return overall.results(), per_type_results


def print_results(split_name, overall, per_type=None):
    def _row(label, r, indent="  "):
        return (f"{indent}[{label}]  MRR:{r['mrr']:.4f}"
                f"  H@1:{r['hits@1']:.4f}  H@3:{r['hits@3']:.4f}"
                f"  H@5:{r['hits@5']:.4f}  H@10:{r['hits@10']:.4f}"
                f"  N:{r['total']}")
    print(f"\n{'─'*70}")
    print(f"  {split_name} Results"
          f"  [RoBERTa + DistMult | WebQSP | Client7 — 7 silos]")
    print(f"{'─'*70}")
    print(_row("Overall       ", overall))
    if per_type:
        print("  Per Answer-Type:")
        for atype in ["place_birth", "nationality", "profession",
                      "cause_death", "location", "organization",
                      "language", "genre", "award", "team",
                      "music", "person", "other"]:
            if atype in per_type:
                print(_row(f"{atype:<14}", per_type[atype], indent="    "))
    print(f"{'─'*70}")
