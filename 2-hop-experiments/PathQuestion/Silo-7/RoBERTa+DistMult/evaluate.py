# evaluate.py — Evaluation for Client7 RoBERTa+DistMult (7 silos, PQ2H)

import torch
from collections import defaultdict
from config import USE_TOPIC_ANCHORING


def detect_answer_type(question):
    """
    Infer answer type from question keywords.
    Silo mapping (Client7):
      A — Marriage    : spouse
      B — Lineage     : parents, children
      C — Vitals      : gender, cause_of_death
      D — Identity    : ethnicity, religion
      E — Citizenship : nationality
      F — Occupation  : profession, institution
      G — Places      : place_of_birth, place_of_death, location
    """
    q = question.lower()
    if any(w in q for w in ["husband", "wife", "spouse", "married",
                             "other half", "partner"]):
        return "family"
    if any(w in q for w in ["child", "children", "offspring", "son",
                             "daughter", "kids"]):
        return "lineage"
    if any(w in q for w in ["parent", "mother", "father", "mom",
                             "dad", "sibling"]):
        return "lineage"
    if any(w in q for w in ["gender", "sex"]):
        return "gender"
    if any(w in q for w in ["cause of death", "die", "died", "death",
                             "killed", "passed"]):
        return "cause_death"
    if any(w in q for w in ["ethnic", "ethnicity", "race"]):
        return "ethnicity"
    if any(w in q for w in ["religion", "faith", "religious", "belief",
                             "worship"]):
        return "religion"
    if any(w in q for w in ["nationality", "citizen", "citizenship"]):
        return "nationality"
    if any(w in q for w in ["institution", "organization", "university",
                             "school", "college", "work for", "employ"]):
        return "institution"
    if any(w in q for w in ["profession", "job", "occupation",
                             "career", "work as", "vocation"]):
        return "profession"
    if any(w in q for w in ["born", "birth", "birthplace",
                             "place of birth", "hometown"]):
        return "place"
    if any(w in q for w in ["place of death", "buried"]):
        return "place"
    if any(w in q for w in ["location", "located", "live",
                             "reside", "stay"]):
        return "place"
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


def evaluate(server, model_a, model_b, model_c, model_d,
             model_e, model_f, model_g,
             qa_loader, device, per_type=True):
    """7-silo DistMult evaluation with RoBERTa encoder."""
    server.eval()
    for m in [model_a, model_b, model_c, model_d,
              model_e, model_f, model_g]:
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
        h_joint = server.fuse(h_a, h_b, h_c, h_d, h_e, h_f, h_g)

        for questions, topic_ids, answer_ids_batch, candidate_ids in qa_loader:
            topic_ids     = topic_ids.to(device)
            candidate_ids = candidate_ids.to(device)

            q_embed = server.question_encoder(questions, device)
            if USE_TOPIC_ANCHORING:
                q_final = q_embed + h_joint[topic_ids]
            else:
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
    print(f"  {split_name} Results  "
          f"[RoBERTa + DistMult | PQ2H | Client7 — 7 silos]")
    print(f"{'─'*70}")
    print(_row("Overall     ", overall))
    if per_type:
        print(f"\n  Per Answer-Type Breakdown:")
        for atype in ["family", "lineage", "gender", "cause_death",
                      "ethnicity", "religion", "nationality",
                      "institution", "profession", "place", "unknown"]:
            if atype in per_type:
                print(_row(f"{atype:<12}", per_type[atype], indent="    "))
    print(f"{'─'*70}")
