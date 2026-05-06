# src/dataset.py — Entity/relation indexing + QA/KGE data loaders
# MetaQA | 3-hop extension
#
# CHANGES FROM 2-HOP VERSION:
#   - precompute_candidates() now supports num_hops=2 or num_hops=3
#   - Added hop3_cap parameter for 3-hop candidate expansion
#   - Everything else is identical (KB format, QA format, KGE dataset)

import re
import torch
from torch.utils.data import Dataset
from collections import defaultdict


# ── Indexing ───────────────────────────────────────────────────────────────────

def build_index(kb_path):
    """Read a silo KB and return triples, entity2id, relation2id."""
    triples, entities, relations = [], set(), set()
    with open(kb_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) != 3:
                continue
            h, r, t = parts
            triples.append((h, r, t))
            entities.update([h, t])
            relations.add(r)
    entity2id   = {e: i for i, e in enumerate(sorted(entities))}
    relation2id = {r: i for i, r in enumerate(sorted(relations))}
    return triples, entity2id, relation2id


def build_shared_entity_index(*entity2id_dicts):
    """Union of entity sets across all silos → shared index."""
    all_entities = set()
    for e2id in entity2id_dicts:
        all_entities |= set(e2id.keys())
    return {e: i for i, e in enumerate(sorted(all_entities))}


def build_neighbor_index(kb_paths, shared_entity2id, max_neighbors=100):
    """
    Bidirectional neighbor index: forward (h→t) AND reverse (t→h).

    Forward alone gives 0% answer coverage for person-topic questions
    because persons are only tails in the original KB.
    Bidirectional gives ~90% coverage.
    """
    neighbors = defaultdict(set)
    for kb_path in kb_paths:
        with open(kb_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) != 3:
                    continue
                h, r, t = parts
                h_id = shared_entity2id.get(h)
                t_id = shared_entity2id.get(t)
                if h_id is not None and t_id is not None:
                    neighbors[h_id].add(t_id)
                    neighbors[t_id].add(h_id)
    return {k: list(v)[:max_neighbors] for k, v in neighbors.items()}


def precompute_candidates(entity_ids, neighbor_index,
                          hop1_cap=50, hop2_cap=20,
                          hop3_cap=0, num_hops=2):
    """
    Precompute candidate sets for ALL entities ONCE at startup.
    Returns dict: { entity_id -> LongTensor of candidate ids }

    Supports both 2-hop and 3-hop expansion:
      num_hops=2 : topic → hop1 → hop2  (original behavior)
      num_hops=3 : topic → hop1 → hop2 → hop3  (new for 3-hop QA)

    For 3-hop, candidate sets are larger. Per-hop caps keep them
    manageable: 50 * 20 * 10 = 10,000 worst case (usually much less
    due to overlap).
    """
    print(f"  Precomputing {num_hops}-hop candidate sets ...", flush=True)
    candidates = {}
    for eid in entity_ids:
        # Hop 1
        hop1 = list(neighbor_index.get(eid, []))[:hop1_cap]

        # Hop 2
        hop2 = set()
        for nb in hop1:
            hop2.update(neighbor_index.get(nb, [])[:hop2_cap])

        if num_hops >= 3 and hop3_cap > 0:
            # Hop 3
            hop3 = set()
            for nb2 in hop2:
                hop3.update(neighbor_index.get(nb2, [])[:hop3_cap])
            cands = list({eid} | set(hop1) | hop2 | hop3)
        else:
            cands = list({eid} | set(hop1) | hop2)

        candidates[eid] = torch.tensor(cands, dtype=torch.long)

    avg = sum(len(v) for v in candidates.values()) // max(len(candidates), 1)
    print(f"  Done. Avg candidates per entity: {avg}", flush=True)
    return candidates


# ── KGE Dataset ───────────────────────────────────────────────────────────────

class KGEDataset(Dataset):
    """Yields (head_id, relation_id, tail_id) for TransE training."""

    def __init__(self, triples, shared_entity2id, relation2id):
        self.data = []
        for h, r, t in triples:
            h_id = shared_entity2id.get(h)
            r_id = relation2id.get(r)
            t_id = shared_entity2id.get(t)
            if h_id is not None and r_id is not None and t_id is not None:
                self.data.append((h_id, r_id, t_id))
        self.num_entities  = len(shared_entity2id)
        self.num_relations = len(relation2id)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        h, r, t = self.data[idx]
        return torch.tensor(h), torch.tensor(r), torch.tensor(t)


# ── QA Dataset ────────────────────────────────────────────────────────────────

def parse_qa_file(qa_path):
    """Parse MetaQA QA file → list of (question, topic_entity, [answers])."""
    samples = []
    with open(qa_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            question_raw, answers_raw = parts
            match = re.search(r"\[(.+?)\]", question_raw)
            if not match:
                continue
            topic_entity   = match.group(1)
            question_clean = re.sub(r"\[(.+?)\]", r"\1", question_raw).strip()
            answers = [a.strip() for a in answers_raw.split("|") if a.strip()]
            if answers:
                samples.append((question_clean, topic_entity, answers))
    return samples


class QADataset(Dataset):
    """
    Yields (question_str, topic_entity_id, answer_ids).

    topic_entity_id:
      - Used for candidate filtering (always)
      - Used for topic anchoring (if USE_TOPIC_ANCHORING=True in config)
    """

    def __init__(self, qa_path, shared_entity2id):
        self.samples = []
        for question, topic_entity, answers in parse_qa_file(qa_path):
            topic_id   = shared_entity2id.get(topic_entity, -1)
            answer_ids = [shared_entity2id[a] for a in answers
                          if a in shared_entity2id]
            if answer_ids and topic_id != -1:
                self.samples.append((question, topic_id, answer_ids))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx): return self.samples[idx]
