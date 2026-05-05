# dataset.py — Entity/relation indexing + QA/KGE data loaders (PQ2H)
#
# Key differences from MetaQA version:
#   1. KB delimiter is TAB (\t) not pipe (|)
#   2. QA format: question\ttopic_entity\tanswer  (topic+answer are explicit columns)
#      No bracket parsing needed — topic entity is not embedded in question text.
#   3. Each QA line has exactly one answer (not pipe-separated multiple answers).

import torch
from torch.utils.data import Dataset
from collections import defaultdict


# ── Indexing ───────────────────────────────────────────────────────────────────

def build_index(kb_path):
    """Read a silo KB (tab-separated) and return triples, entity2id, relation2id."""
    triples, entities, relations = [], set(), set()
    with open(kb_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
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

    Note: KB files are tab-separated for PQ2H.
    """
    neighbors = defaultdict(set)
    for kb_path in kb_paths:
        with open(kb_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
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
                           hop1_cap=50, hop2_cap=20):
    """
    Precompute 2-hop candidate sets for ALL entities ONCE at startup.
    Returns dict: { entity_id -> LongTensor of candidate ids }

    Replaces the per-batch Python loop — makes training ~100x faster.
    """
    print("  Precomputing 2-hop candidate sets...", flush=True)
    candidates = {}
    for eid in entity_ids:
        hop1 = list(neighbor_index.get(eid, []))[:hop1_cap]
        hop2 = set()
        for nb in hop1:
            hop2.update(neighbor_index.get(nb, [])[:hop2_cap])
        cands = list({eid} | set(hop1) | hop2)
        candidates[eid] = torch.tensor(cands, dtype=torch.long)
    avg = sum(len(v) for v in candidates.values()) // max(len(candidates), 1)
    print(f"  Done. Avg candidates per entity: {avg}", flush=True)
    return candidates


# ── KGE Dataset ───────────────────────────────────────────────────────────────

class KGEDataset(Dataset):
    """Yields (head_id, relation_id, tail_id) for KGE training."""

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
    """
    Parse PQ2H QA file.

    Format (tab-separated, 3 columns):
        question \\t topic_entity \\t answer

    Unlike MetaQA, the topic entity is a standalone column (not embedded
    in brackets inside the question), and each line has exactly one answer.

    Returns: list of (question, topic_entity, [answer])
    """
    samples = []
    with open(qa_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            question, topic_entity, answer = parts
            question     = question.strip()
            topic_entity = topic_entity.strip()
            answer       = answer.strip()
            if question and topic_entity and answer:
                samples.append((question, topic_entity, [answer]))
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
