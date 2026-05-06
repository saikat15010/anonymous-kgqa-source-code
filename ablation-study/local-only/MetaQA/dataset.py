# dataset.py — Local-Only baseline data loaders
# Builds per-silo neighbor indices. Assigns each QA sample to the silo
# that has the most triples involving the topic entity.

import re
import torch
from torch.utils.data import Dataset
from collections import defaultdict, OrderedDict
from config import KB_DELIMITER


def build_index(kb_path):
    triples, entities, relations = [], set(), set()
    with open(kb_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(KB_DELIMITER)
            if len(parts) != 3:
                continue
            h, r, t = parts
            triples.append((h, r, t))
            entities.update([h, t])
            relations.add(r)
    entity2id   = {e: i for i, e in enumerate(sorted(entities))}
    relation2id = {r: i for i, r in enumerate(sorted(relations))}
    return triples, entity2id, relation2id


def build_neighbor_index_single(kb_path, entity2id, max_neighbors=100):
    """Build neighbor index for a single silo KB."""
    neighbors = defaultdict(set)
    with open(kb_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(KB_DELIMITER)
            if len(parts) != 3:
                continue
            h, r, t = parts
            h_id = entity2id.get(h)
            t_id = entity2id.get(t)
            if h_id is not None and t_id is not None:
                neighbors[h_id].add(t_id)
                neighbors[t_id].add(h_id)
    return {k: list(v)[:max_neighbors] for k, v in neighbors.items()}


def count_entity_triples(kb_path, entity2id):
    """Count how many triples each entity appears in for a given silo."""
    counts = defaultdict(int)
    with open(kb_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(KB_DELIMITER)
            if len(parts) != 3:
                continue
            h, r, t = parts
            h_id = entity2id.get(h)
            t_id = entity2id.get(t)
            if h_id is not None:
                counts[h_id] += 1
            if t_id is not None:
                counts[t_id] += 1
    return counts


def assign_topic_to_silo(topic_id, silo_counts):
    """
    Assign a topic entity to the silo with the most triples involving it.
    Returns silo index (0, 1, 2) or -1 if not found in any silo.
    """
    best_silo = -1
    best_count = 0
    for silo_idx, counts in enumerate(silo_counts):
        c = counts.get(topic_id, 0)
        if c > best_count:
            best_count = c
            best_silo = silo_idx
    return best_silo


def precompute_candidates(entity_ids, neighbor_index,
                          hop1_cap=50, hop2_cap=20):
    print("  Precomputing 2-hop candidate sets ...", flush=True)
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


class KGEDataset(Dataset):
    def __init__(self, triples, entity2id, relation2id):
        self.data = []
        for h, r, t in triples:
            h_id = entity2id.get(h)
            r_id = relation2id.get(r)
            t_id = entity2id.get(t)
            if h_id is not None and r_id is not None and t_id is not None:
                self.data.append((h_id, r_id, t_id))

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        h, r, t = self.data[idx]
        return torch.tensor(h), torch.tensor(r), torch.tensor(t)


# ── QA parsing ────────────────────────────────────────────────────────────────

def parse_qa_file_metaqa(qa_path):
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
            topic = match.group(1)
            question = re.sub(r"\[(.+?)\]", r"\1", question_raw).strip()
            answers = [a.strip() for a in answers_raw.split("|") if a.strip()]
            if answers:
                samples.append((question, topic, answers))
    return samples


def parse_qa_file_tabular(qa_path):
    grouped = OrderedDict()
    with open(qa_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            q, topic, ans = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if not (q and topic and ans):
                continue
            key = (q, topic)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(ans)
    return [(q, t, answers) for (q, t), answers in grouped.items()]


def parse_qa_file(qa_path):
    with open(qa_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    if len(first_line.split("\t")) == 2:
        return parse_qa_file_metaqa(qa_path)
    else:
        return parse_qa_file_tabular(qa_path)


class QADataset(Dataset):
    """
    Local-only QA dataset.
    Each sample includes (question, topic_id, answer_ids, assigned_silo).
    """
    def __init__(self, qa_path, entity2id, silo_counts):
        self.samples = []
        for question, topic, answers in parse_qa_file(qa_path):
            topic_id   = entity2id.get(topic, -1)
            answer_ids = [entity2id[a] for a in answers if a in entity2id]
            if answer_ids and topic_id != -1:
                silo_idx = assign_topic_to_silo(topic_id, silo_counts)
                if silo_idx >= 0:
                    self.samples.append((question, topic_id, answer_ids, silo_idx))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx): return self.samples[idx]
