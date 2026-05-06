# dataset.py — Centralized baseline data loaders
# Merges all silo KBs into a single KB. Supports both | and \t delimiters.

import re
import torch
from torch.utils.data import Dataset
from collections import defaultdict, OrderedDict
from config import KB_DELIMITER


def build_index_from_paths(kb_paths):
    """Read ALL silo KBs and merge into a single triple set."""
    triples, entities, relations = [], set(), set()
    for kb_path in kb_paths:
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


def build_neighbor_index(kb_paths, entity2id, max_neighbors=100):
    """Bidirectional neighbor index from all KB files."""
    neighbors = defaultdict(set)
    for kb_path in kb_paths:
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
    """Auto-detect format: MetaQA (2 cols with brackets) vs tabular (3 cols)."""
    with open(qa_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    if len(first_line.split("\t")) == 2:
        return parse_qa_file_metaqa(qa_path)
    else:
        return parse_qa_file_tabular(qa_path)


class QADataset(Dataset):
    def __init__(self, qa_path, entity2id):
        self.samples = []
        for question, topic, answers in parse_qa_file(qa_path):
            topic_id   = entity2id.get(topic, -1)
            answer_ids = [entity2id[a] for a in answers if a in entity2id]
            if answer_ids and topic_id != -1:
                self.samples.append((question, topic_id, answer_ids))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx): return self.samples[idx]
