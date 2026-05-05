# dataset.py — Entity/relation indexing + KGE/QA data loaders
# WebQSP | Client3 | 3 silos | BERT+DistMult
#
# KB format : head \t relation \t tail
# QA format : question \t topic_entity \t answer  (one row per answer)
# Model-agnostic — identical across all WebQSP Client3 experiments.

import torch
from torch.utils.data import Dataset
from collections import defaultdict, OrderedDict


def build_index(kb_path: str):
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
    all_entities = set()
    for e2id in entity2id_dicts:
        all_entities |= set(e2id.keys())
    return {e: i for i, e in enumerate(sorted(all_entities))}


def build_neighbor_index(kb_paths: list,
                         shared_entity2id: dict,
                         max_neighbors: int = 200) -> dict:
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


def precompute_candidates(entity_ids: list,
                          neighbor_index: dict,
                          hop1_cap: int = 100,
                          hop2_cap: int = 30) -> dict:
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
    def __init__(self, triples, shared_entity2id, relation2id):
        self.data = []
        for h, r, t in triples:
            h_id = shared_entity2id.get(h)
            r_id = relation2id.get(r)
            t_id = shared_entity2id.get(t)
            if h_id is not None and r_id is not None and t_id is not None:
                self.data.append((h_id, r_id, t_id))

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        h, r, t = self.data[idx]
        return torch.tensor(h), torch.tensor(r), torch.tensor(t)


def parse_qa_file(qa_path: str) -> list:
    samples = OrderedDict()
    with open(qa_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            question, topic, answer = parts
            question, topic, answer = (question.strip(),
                                       topic.strip(), answer.strip())
            if not (question and topic and answer):
                continue
            key = (question, topic)
            if key not in samples:
                samples[key] = []
            samples[key].append(answer)
    return [(q, t, answers) for (q, t), answers in samples.items()]


class QADataset(Dataset):
    def __init__(self, qa_path: str, shared_entity2id: dict):
        self.samples = []
        for question, topic_entity, answers in parse_qa_file(qa_path):
            topic_id   = shared_entity2id.get(topic_entity, -1)
            answer_ids = [shared_entity2id[a] for a in answers
                          if a in shared_entity2id]
            if answer_ids and topic_id != -1:
                self.samples.append((question, topic_id, answer_ids))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx): return self.samples[idx]
