# split_kb_no_owl_metaqa.py — Split original MetaQA kb.txt into 3 silos
# WITHOUT inverse/property-chain axioms
#
# Original kb.txt relations (9 total):
#   directed_by, written_by, starred_actors, has_tags,
#   release_year, in_language, has_genre,
#   has_imdb_rating, has_imdb_votes
#
# OWL-enriched relations (added by enrichment, EXCLUDED here):
#   directed, wrote, associated_actor,
#   associated_genre, associated_language, associated_year
#
# Output: /home/islamm9/ISWC/Dataset/MetaQA/Client3/data/silos_no_owl/

import os

KB_PATH  = "/home/islamm9/ISWC/Dataset/MetaQA/Client3/data/kb.txt"
SILO_DIR = "/home/islamm9/ISWC/Dataset/MetaQA/Client3/data/silos_no_owl"
os.makedirs(SILO_DIR, exist_ok=True)

# Same silo partition as enriched version, but only original relations
SILOS = {
    'a': {'directed_by', 'written_by'},
    'b': {'starred_actors', 'has_tags'},
    'c': {'release_year', 'in_language', 'has_genre',
          'has_imdb_rating', 'has_imdb_votes'},
}

# Load original KB
triples = []
with open(KB_PATH, encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('|')
        if len(parts) != 3:
            continue
        triples.append((parts[0], parts[1], parts[2]))

print(f"Original KB: {len(triples):,} triples")

# Split into silos
for silo_name, rels in sorted(SILOS.items()):
    silo_triples = [(h, r, t) for h, r, t in triples if r in rels]
    out_path = os.path.join(SILO_DIR, f'kb_silo_{silo_name}.txt')
    with open(out_path, 'w') as f:
        for h, r, t in silo_triples:
            f.write(f"{h}|{r}|{t}\n")
    ents = set()
    for h, r, t in silo_triples:
        ents.add(h); ents.add(t)
    print(f"  Silo {silo_name.upper()}: {len(silo_triples):>7,} triples | "
          f"{len(rels)} relations ({', '.join(sorted(rels))}) | "
          f"{len(ents):,} entities")

print(f"\nDone. Silo files → {SILO_DIR}")
