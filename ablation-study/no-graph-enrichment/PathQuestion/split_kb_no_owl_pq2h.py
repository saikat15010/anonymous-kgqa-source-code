# split_kb_no_owl_pq2h.py — Split original Freebase13.txt into 3 silos
# WITHOUT inverse triples (parents↔children, spouse↔spouse)
#
# The original Freebase13.txt already has parents, children, spouse
# as separate relations — but the enrichment added MISSING inverse
# triples. This script uses only the original triples as-is.
#
# Output: /home/islamm9/ISWC/Dataset/PQ2H/data/Client3/silos_no_owl/

import os

KB_PATH  = "/home/islamm9/ISWC/Dataset/PQ2H/Freebase13.txt"
SILO_DIR = "/home/islamm9/ISWC/Dataset/PQ2H/data/Client3/silos_no_owl"
os.makedirs(SILO_DIR, exist_ok=True)

# Same silo partition as enriched version
SILOS = {
    'a': {'parents', 'children', 'spouse'},
    'b': {'gender', 'nationality', 'ethnicity', 'religion', 'cause_of_death'},
    'c': {'profession', 'institution', 'place_of_birth',
          'place_of_death', 'location'},
}

# Load ORIGINAL Freebase13 (no enrichment)
triples = []
with open(KB_PATH, encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) != 3:
            continue
        triples.append((parts[0], parts[1], parts[2]))

print(f"Original Freebase13: {len(triples):,} triples")

# Split into silos
for silo_name, rels in sorted(SILOS.items()):
    silo_triples = [(h, r, t) for h, r, t in triples if r in rels]
    out_path = os.path.join(SILO_DIR, f'kb_silo_{silo_name}.txt')
    with open(out_path, 'w') as f:
        for h, r, t in silo_triples:
            f.write(f"{h}\t{r}\t{t}\n")
    ents = set()
    for h, r, t in silo_triples:
        ents.add(h); ents.add(t)
    print(f"  Silo {silo_name.upper()}: {len(silo_triples):>7,} triples | "
          f"{len(rels)} relations | {len(ents):,} entities")

print(f"\nDone. Silo files → {SILO_DIR}")
