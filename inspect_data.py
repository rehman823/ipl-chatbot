"""Report the real structure of the data source (CSV files in data/ + the ChromaDB collection).

Usage:  python inspect_data.py
Paste the output back so the golden set can be built against your real columns.
"""
import csv
from collections import Counter

import chromadb

import config

csv_files = sorted(p for p in config.DATA_DIR.rglob("*.csv") if "__MACOSX" not in p.parts)
print(f"== CSV files in {config.DATA_DIR}: {len(csv_files)}")

schemas = {}
for p in csv_files:
    with open(p, newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        rows = sum(1 for _ in reader)
        schemas[p] = (tuple(reader.fieldnames or ()), rows)

for p, (cols, rows) in list(schemas.items())[:15]:
    print(f"  {p.relative_to(config.DATA_DIR)}  rows={rows}")
if len(csv_files) > 15:
    print(f"  ... and {len(csv_files) - 15} more")
print(f"  total rows: {sum(r for _, r in schemas.values())}")

distinct = Counter(cols for cols, _ in schemas.values())
print(f"\n== Distinct column layouts: {len(distinct)}")
for cols, n in distinct.most_common(3):
    print(f"  {n} file(s): {list(cols)}")

if csv_files:
    first = csv_files[0]
    print(f"\n== Sample rows from {first.name}")
    with open(first, newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for r in rows[:3]:
        print("  ", {k: (v[:80] if isinstance(v, str) else v) for k, v in r.items()})
    print("\n== Low-cardinality columns in that file (<=12 distinct values)")
    for col in rows[0].keys() if rows else []:
        vals = Counter(r.get(col) for r in rows)
        if len(vals) <= 12:
            print(f"  {col}: {dict(vals.most_common(12))}")

print("\n== ChromaDB collection")
try:
    col = chromadb.PersistentClient(path=str(config.CHROMA_DIR)).get_collection(config.COLLECTION_NAME)
    print(f"  chunks: {col.count()}")
    sample = col.get(limit=1, include=["documents", "metadatas"])
    if sample["ids"]:
        print("  metadata:", sample["metadatas"][0])
        print("  document:", sample["documents"][0][:400].replace("\n", " | "))
except Exception as e:
    print("  not available:", e)
