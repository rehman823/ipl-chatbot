"""Extract the commentary ZIP, chunk the text, embed it and store it in ChromaDB.

Usage:
    python ingest.py                 # finds the .zip in this folder
    python ingest.py --zip file.zip  # use a specific ZIP
    python ingest.py --reset         # wipe the collection and start over

Safe to re-run: chunk IDs are deterministic, so nothing is duplicated.
"""
import argparse
import hashlib
import logging
import re
import zipfile
from pathlib import Path
import csv
import chromadb

import config
from ollama_client import check_ollama, embed

log = logging.getLogger("ingest")


# ---------- 1. Find and extract the ZIP ----------
def find_zip(explicit=None):
    if explicit:
        return Path(explicit)
    zips = sorted(config.BASE_DIR.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zips:
        raise FileNotFoundError(f"No .zip file found in {config.BASE_DIR}")
    if len(zips) > 1:
        log.warning("Several ZIPs found, using the newest: %s", zips[0].name)
    return zips[0]


def extract_zip(zip_path):
    config.DATA_DIR.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(config.DATA_DIR)
    log.info("Extracted %s to %s", zip_path.name, config.DATA_DIR)


def list_text_files():
    files = [
        p for p in config.DATA_DIR.rglob("*")
        if p.is_file()
        and p.suffix.lower() in config.TEXT_EXTENSIONS
        and not p.name.startswith(".")
        and "__MACOSX" not in p.parts
    ]
    return sorted(files)

def read_file(path):
    """Plain text is read as-is; CSV rows become 'column: value, ...' lines."""
    if path.suffix.lower() != ".csv":
        return path.read_text(encoding="utf-8", errors="replace")
    lines = []
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        for row in csv.DictReader(f):
            parts = [f"{k.strip()}: {v.replace(chr(10), ' ').strip()}"
                     for k, v in row.items() if k and isinstance(v, str) and v.strip()]
            if parts:
                lines.append(", ".join(parts))
    return "\n".join(lines)

# ---------- 2. Clean and chunk ----------
def clean_text(text):
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)   # control characters
    text = re.sub(r"[ \t]+", " ", text)                          # repeated spaces
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def approx_tokens(text):
    return int(len(text.split()) * config.TOKENS_PER_WORD) + 1


def split_long_line(line, max_tokens):
    """A single very long line is split on words so it still fits in a chunk."""
    if approx_tokens(line) <= max_tokens:
        return [line]
    words = line.split()
    step = max(1, int(max_tokens / config.TOKENS_PER_WORD))
    return [" ".join(words[i:i + step]) for i in range(0, len(words), step)]


def chunk_text(text, max_tokens, overlap_tokens):
    """Group whole lines (ball-by-ball entries stay intact) into overlapping chunks."""
    lines = []
    for line in text.split("\n"):
        if line.strip():
            lines.extend(split_long_line(line.strip(), max_tokens))

    chunks, current, size = [], [], 0
    for line in lines:
        t = approx_tokens(line)
        if current and size + t > max_tokens:
            chunks.append("\n".join(current))
            carry, carry_size = [], 0           # last few lines become the overlap
            for prev in reversed(current):
                pt = approx_tokens(prev)
                if carry_size + pt > overlap_tokens:
                    break
                carry.insert(0, prev)
                carry_size += pt
            current, size = carry, carry_size
        current.append(line)
        size += t
    if current:
        chunks.append("\n".join(current))
    return chunks


# ---------- 3. Embed and store ----------
def chunk_id(source, index, text):
    return hashlib.sha1(f"{source}|{index}|{text}".encode("utf-8")).hexdigest()


def ingest_file(path, collection):
    source = path.relative_to(config.DATA_DIR).as_posix()
    text = clean_text(read_file(path))
    chunks = chunk_text(text, config.CHUNK_TOKENS, config.CHUNK_OVERLAP)
    if not chunks:
        log.warning("%s is empty after cleaning, skipping", source)
        return 0, 0

    match = re.sub(r"[_\-]+", " ", path.stem).strip()
    documents = [f"Match: {match}\n{c}" for c in chunks]   # keep match context in every chunk
    ids = [chunk_id(source, i, d) for i, d in enumerate(documents)]

    # If the file changed since the last run, drop its old chunks
    old_ids = collection.get(where={"source": source}, include=[])["ids"]
    stale = [i for i in old_ids if i not in set(ids)]
    if stale:
        collection.delete(ids=stale)

    # Only embed chunks that are not stored yet (this is what makes re-runs cheap)
    existing = set(collection.get(ids=ids, include=[])["ids"])
    todo = [n for n, i in enumerate(ids) if i not in existing]

    for start in range(0, len(todo), config.EMBED_BATCH_SIZE):
        batch = todo[start:start + config.EMBED_BATCH_SIZE]
        vectors = embed([documents[n] for n in batch], prefix=config.DOC_PREFIX)
        collection.upsert(
            ids=[ids[n] for n in batch],
            documents=[documents[n] for n in batch],
            embeddings=vectors,
            metadatas=[
                {"source": source, "match": match, "chunk_index": n, "total_chunks": len(chunks)}
                for n in batch
            ],
        )
    return len(todo), len(chunks)


def main():
    parser = argparse.ArgumentParser(description="Ingest IPL commentary into ChromaDB")
    parser.add_argument("--zip", help="path to the ZIP (default: newest .zip in this folder)")
    parser.add_argument("--reset", action="store_true", help="delete the collection first")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    check_ollama([config.EMBED_MODEL])
    extract_zip(find_zip(args.zip))
    files = list_text_files()
    if not files:
        raise SystemExit(f"No {sorted(config.TEXT_EXTENSIONS)} files found in {config.DATA_DIR}")

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    if args.reset:
        try:
            client.delete_collection(config.COLLECTION_NAME)
            log.info("Deleted existing collection")
        except Exception:
            pass
    collection = client.get_or_create_collection(
        config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    added = failed = 0
    for n, path in enumerate(files, 1):
        try:
            new, total = ingest_file(path, collection)
            added += new
            log.info("[%d/%d] %s: %d chunks (%d new)", n, len(files), path.name, total, new)
        except Exception as e:     # bad file: log it and keep going
            failed += 1
            log.error("[%d/%d] %s FAILED: %s", n, len(files), path.name, e)

    log.info("Done. %d new chunks, %d files failed, %d chunks in collection",
             added, failed, collection.count())


if __name__ == "__main__":
    main()
