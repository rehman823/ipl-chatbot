"""Tiny helper for talking to a local Ollama server."""
import time

import requests

import config


def check_ollama(models):
    """Raise a clear error if Ollama is down or a model has not been pulled."""
    try:
        r = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {config.OLLAMA_URL}. Is it running? ({e})"
        )
    installed = [m["name"] for m in r.json().get("models", [])]
    for model in models:
        if not any(n == model or n.startswith(model + ":") for n in installed):
            raise RuntimeError(f"Model '{model}' not found. Run: ollama pull {model}")


def embed(texts, prefix=""):
    """Return one embedding per text, using nomic-embed-text via Ollama."""
    payload = {"model": config.EMBED_MODEL, "input": [prefix + t for t in texts]}
    last_error = None
    for attempt in range(3):
        try:
            r = requests.post(f"{config.OLLAMA_URL}/api/embed", json=payload, timeout=120)
            r.raise_for_status()
            return r.json()["embeddings"]
        except (requests.RequestException, KeyError) as e:
            last_error = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Embedding request failed after 3 attempts: {last_error}")
