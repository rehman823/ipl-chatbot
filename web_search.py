"""Live web search through Tavily, for IPL matches that are not in the dataset (e.g. 2026).

Test it on its own:  python web_search.py "IPL 2026 latest match result"
"""
import re
import sys

import requests

import config

LIVE_WORDS = re.compile(
    r"\b(live|today|tonight|yesterday|latest|current|currently|recent|recently|right now|"
    r"upcoming|next match|this season|schedule|points table|standings)\b", re.I)
NEWS_WORDS = re.compile(r"\b(live|today|tonight|yesterday|latest|right now|recent|recently)\b", re.I)


class WebSearchError(Exception):
    """Raised with a short, user-readable message when the web search fails."""


def is_enabled():
    return bool(config.TAVILY_API_KEY)


def needs_live(question):
    """True when the question is about a season after the dataset, or about 'now'."""
    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", question)]
    if any(y > config.DATASET_LAST_SEASON for y in years):
        return True
    return bool(LIVE_WORDS.search(question))


def search_web(question):
    """Return a list of {title, url, content, published_date, score} from Tavily."""
    if not is_enabled():
        raise WebSearchError("TAVILY_API_KEY is not set.")

    query = question if re.search(r"\bipl\b", question, re.I) else f"IPL cricket {question}"
    body = {
        "query": query,
        "topic": "news" if NEWS_WORDS.search(question) else "general",
        "search_depth": "basic",
        "max_results": config.TAVILY_MAX_RESULTS,
        "include_answer": False,
    }
    if config.TAVILY_PREFER_DOMAINS:
        body["include_domains"] = config.TAVILY_PREFER_DOMAINS
        body["include_domains_mode"] = "prefer"
    if config.TAVILY_TIME_RANGE:
        body["time_range"] = config.TAVILY_TIME_RANGE

    try:
        r = requests.post(
            config.TAVILY_URL,
            json=body,
            headers={"Authorization": f"Bearer {config.TAVILY_API_KEY}"},
            timeout=20,
        )
    except requests.RequestException as e:
        raise WebSearchError(f"Could not reach Tavily: {e}")

    if r.status_code == 401:
        raise WebSearchError("Tavily rejected the API key (401). Check TAVILY_API_KEY.")
    if r.status_code in (429, 432, 433):
        raise WebSearchError(f"Tavily rate or plan limit reached ({r.status_code}).")
    if not r.ok:
        raise WebSearchError(f"Tavily returned HTTP {r.status_code}: {r.text[:200]}")

    results = []
    for item in r.json().get("results", []):
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": (item.get("content") or "")[:config.TAVILY_SNIPPET_CHARS],
            "published_date": item.get("published_date"),
            "score": item.get("score"),
        })
    return results


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "IPL 2026 latest match result"
    print(f"needs_live: {needs_live(q)}")
    try:
        for res in search_web(q):
            print(f"\n- {res['title']}\n  {res['url']}  ({res['published_date']})\n  {res['content'][:200]}")
    except WebSearchError as e:
        print("ERROR:", e)
