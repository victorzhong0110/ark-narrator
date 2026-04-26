"""
Arknights story script fetcher.
Source: Kengxxiao/ArknightsGameData (official game data, public GitHub repo)

Fetches all story .txt files listed in story_review_table.json.
Path mapping: storyInfo "info/X/Y" -> "story/X/Y.txt"
"""

import json
import time
import logging
import requests
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DIR = Path("./data/raw")
STORY_DIR = Path("./data/raw/stories")
RAW_DIR.mkdir(parents=True, exist_ok=True)
STORY_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = (
    "https://raw.githubusercontent.com/Kengxxiao/ArknightsGameData"
    "/master/zh_CN/gamedata"
)

TABLES = {
    "character_table": f"{BASE_URL}/excel/character_table.json",
    "story_review":    f"{BASE_URL}/excel/story_review_table.json",
}


def fetch_json(url: str, name: str) -> dict:
    cache = RAW_DIR / f"{name}.json"
    if cache.exists():
        logger.info(f"Using cached {name}.json")
        with open(cache, encoding="utf-8") as f:
            return json.load(f)
    logger.info(f"Fetching {name} ...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved -> {cache}")
    return data


def story_info_to_path(story_info: str) -> str:
    """
    Convert storyInfo field to actual file path.
    e.g. "info/activities/a001/level_a001_01_beg"
      -> "story/activities/a001/level_a001_01_beg.txt"
    """
    if story_info.startswith("info/"):
        story_info = story_info[5:]
    return f"story/{story_info}.txt"


def fetch_story(story_info: str, delay: float = 0.05) -> str | None:
    """Fetch a single story script. Returns raw text or None on failure."""
    # Use flat cache filename to avoid deep nested dirs
    safe_name = story_info.replace("/", "_").replace("info_", "") + ".txt"
    cache = STORY_DIR / safe_name

    if cache.exists():
        return cache.read_text(encoding="utf-8")

    rel_path = story_info_to_path(story_info)
    url = f"{BASE_URL}/{rel_path}"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        text = resp.text
        cache.write_text(text, encoding="utf-8")
        time.sleep(delay)
        return text
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None


class ArknightsStoryFetcher:
    def __init__(self):
        self.story_review: dict = {}
        self.char_names: set[str] = set()

    def load_metadata(self):
        self.story_review = fetch_json(TABLES["story_review"], "story_review")
        char_table = fetch_json(TABLES["character_table"], "character_table")
        self.char_names = {
            v["name"] for v in char_table.values()
            if v.get("name") and v.get("profession") not in ("TOKEN", "TRAP")
        }
        logger.info(
            f"Loaded: {len(self.story_review)} chapters, "
            f"{len(self.char_names)} known operator names"
        )

    def collect_story_infos(self) -> list[dict]:
        """Collect all storyInfo entries from story_review."""
        entries = []
        for chapter_id, chapter in self.story_review.items():
            chapter_name = chapter.get("name", chapter_id)
            act_type = chapter.get("actType", "")
            for node in chapter.get("infoUnlockDatas", []):
                story_info = node.get("storyInfo", "")
                if not story_info:
                    continue
                entries.append({
                    "chapter_id":   chapter_id,
                    "chapter_name": chapter_name,
                    "act_type":     act_type,
                    "story_id":     node.get("storyId", ""),
                    "story_name":   node.get("storyName", ""),
                    "story_info":   story_info,
                })
        logger.info(f"Collected {len(entries)} story nodes")
        return entries

    def fetch_all(self, limit: int = None, delay: float = 0.05) -> list[dict]:
        """
        Fetch all story scripts. Returns list of dicts with metadata + raw text.
        """
        entries = self.collect_story_infos()
        if limit:
            entries = entries[:limit]

        results = []
        failed = 0
        for entry in tqdm(entries, desc="Fetching story scripts"):
            text = fetch_story(entry["story_info"], delay=delay)
            if text is None:
                failed += 1
                continue
            results.append({**entry, "raw_text": text})

        logger.info(f"Fetched {len(results)} stories, {failed} failed")

        # Save manifest
        manifest = [{k: v for k, v in r.items() if k != "raw_text"} for r in results]
        manifest_path = RAW_DIR / "story_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved manifest -> {manifest_path}")

        return results


# Backwards-compatible wrapper
class PRTSScraper:
    def __init__(self, delay: float = 0.05):
        self.fetcher = ArknightsStoryFetcher()
        self.delay = delay

    def scrape_all(self, limit: int = None):
        self.fetcher.load_metadata()
        stories = self.fetcher.fetch_all(limit=limit, delay=self.delay)
        logger.info(f"Done. Total stories fetched: {len(stories)}")
        return stories


if __name__ == "__main__":
    scraper = PRTSScraper()
    scraper.scrape_all(limit=50)
