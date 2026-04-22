"""
Arknights lore data fetcher.
Source: Kengxxiao/ArknightsGameData (official game data, public GitHub repo)
  - character_table.json  → operator basic info
  - handbook_info_table.json → operator profiles / lore (档案资料)
  - story_review_table.json → main story chapter list
  - activity_table.json → event info

No scraping needed — raw GitHub CDN, no anti-bot issues.
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
RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = (
    "https://raw.githubusercontent.com/Kengxxiao/ArknightsGameData"
    "/master/zh_CN/gamedata/excel"
)

TABLES = {
    "character_table":   f"{BASE_URL}/character_table.json",
    "handbook_info":     f"{BASE_URL}/handbook_info_table.json",
    "story_review":      f"{BASE_URL}/story_review_table.json",
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
    logger.info(f"Saved → {cache}")
    return data


class ArknightsDataFetcher:
    def __init__(self):
        self.chars = {}
        self.handbook = {}
        self.stories = {}

    def load(self):
        self.chars    = fetch_json(TABLES["character_table"], "character_table")
        self.handbook = fetch_json(TABLES["handbook_info"],   "handbook_info")
        self.stories  = fetch_json(TABLES["story_review"],    "story_review")
        logger.info(
            f"Loaded: {len(self.chars)} operators, "
            f"{len(self.handbook.get('handbookDict', {}))} profiles"
        )

    def build_operator_profiles(self, limit: int = None) -> list[dict]:
        """
        Merge character_table + handbook_info into clean operator profiles.
        Returns list of dicts ready for dataset_builder.
        """
        handbook_dict = self.handbook.get("handbookDict", {})
        profiles = []

        char_items = list(self.chars.items())
        if limit:
            char_items = char_items[:limit]

        for char_id, char in tqdm(char_items, desc="Building profiles"):
            # Skip non-operator entries (enemy units, traps, etc.)
            if not char.get("isNotObtainable") == False and char.get("profession") == "TRAP":
                continue
            if char.get("profession") in ("TOKEN", "TRAP"):
                continue
            name = char.get("name", "")
            if not name:
                continue

            # Get handbook (lore) data
            hb = handbook_dict.get(char_id, {})
            story_text_list = hb.get("storyTextAudio", [])

            sections = {}
            for story in story_text_list:
                title = story.get("storyTitle", "")
                text  = story.get("stories", [{}])[0].get("storyText", "") if story.get("stories") else ""
                if title and text:
                    sections[title] = text.strip()

            if not sections:
                continue  # skip operators with no lore yet

            profiles.append({
                "char_id":    char_id,
                "name":       name,
                "rarity":     char.get("rarity", ""),
                "profession":  char.get("profession", ""),
                "description": char.get("description", ""),
                "tagList":    char.get("tagList", []),
                "sections":   sections,
            })

        logger.info(f"Built {len(profiles)} operator profiles with lore")

        out = RAW_DIR / "operator_profiles.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(profiles, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved → {out}")
        return profiles


# ── Keep original interface for run_pipeline.py ───────────────────────────────

class PRTSScraper:
    """Backwards-compatible wrapper — now uses ArknightsGameData instead of PRTS."""

    def __init__(self, delay: float = 0):
        self.fetcher = ArknightsDataFetcher()

    def scrape_all(self, limit: int = None):
        self.fetcher.load()
        profiles = self.fetcher.build_operator_profiles(limit=limit)
        logger.info(f"Done. Total profiles: {len(profiles)}")


if __name__ == "__main__":
    scraper = PRTSScraper()
    scraper.scrape_all(limit=20)
