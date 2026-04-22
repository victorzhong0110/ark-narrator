"""
Arknights lore data scraper.
Sources: PRTS Wiki (prts.wiki) — operator profiles, story texts, world lore.
"""

import requests
import time
import json
import logging
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PRTS_BASE = "https://prts.wiki"
HEADERS = {"User-Agent": "ArkNarrator-Research/1.0 (academic use)"}
RAW_DIR = Path("./data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


class PRTSScraper:
    """Scrape operator profiles and story content from PRTS wiki."""

    def __init__(self, delay: float = 1.5):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.delay = delay  # polite crawl delay

    def get_operator_list(self) -> list[dict]:
        """Fetch list of all operators with basic info."""
        url = f"{PRTS_BASE}/w/干员一览"
        resp = self.session.get(url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        operators = []
        for row in soup.select("table.wikitable tr")[1:]:
            cols = row.find_all("td")
            if len(cols) < 3:
                continue
            name = cols[0].get_text(strip=True)
            rarity = cols[1].get_text(strip=True)
            profession = cols[2].get_text(strip=True)
            if name:
                operators.append({"name": name, "rarity": rarity, "profession": profession})

        logger.info(f"Found {len(operators)} operators")
        return operators

    def get_operator_profile(self, name: str) -> dict:
        """Fetch full operator profile: lore, story, voice lines."""
        url = f"{PRTS_BASE}/w/{name}"
        resp = self.session.get(url, timeout=15)
        if resp.status_code != 200:
            return {}

        soup = BeautifulSoup(resp.text, "html.parser")
        profile = {"name": name, "sections": {}}

        # Extract profile sections (档案资料, 模组档案, 语音台词, etc.)
        for section in soup.select(".mw-headline"):
            section_title = section.get_text(strip=True)
            content_div = section.find_parent().find_next_sibling()
            if content_div:
                profile["sections"][section_title] = content_div.get_text(
                    separator="\n", strip=True
                )

        time.sleep(self.delay)
        return profile

    def scrape_all(self, limit: int = None) -> None:
        """Scrape all operators and save raw data."""
        operators = self.get_operator_list()
        if limit:
            operators = operators[:limit]

        results = []
        for op in tqdm(operators, desc="Scraping operators"):
            profile = self.get_operator_profile(op["name"])
            if profile:
                profile.update(op)
                results.append(profile)

        out_path = RAW_DIR / "operator_profiles.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(results)} profiles → {out_path}")


if __name__ == "__main__":
    scraper = PRTSScraper()
    scraper.scrape_all(limit=50)  # start with 50 for testing
