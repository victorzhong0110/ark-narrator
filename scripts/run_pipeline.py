"""
One-command pipeline runner.
Usage:
  python scripts/run_pipeline.py --mode scrape     # Collect raw data
  python scripts/run_pipeline.py --mode build      # Build dataset
  python scripts/run_pipeline.py --mode train      # Fine-tune model
  python scripts/run_pipeline.py --mode eval       # Run evaluation
  python scripts/run_pipeline.py --mode serve      # Start API server
  python scripts/run_pipeline.py --mode all        # Run full pipeline
"""

import argparse
import logging
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_scrape(args):
    logger.info("=== Phase 1: Scraping Arknights data ===")
    from data_pipeline.scraper import PRTSScraper
    scraper = PRTSScraper()
    scraper.scrape_all(limit=args.limit)


def run_build(args):
    logger.info("=== Phase 2: Building dataset ===")
    import json
    from pathlib import Path
    from data_pipeline.dataset_builder import DatasetBuilder

    with open("./data/raw/operator_profiles.json", encoding="utf-8") as f:
        profiles = json.load(f)
    builder = DatasetBuilder()
    builder.build_from_profiles(profiles)
    builder.to_jsonl()
    logger.info(f"Dataset ready: {len(builder.samples)} samples")


def run_train(args):
    logger.info("=== Phase 3: Fine-tuning ===")
    from finetune.train import main as train_main
    train_main(args.config)


def run_eval(args):
    logger.info("=== Phase 4: Evaluation ===")
    # Placeholder — see eval/evaluator.py for full implementation
    logger.info("Run eval/evaluator.py with your generated outputs.")


def run_serve(args):
    logger.info("=== Starting inference server ===")
    subprocess.run([sys.executable, "-m", "uvicorn", "inference.server:app",
                    "--host", "0.0.0.0", "--port", "8000", "--reload"])


MODES = {
    "scrape": run_scrape,
    "build": run_build,
    "train": run_train,
    "eval": run_eval,
    "serve": run_serve,
}


def main():
    parser = argparse.ArgumentParser(description="ArkNarrator pipeline runner")
    parser.add_argument("--mode", choices=[*MODES.keys(), "all"], default="all")
    parser.add_argument("--config", default="./finetune/config/qwen2_5_lora.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Operator scrape limit")
    args = parser.parse_args()

    if args.mode == "all":
        for mode_fn in MODES.values():
            mode_fn(args)
    else:
        MODES[args.mode](args)


if __name__ == "__main__":
    main()
