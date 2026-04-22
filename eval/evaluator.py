"""
Main evaluation pipeline.
Compares: base model vs. fine-tuned model vs. GPT-4o baseline.
"""

import json
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from eval.metrics import LoreChecker, GPTJudge, EvalResult

logger = logging.getLogger(__name__)
RESULTS_DIR = Path("./eval/results")
RESULTS_DIR.mkdir(exist_ok=True)


class Evaluator:
    def __init__(self, use_gpt_judge: bool = True):
        self.lore_checker = LoreChecker()
        self.gpt_judge = GPTJudge() if use_gpt_judge else None

    def evaluate_outputs(
        self,
        samples: list[dict],
        model_name: str,
    ) -> list[EvalResult]:
        results = []
        for i, s in enumerate(tqdm(samples, desc=f"Evaluating [{model_name}]")):
            lore_score = self.lore_checker.score(s["output"])

            if self.gpt_judge:
                scores = self.gpt_judge.judge(
                    s["instruction"], s["output"], s.get("task_type", "unknown")
                )
                result = EvalResult(
                    sample_id=str(i),
                    task_type=s.get("task_type", "unknown"),
                    lore_score=scores.get("lore", lore_score),
                    consistency_score=scores.get("consistency", 0),
                    fluency_score=scores.get("fluency", 0),
                    overall=round(
                        (scores.get("lore", 0) + scores.get("consistency", 0) + scores.get("fluency", 0)) / 3, 2
                    ),
                    judge_reasoning=scores.get("reasoning", ""),
                )
            else:
                result = EvalResult(
                    sample_id=str(i),
                    task_type=s.get("task_type", "unknown"),
                    lore_score=lore_score,
                    consistency_score=0,
                    fluency_score=0,
                    overall=lore_score,
                    judge_reasoning="rule-based only",
                )
            results.append(result)

        # Save results
        df = pd.DataFrame([r.__dict__ for r in results])
        out = RESULTS_DIR / f"{model_name}_eval.csv"
        df.to_csv(out, index=False)
        logger.info(f"Results saved → {out}")
        logger.info(f"[{model_name}] Average overall: {df['overall'].mean():.2f}")
        return results

    @staticmethod
    def compare_models(model_results: dict[str, list[EvalResult]]) -> pd.DataFrame:
        """Generate comparison table across models."""
        rows = []
        for model_name, results in model_results.items():
            df = pd.DataFrame([r.__dict__ for r in results])
            rows.append({
                "model": model_name,
                "lore_score": df["lore_score"].mean(),
                "consistency_score": df["consistency_score"].mean(),
                "fluency_score": df["fluency_score"].mean(),
                "overall": df["overall"].mean(),
            })
        comparison = pd.DataFrame(rows).round(2)
        comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
        print("\n=== Model Comparison ===")
        print(comparison.to_string(index=False))
        return comparison
