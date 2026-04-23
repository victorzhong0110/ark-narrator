"""
Build instruction-tuning dataset from raw Arknights lore data.
Output format: JSONL with {"instruction": ..., "input": ..., "output": ...}
"""

import json
import random
import logging
from pathlib import Path
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DIR = Path("./data/raw")
PROCESSED_DIR = Path("./data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = (
    "你是明日方舟（Arknights）世界中的内容创作助手，"
    "精通泰拉大陆的历史、干员档案与源石技艺体系。"
    "请根据设定生成符合世界观的内容。"
)

# Instruction templates for different task types
TEMPLATES = {
    "profile_qa": [
        "请介绍干员{name}的背景故事。",
        "干员{name}是谁？请描述她/他的过去。",
        "根据档案，{name}加入罗德岛的原因是什么？",
    ],
    "dialogue": [
        "以干员{name}的口吻，写一段与博士初次见面的对话。",
        "干员{name}在执行任务前会说什么？请写3句符合她/他性格的台词。",
        "写一段{name}与{name2}之间关于源石病的对话。",
    ],
    "worldbuilding": [
        "请介绍泰拉大陆中{faction}的历史与现状。",
        "源石技艺是什么？请从学术角度简要解释。",
        "移动城市制度是如何形成的？",
    ],
}


@dataclass
class DataSample:
    instruction: str
    input: str
    output: str
    task_type: str
    source: str


def clean_output(text: str) -> str:
    """
    Remove noise from raw game data text:
    - 【来源】 citation blocks with broken links
    - [链接已失效] markers
    - Excessive whitespace
    """
    import re
    # Remove 【来源】 block: everything from 【来源】 to the first blank line
    text = re.sub(r'【来源】.*?(?=\n\n|\Z)', '', text, flags=re.DOTALL)
    # Remove inline broken-link markers
    text = re.sub(r'\[链接已失效\]', '', text)
    text = re.sub(r'\[.*?链接.*?\]', '', text)
    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


class DatasetBuilder:
    def __init__(self):
        self.samples: list[DataSample] = []

    def build_from_profiles(self, profiles: list[dict]) -> None:
        """Convert operator profiles into instruction-tuning samples."""
        for op in profiles:
            name = op.get("name", "")
            sections = op.get("sections", {})

            # Profile Q&A pairs
            for section_name, content in sections.items():
                if not content or len(content) < 50:
                    continue
                if "档案" in section_name:
                    cleaned = clean_output(content)
                    if len(cleaned) < 30:   # skip if cleaning left nothing
                        continue
                    instr = random.choice(TEMPLATES["profile_qa"]).format(name=name)
                    self.samples.append(
                        DataSample(
                            instruction=instr,
                            input="",
                            output=cleaned[:800],
                            task_type="profile_qa",
                            source=f"ark/{name}",
                        )
                    )

            # Dialogue generation (uses profile as context)
            if sections:
                context = "\n".join(list(sections.values())[:2])[:400]
                instr = random.choice(TEMPLATES["dialogue"]).format(
                    name=name, name2="博士"
                )
                self.samples.append(
                    DataSample(
                        instruction=instr,
                        input=f"参考档案：{context}",
                        output="",  # placeholder — fill with GPT augmentation
                        task_type="dialogue",
                        source=f"prts/{name}",
                    )
                )

    def to_jsonl(self, split: float = 0.9) -> None:
        """Split into train/eval and save as JSONL."""
        valid = [s for s in self.samples if s.output]
        random.shuffle(valid)
        split_idx = int(len(valid) * split)
        train, eval_ = valid[:split_idx], valid[split_idx:]

        for name, data in [("train", train), ("eval", eval_)]:
            path = PROCESSED_DIR / f"{name}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for s in data:
                    record = {
                        "instruction": s.instruction,
                        "input": s.input,
                        "output": s.output,
                        "task_type": s.task_type,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info(f"Saved {len(data)} samples → {path}")


if __name__ == "__main__":
    with open(RAW_DIR / "operator_profiles.json", encoding="utf-8") as f:
        profiles = json.load(f)

    builder = DatasetBuilder()
    builder.build_from_profiles(profiles)
    builder.to_jsonl()
    logger.info(f"Total samples: {len(builder.samples)}")
