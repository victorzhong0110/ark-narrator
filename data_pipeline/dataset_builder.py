"""
Build three training dataset formats from Arknights story scripts.

Format A -- Narrative Continuation (叙事续写)
  Goal: Learn the world's narrative style and storytelling conventions.
  Format: {"text": "<chapter header>\n角色A：台词\n角色B：台词\n..."}
  mlx-lm: raw text, causal LM objective.

Format B -- Dialogue Window (对话窗口)
  Goal: Learn contextual next-line prediction with speaker tags.
  Format: {"messages": [system, user(context), assistant(next line)]}
  mlx-lm: chat format, applies template automatically.

Format C -- Roleplay (角色扮演)
  Goal: Learn to BE a specific character — respond as that character
        given what others have said, across multiple exchanges.
  Format: {"messages": [system(char card), user(others), assistant(char), ...]}
  mlx-lm: chat format, multi-turn.
"""

import json
import re
import random
import logging
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("./data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE    = 6     # turns per dialogue-window sample (B)
MIN_LINE_LEN   = 4     # minimum character length for a valid dialogue line
MIN_RP_TURNS   = 2     # minimum assistant turns for a valid roleplay sample (C)
# P6 fix: discard roleplay samples whose estimated token count exceeds the
# training max_seq_length.  Long sequences trigger NaN gradient explosions.
# Estimate: ~1.5 tokens per CJK character (conservative upper bound).
MAX_RP_TOKENS  = 3800  # leave headroom below 4096

TAG_RE  = re.compile(r"\[.*?\]")
NAME_RE = re.compile(r'^\[name="(.+?)"\]\s*(.*)')

NARRATIVE_SYSTEM = (
    "你是明日方舟（Arknights）的叙事生成助手，"
    "精通泰拉大陆的世界观与干员性格。"
    "请根据对话上下文，生成符合角色性格和当前情景的台词。"
)


# ---------------------------------------------------------------------------
# Script parser
# ---------------------------------------------------------------------------

@dataclass
class DialogueLine:
    speaker: str
    text: str


@dataclass
class StoryScript:
    chapter_name: str
    story_name: str
    act_type: str
    lines: list[DialogueLine] = field(default_factory=list)


def parse_script(raw_text: str, chapter_name: str, story_name: str, act_type: str) -> StoryScript:
    script = StoryScript(chapter_name=chapter_name, story_name=story_name, act_type=act_type)
    for raw_line in raw_text.splitlines():
        raw_line = raw_line.strip()
        m = NAME_RE.match(raw_line)
        if not m:
            continue
        speaker = m.group(1).strip()
        text = TAG_RE.sub("", m.group(2)).strip()
        if len(text) < MIN_LINE_LEN:
            continue
        if speaker in ("", "旁白"):
            continue
        script.lines.append(DialogueLine(speaker=speaker, text=text))
    return script


# ---------------------------------------------------------------------------
# Format A: Narrative Continuation
# ---------------------------------------------------------------------------

def build_narrative_samples(script: StoryScript, max_chars: int = 1500) -> list[dict]:
    if not script.lines:
        return []
    header = f"【{script.chapter_name}·{script.story_name}】\n"
    samples, buffer = [], header
    for line in script.lines:
        entry = f"{line.speaker}：{line.text}\n"
        if len(buffer) + len(entry) > max_chars and len(buffer) > len(header) + 50:
            samples.append({"text": buffer.strip(), "format": "narrative"})
            buffer = header
        buffer += entry
    if len(buffer) > len(header) + 50:
        samples.append({"text": buffer.strip(), "format": "narrative"})
    return samples


# ---------------------------------------------------------------------------
# Format B: Dialogue Window
# ---------------------------------------------------------------------------

def build_dialogue_window_samples(script: StoryScript, window_size: int = WINDOW_SIZE) -> list[dict]:
    lines = script.lines
    if len(lines) < window_size:
        return []
    scene_header = f"【场景：{script.chapter_name}·{script.story_name}】\n"
    samples = []
    for i in range(len(lines) - window_size + 1):
        window = lines[i: i + window_size]
        context_str = "\n".join(f"{l.speaker}：{l.text}" for l in window[:-1])
        target = window[-1]
        samples.append({
            "format": "dialogue_window",
            "messages": [
                {"role": "system",    "content": NARRATIVE_SYSTEM},
                {"role": "user",      "content": scene_header + context_str},
                {"role": "assistant", "content": f"{target.speaker}：{target.text}"},
            ]
        })
    return samples


# ---------------------------------------------------------------------------
# Format C: Roleplay
# ---------------------------------------------------------------------------

def _make_char_card(char_name: str, char_info: dict) -> str:
    """
    Build a system prompt (character card) for a given operator.
    Priority: handbook profile sections > character_table description > fallback.
    """
    parts = [f"你是明日方舟干员{char_name}。"]

    # 1. Use handbook profile sections (档案资料) if available
    handbook = char_info.get("_handbook", {})
    if handbook:
        # 档案一 is the most personality-revealing section
        for key in ("档案资料一", "档案一", "基础档案"):
            if key in handbook:
                excerpt = handbook[key][:300].strip()
                if excerpt:
                    parts.append(f"干员档案：{excerpt}")
                    break
        # Add speaking style hint from 综合体检测试 if present
        for key in ("综合体检测试", "晋升记录"):
            if key in handbook:
                excerpt = handbook[key][:150].strip()
                if excerpt:
                    parts.append(f"测试记录摘录：{excerpt}")
                    break

    # 2. Fall back to character_table description
    elif char_info.get("description"):
        parts.append(f"干员简介：{char_info['description']}")

    parts.append(
        f"请始终保持{char_name}的性格与说话方式进行对话，不要脱离角色，不要透露你是AI。"
    )
    return "\n".join(parts)


def build_roleplay_samples(
    script: StoryScript,
    known_chars: dict[str, dict],   # name -> char_info dict from character_table
    min_turns: int = MIN_RP_TURNS,
) -> list[dict]:
    """
    For each known operator appearing in the script, restructure the dialogue
    so that the operator's lines become assistant turns and everything else
    (possibly multiple speakers in a row) becomes user turns.

    Consecutive lines from other speakers are merged into one user message.
    Consecutive lines from the target character are merged into one assistant message.
    """
    if not script.lines:
        return []

    # Which known operators actually appear here?
    present = {l.speaker for l in script.lines} & set(known_chars.keys())
    if not present:
        return []

    scene_header = f"【{script.chapter_name}·{script.story_name}】\n"
    samples = []

    for char_name in present:
        char_info = known_chars[char_name]

        # P5 fix: skip operators with insufficient character card data.
        # Operators without handbook_info and only a one-line description produce
        # low-quality character cards that contaminate roleplay training signal.
        if "_handbook" not in char_info and len(char_info.get("description", "")) < 20:
            continue

        system_prompt = _make_char_card(char_name, char_info)
        messages = [{"role": "system", "content": system_prompt}]

        pending_others: list[DialogueLine] = []
        pending_self:   list[DialogueLine] = []
        assistant_turns = 0
        first_user = True

        def flush_others():
            nonlocal first_user
            if not pending_others:
                return
            ctx = "\n".join(f"{l.speaker}：{l.text}" for l in pending_others)
            if first_user:
                ctx = scene_header + ctx
                first_user = False
            messages.append({"role": "user", "content": ctx})
            pending_others.clear()

        def flush_self():
            nonlocal assistant_turns
            if not pending_self:
                return
            text = "\n".join(l.text for l in pending_self)
            messages.append({"role": "assistant", "content": text})
            assistant_turns += 1
            pending_self.clear()

        for line in script.lines:
            if line.speaker == char_name:
                flush_others()
                pending_self.append(line)
            else:
                flush_self()
                pending_others.append(line)

        # Final flush — only save if ends with assistant
        if pending_self:
            flush_others()
            flush_self()
        # else: discard trailing others

        # Validate: need enough exchanges and must start user→assistant
        if assistant_turns < min_turns:
            continue
        if len(messages) < 3:  # system + user + assistant minimum
            continue
        if messages[1]["role"] != "user":
            continue
        if messages[-1]["role"] != "assistant":
            continue

        # P6 fix: discard samples that are too long for max_seq_length.
        # CJK chars tokenise to ~1.5 tokens each (conservative upper bound).
        # Keeping these caused NaN gradient explosions during training.
        estimated_tokens = int(sum(len(m["content"]) for m in messages) * 1.5)
        if estimated_tokens > MAX_RP_TOKENS:
            continue

        samples.append({
            "format": "roleplay",
            "character": char_name,
            "messages": messages,
        })

    return samples


# ---------------------------------------------------------------------------
# MLX format converters
# ---------------------------------------------------------------------------

def to_mlx_narrative(s: dict) -> dict:
    return {"text": s["text"]}

def to_mlx_chat(s: dict) -> dict:
    return {"messages": s["messages"]}


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

class DatasetBuilder:
    def __init__(self, window_size: int = WINDOW_SIZE):
        self.window_size = window_size
        self.known_chars: dict[str, dict] = {}
        self.narrative_samples:       list[dict] = []
        self.dialogue_window_samples: list[dict] = []
        self.roleplay_samples:        list[dict] = []

    def load_char_table(
        self,
        char_table_path:    str = "./data/raw/character_table.json",
        handbook_path:      str = "./data/raw/handbook_info.json",
    ):
        """Load operator names + descriptions + handbook profiles for character cards."""
        with open(char_table_path, encoding="utf-8") as f:
            table = json.load(f)

        # Load handbook profile sections if available
        handbook_by_id: dict[str, dict] = {}
        if Path(handbook_path).exists():
            with open(handbook_path, encoding="utf-8") as f:
                hb_data = json.load(f)
            hb_dict = hb_data.get("handbookDict", hb_data)
            for char_id, hb in hb_dict.items():
                sections = {}
                for story in hb.get("storyTextAudio", []):
                    title = story.get("storyTitle", "")
                    stories = story.get("stories", [])
                    text = stories[0].get("storyText", "") if stories else ""
                    text = TAG_RE.sub("", text).strip()
                    if title and text:
                        sections[title] = text
                if sections:
                    handbook_by_id[char_id] = sections
            logger.info(f"Loaded handbook profiles for {len(handbook_by_id)} operators")
        else:
            logger.warning(f"handbook_info.json not found at {handbook_path}, using descriptions only")

        self.known_chars = {}
        for char_id, v in table.items():
            name = v.get("name", "")
            if not name:
                continue
            if v.get("profession") in ("TOKEN", "TRAP"):
                continue
            if v.get("isNotObtainable", False):
                continue
            entry = dict(v)
            if char_id in handbook_by_id:
                entry["_handbook"] = handbook_by_id[char_id]
            self.known_chars[name] = entry

        has_handbook = sum(1 for v in self.known_chars.values() if "_handbook" in v)
        logger.info(
            f"Loaded {len(self.known_chars)} operators "
            f"({has_handbook} with handbook profiles)"
        )
        logger.info(f"Loaded {len(self.known_chars)} known operator names")

    def add_story(self, raw_text: str, chapter_name: str, story_name: str, act_type: str):
        script = parse_script(raw_text, chapter_name, story_name, act_type)
        if not script.lines:
            return
        self.narrative_samples.extend(build_narrative_samples(script))
        self.dialogue_window_samples.extend(
            build_dialogue_window_samples(script, self.window_size)
        )
        if self.known_chars:
            self.roleplay_samples.extend(
                build_roleplay_samples(script, self.known_chars)
            )

    def build_from_fetched(self, stories: list[dict]):
        for story in stories:
            self.add_story(
                raw_text=story["raw_text"],
                chapter_name=story.get("chapter_name", ""),
                story_name=story.get("story_name", ""),
                act_type=story.get("act_type", ""),
            )
        self.stats()

    def stats(self):
        rp_chars = len({s["character"] for s in self.roleplay_samples})
        logger.info(
            f"Format A (narrative):       {len(self.narrative_samples):>6,} samples\n"
            f"Format B (dialogue_window): {len(self.dialogue_window_samples):>6,} samples\n"
            f"Format C (roleplay):        {len(self.roleplay_samples):>6,} samples "
            f"across {rp_chars} operators"
        )

    def save(self, split: float = 0.9):
        configs = [
            ("narrative",       self.narrative_samples,       to_mlx_narrative),
            ("dialogue_window", self.dialogue_window_samples, to_mlx_chat),
            ("roleplay",        self.roleplay_samples,        to_mlx_chat),
        ]
        for fmt_name, samples, converter in configs:
            if not samples:
                logger.warning(f"No samples for {fmt_name}, skipping.")
                continue
            random.shuffle(samples)
            idx = int(len(samples) * split)
            for split_name, data in [("train", samples[:idx]), ("eval", samples[idx:])]:
                path = PROCESSED_DIR / f"{fmt_name}_{split_name}.jsonl"
                with open(path, "w", encoding="utf-8") as f:
                    for s in data:
                        f.write(json.dumps(converter(s), ensure_ascii=False) + "\n")
                logger.info(f"Saved {len(data):>6,} {fmt_name} {split_name} -> {path}")

        # Save roleplay character distribution for inspection
        if self.roleplay_samples:
            char_counts = defaultdict(int)
            for s in self.roleplay_samples:
                char_counts[s["character"]] += 1
            dist_path = PROCESSED_DIR / "roleplay_char_distribution.json"
            with open(dist_path, "w", encoding="utf-8") as f:
                json.dump(
                    dict(sorted(char_counts.items(), key=lambda x: -x[1])),
                    f, ensure_ascii=False, indent=2
                )
            logger.info(f"Roleplay char distribution -> {dist_path}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_pipeline.scraper import PRTSScraper

    scraper = PRTSScraper()
    stories = scraper.scrape_all(limit=None)

    builder = DatasetBuilder(window_size=WINDOW_SIZE)
    builder.load_char_table()
    builder.build_from_fetched(stories)
    builder.save()
