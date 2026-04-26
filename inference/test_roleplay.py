"""
Quick roleplay inference test — applies chat template with a character card
system prompt, which is the format the model was trained on.

Usage:
  python inference/test_roleplay.py
  python inference/test_roleplay.py --char 凯尔希
  python inference/test_roleplay.py --char 能天使 --turns 3
"""

import argparse
from pathlib import Path
from mlx_lm import load, generate

# mlx-lm API varies by version:
# older: generate(..., temp=0.8)
# middle: generate(..., temperature=0.8)
# newer: generate(..., sampler=make_sampler(temp=0.8))
# Detect and wrap so the script works across versions.
def _make_generate_kwargs(temperature: float = 0.8) -> dict:
    import inspect
    sig = inspect.signature(generate)
    if "temp" in sig.parameters:
        return {"temp": temperature}
    if "temperature" in sig.parameters:
        return {"temperature": temperature}
    # Newer versions use sampler
    try:
        from mlx_lm.sample_utils import make_sampler
        return {"sampler": make_sampler(temp=temperature)}
    except Exception:
        pass
    return {}  # fall back to greedy if nothing works

# Minimal hand-crafted character cards for quick testing.
# In production these come from _make_char_card() in dataset_builder.py.
CHARACTER_CARDS = {
    "能天使": (
        "你是明日方舟干员能天使。\n"
        "干员档案：能天使是莱茵生命旗下的实验品干员，后加入罗德岛。"
        "性格活泼直接，说话随意不绕弯子，有时大大咧咧，偶尔会提到想吃东西。"
        "她重视伙伴，行动力强，不喜欢拖泥带水。\n"
        "请始终保持能天使的性格与说话方式进行对话，不要脱离角色，不要透露你是AI。"
    ),
    "凯尔希": (
        "你是明日方舟干员凯尔希。\n"
        "干员档案：凯尔希是罗德岛首席医疗官，话语简洁克制，判断冷静精准，"
        "不轻易表露情绪，但对干员的健康和任务成败有强烈责任感。"
        "她措辞严谨，习惯用长句子陈述事实，极少使用感叹词。\n"
        "请始终保持凯尔希的性格与说话方式进行对话，不要脱离角色，不要透露你是AI。"
    ),
    "阿米娅": (
        "你是明日方舟干员阿米娅。\n"
        "干员档案：阿米娅是罗德岛的领袖，年轻但意志坚定。说话温柔而有力，"
        "对博士和干员充满信任，面对困难时会感到迷茫但始终不放弃。"
        "语气中带着成长中的少女的真诚与认真。\n"
        "请始终保持阿米娅的性格与说话方式进行对话，不要脱离角色，不要透露你是AI。"
    ),
}

DEFAULT_CHAR      = "能天使"
QWEN_ADAPTER_DIR  = "checkpoints/qwen2_5_mlx_roleplay/"
QWEN_MODEL_PATH   = "mlx-community/Qwen2.5-7B-Instruct-4bit"
GEMMA_ADAPTER_DIR = "checkpoints/gemma4_mlx_roleplay/"
GEMMA_MODEL_PATH  = "mlx-community/gemma-4-E4B-it-4bit"


def run_single_turn(model, tokenizer, char_name: str, user_input: str) -> str:
    system = CHARACTER_CARDS[char_name]
    messages = [
        {"role": "system",    "content": system},
        {"role": "user",      "content": user_input},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return generate(model, tokenizer, prompt=prompt, max_tokens=300,
                    **_make_generate_kwargs(0.8), verbose=False)


def run_multi_turn(model, tokenizer, char_name: str, turns: int):
    system  = CHARACTER_CARDS[char_name]
    history = [{"role": "system", "content": system}]

    print(f"\n[Multi-turn roleplay — {char_name}]  (输入 'quit' 退出)\n")
    for _ in range(turns):
        user_input = input("你：").strip()
        if user_input.lower() in ("quit", "exit", "退出"):
            break

        history.append({"role": "user", "content": user_input})
        prompt = tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True
        )
        response = generate(model, tokenizer, prompt=prompt, max_tokens=300,
                            **_make_generate_kwargs(0.8), verbose=False)
        history.append({"role": "assistant", "content": response})
        print(f"\n{char_name}：{response}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--char",  default=DEFAULT_CHAR, choices=list(CHARACTER_CARDS))
    parser.add_argument("--turns", type=int, default=1,
                        help="Number of turns (>1 enters interactive mode)")
    parser.add_argument("--prompt", default="你好，最近任务怎么样？",
                        help="Single-turn user prompt")
    parser.add_argument(
        "--model", default="qwen", choices=["qwen", "gemma"],
        help="Which base model to load (qwen=Qwen2.5-7B, gemma=Gemma4-27B)",
    )
    parser.add_argument("--adapter-dir", default=None,
                        help="Override adapter directory (default: per-model default)")
    parser.add_argument("--model-path", default=None,
                        help="Override HuggingFace / mlx-community model path")
    args = parser.parse_args()

    if args.model == "gemma":
        adapter_dir = args.adapter_dir or GEMMA_ADAPTER_DIR
        model_path  = args.model_path  or GEMMA_MODEL_PATH
    else:
        adapter_dir = args.adapter_dir or QWEN_ADAPTER_DIR
        model_path  = args.model_path  or QWEN_MODEL_PATH

    if args.char not in CHARACTER_CARDS:
        print(f"未知角色 {args.char!r}，可选：{list(CHARACTER_CARDS)}")
        return

    print(f"Loading {args.model} model + adapter from {adapter_dir} ...")
    model, tokenizer = load(model_path, adapter_path=adapter_dir)
    print("Ready.\n")

    if args.turns > 1:
        run_multi_turn(model, tokenizer, args.char, args.turns)
    else:
        print(f"[Single turn — {args.char}]")
        print(f"用户：{args.prompt}")
        response = run_single_turn(model, tokenizer, args.char, args.prompt)
        print(f"{args.char}：{response}")


if __name__ == "__main__":
    main()
