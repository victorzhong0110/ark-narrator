"""
Patch mlx-lm 0.31.3 to support Gemma 4 E4B architecture.

Problem
-------
mlx-lm misinterprets `num_kv_shared_layers=18` in the E4B config as
"layers 24–41 have no k_proj / v_proj / k_norm — reuse KV vectors from
earlier layers." The actual E4B checkpoint has independent projections for
all 42 layers (126 missing params without this patch).

Fix
---
Two changes in gemma4_text.py:
  1. Attention.__init__: has_kv = True for all layers
  2. Gemma4TextModel.__init__: previous_kvs[i] = i for all i (no cross-layer reuse)

Tested on: mlx-lm 0.31.3, mlx-community/gemma-4-E4B-it-4bit
"""

import sys
import re
from pathlib import Path


def find_gemma4_text() -> Path:
    try:
        import mlx_lm.models.gemma4_text as m
        return Path(m.__file__)
    except ImportError:
        sys.exit("mlx-lm not found. Install with: pip install mlx-lm==0.31.3")


def patch(path: Path) -> None:
    src = path.read_text(encoding="utf-8")

    # --- patch 1: has_kv ---
    old1 = "self.has_kv = layer_idx < config.num_hidden_layers - config.num_kv_shared_layers"
    new1 = (
        "# All layers have independent KV projections in the E4B checkpoint;\n"
        "        # num_kv_shared_layers means weight-tying in HF, not KV-vector reuse.\n"
        "        self.has_kv = True"
    )
    if old1 not in src:
        if "self.has_kv = True" in src:
            print("✓ Patch 1 already applied (has_kv = True).")
        else:
            print("⚠ Patch 1: target line not found — mlx-lm version may differ.")
        patched1 = src
    else:
        patched1 = src.replace(old1, new1, 1)
        print("✓ Patch 1 applied: has_kv = True for all layers.")

    # --- patch 2: previous_kvs ---
    old2 = (
        "        # Arrange for shared KVs\n"
        "        self.previous_kvs = list(range(len(self.layers)))\n"
        "        if config.num_kv_shared_layers > 0:\n"
        "            N = len(self.layers)\n"
        "            M = N - config.num_kv_shared_layers\n"
        "            kvs_by_type = {}\n"
        "            for i in range(M):\n"
        "                kvs_by_type[self.layers[i].layer_type] = i\n"
        "            for j in range(M, N):\n"
        "                self.previous_kvs[j] = kvs_by_type[self.layers[j].layer_type]"
    )
    new2 = (
        "        # Every layer computes its own K/V (no cross-layer reuse).\n"
        "        self.previous_kvs = list(range(len(self.layers)))"
    )
    if old2 not in patched1:
        if "Every layer computes its own K/V" in patched1:
            print("✓ Patch 2 already applied (previous_kvs no sharing).")
        else:
            print("⚠ Patch 2: target block not found — mlx-lm version may differ.")
    else:
        patched1 = patched1.replace(old2, new2, 1)
        print("✓ Patch 2 applied: previous_kvs[i] = i for all i.")

    path.write_text(patched1, encoding="utf-8")
    print(f"\nPatched: {path}")
    print('Verify: python -c "from mlx_lm import load; m,t = load(\'mlx-community/gemma-4-E4B-it-4bit\'); print(\'OK\')"')


if __name__ == "__main__":
    patch(find_gemma4_text())
