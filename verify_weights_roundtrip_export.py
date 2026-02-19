# -*- coding: utf-8 -*-
"""
verify_weights_roundtrip_export.py

Round-trip check:
  original weights.hpp -> load into model -> export to roundtrip.hpp
  then compare token-by-token (numeric) for each array.

Usage (PowerShell):
  python verify_weights_roundtrip_export.py --weights "C:\path\to\weights.hpp"

Note:
  不再依賴 champ_weight_loader 的私有函式（_extract_tokens/_int_from_token），避免你改 loader 後爆掉。
"""

import argparse
import re
from pathlib import Path

import structure
import export
from champ_weight_loader import load_champ_weights_into_model


# 你要比對的 arrays（依你目前的 weights.hpp 命名）
ARRAYS = [
    # conv weights
    "conv_0_w_new",
    "conv_1_w_new",
    "conv_2_w_new",
    "conv_3_w_new",
    "conv_4_w_new",
    "conv_5_w_new",
    "conv_6_w_new",
    "conv_7_w_new",
    "conv_8_w_new",
    # bn fold inc/bias
    "conv_0_inc_new", "conv_0_bias_new",
    "conv_1_inc_new", "conv_1_bias_new",
    "conv_2_inc_new", "conv_2_bias_new",
    "conv_3_inc_new", "conv_3_bias_new",
    "conv_4_inc_new", "conv_4_bias_new",
    "conv_5_inc_new", "conv_5_bias_new",
    "conv_6_inc_new", "conv_6_bias_new",
    "conv_7_inc_new", "conv_7_bias_new",
    # head bias
    "conv_8_bias_new",
]

# 抓出 {} 內的所有數字 token：支援 0x.. 與 -0x..
NUM_RE = re.compile(r"-?0x[0-9a-fA-F]+")

def _extract_array_block(text: str, name: str) -> str:
    """
    抓出陣列 name 的 initializer 區塊（從 '=' 後的第一個 '{' 到對應的 '};'）
    這個寫法對格式/換行很寬鬆，避免被排版影響。
    """
    # 找到 name 出現的位置
    pos = text.find(name)
    if pos < 0:
        raise KeyError(f"cannot find array name: {name}")

    eq = text.find("=", pos)
    if eq < 0:
        raise KeyError(f"cannot find '=' after array name: {name}")

    lb = text.find("{", eq)
    if lb < 0:
        raise KeyError(f"cannot find '{{' after '=' for array: {name}")

    # 往後找對應的 '};'（粗略但對這種 header 通常夠用）
    end = text.find("};", lb)
    if end < 0:
        raise KeyError(f"cannot find '}};' end for array: {name}")

    return text[lb:end+1]

def _extract_nums(text: str, name: str):
    block = _extract_array_block(text, name)
    toks = NUM_RE.findall(block)
    return [int(t, 16) for t in toks]

def compare_one(name: str, a, b, show_first=5) -> int:
    if len(a) != len(b):
        print(f"[{name}] LEN mismatch: {len(a)} vs {len(b)}")
        return 999999

    mismatch = 0
    first = []
    for i, (x, y) in enumerate(zip(a, b)):
        if int(x) != int(y):
            mismatch += 1
            if len(first) < show_first:
                first.append((i, int(x), int(y)))

    print(f"[{name}] tokens={len(a)} mismatch={mismatch}")
    for i, x, y in first:
        print(f"  idx={i} orig={x} roundtrip={y}")
    return mismatch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="roundtrip_weights.hpp")
    args = ap.parse_args()

    wpath = Path(args.weights)
    if not wpath.exists():
        raise FileNotFoundError(wpath)

    # 1) build model (match champion config)
    model = structure.HLSLikeQATDetector(
        shifts=[19, 15, 15, 15, 15, 15, 15, 15],
        inc_bits_new=[15, 13, 12, 11, 12, 12, 12, 12],
        bias_bits_new=[25, 21, 21, 21, 21, 21, 21, 21],
        head_bias_bits_new=11,
    )

    # 2) load original weights.hpp into model
    load_champ_weights_into_model(model, wpath)

    # 3) export back using your export packer
    export.write_weights_hpp_from_model(model, args.out)
    print(f"[OK] wrote: {args.out}")

    # 4) token-wise numeric compare per array
    t_orig = wpath.read_text(encoding="utf-8", errors="ignore")
    t_new = Path(args.out).read_text(encoding="utf-8", errors="ignore")

    total = 0
    for name in ARRAYS:
        a = _extract_nums(t_orig, name)
        b = _extract_nums(t_new, name)
        total += compare_one(name, a, b)

    print("\n[SUMMARY] total_mismatch =", total)
    if total == 0:
        print("=> ✅ weight 反解/再打包完全一致（loader + export packing 都對）")
    else:
        print("=> ❌ mismatch > 0：優先檢查 nibble packing / kr,kc 順序 / PE blocking / inc,bias 取值")


if __name__ == "__main__":
    main()
