# -*- coding: utf-8 -*-
"""
champ_weight_loader.py (FIXED)

重點修正：
- conv1~conv7 的 4-bit weight 在 HLS conv2d.hpp 裡是以 ap_uint 方式使用，並用 subdata 做 offset 校正
  => 等價於 w_eff = (nibble - 8)  (offset-binary)
- conv0 / conv8 (head) 仍用 two's complement 解 nibble 較合理
"""

from __future__ import annotations
import re
import numpy as np
import torch


# ---------- 4-bit decode helpers ----------
def _u4(n: int) -> int:
    return int(n) & 0xF

def _s4_twos(n: int) -> int:
    """two's complement: 0..7 => 0..7, 8..15 => -8..-1"""
    n = _u4(n)
    return n - 16 if n >= 8 else n

def _s4_offset(n: int) -> int:
    """offset-binary: 0..15 => -8..7"""
    return _u4(n) - 8

def _unpack_int4_word(word: int, simd: int, mode: str) -> np.ndarray:
    """
    word: packed as little-endian nibbles (lowest nibble is i=0)
    mode:
      - "twos"   : two's complement nibble -> [-8,7]
      - "offset" : offset-binary nibble -> [-8,7] via (u4-8)
    """
    out = np.empty((simd,), dtype=np.int8)
    for i in range(simd):
        nib = (int(word) >> (4 * i)) & 0xF
        if mode == "twos":
            out[i] = _s4_twos(nib)
        elif mode == "offset":
            out[i] = _s4_offset(nib)
        else:
            raise ValueError(f"unknown mode={mode}")
    return out


# ---------- parse weights.hpp ----------
_re_hex = re.compile(r"0x[0-9a-fA-F]+")

def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _extract_array_block(text: str, name: str) -> str:
    """
    抓出 `name = { ... };` 之間的大括號內容（純文字），讓我們把 hex 全抓出來。
    """
    # 很寬鬆：從 name 開始抓到第一個 ';'
    m = re.search(rf"{re.escape(name)}\s*=\s*\{{", text)
    if not m:
        raise KeyError(f"cannot find array: {name}")
    start = m.end()
    end = text.find(";", start)
    if end < 0:
        raise KeyError(f"cannot find ';' for array: {name}")
    return text[start:end]


def _load_conv0_w(text: str) -> torch.Tensor:
    """
    conv_0_w_new[16][3][3] each is ap_uint<12> packing 3*nibble (RGB channel weights)
    HLS conv0 用 ap_int<4> reinterpret bits => two's complement
    return: (16,3,3,3) int8
    """
    block = _extract_array_block(text, "const ap_uint<12>conv_0_w_new[16][3][3]")
    hexes = _re_hex.findall(block)
    if len(hexes) != 16 * 3 * 3:
        raise ValueError(f"conv0 hex count mismatch: {len(hexes)}")
    w = np.zeros((16, 3, 3, 3), dtype=np.int8)
    k = 0
    for oc in range(16):
        for kr in range(3):
            for kc in range(3):
                word = int(hexes[k], 16); k += 1
                # 12-bit: [ch0 nib][ch1 nib][ch2 nib]
                n0 = (word >> 0) & 0xF
                n1 = (word >> 4) & 0xF
                n2 = (word >> 8) & 0xF
                w[oc, 0, kr, kc] = _s4_twos(n0)
                w[oc, 1, kr, kc] = _s4_twos(n1)
                w[oc, 2, kr, kc] = _s4_twos(n2)
    return torch.from_numpy(w)


def _load_conv3x3_w(text: str, layer: int, pe: int, in_ch: int, out_ch: int, simd: int, mode: str) -> torch.Tensor:
    """
    conv_{layer}_w_new[pe][3][K]  where K = infold*(out_ch/pe)
    這裡照你的 export.py / HLS config 的 indexing 去還原成 (out_ch,in_ch,3,3)

    mode:
      - conv1~7: "offset"
      - (如果你有別的設計可改)
    """
    name = f"const ap_uint<{simd*4}>conv_{layer}_w_new"
    # weights.hpp 裡實際 word width 是 64/128/256，但 simd*4 也會對上
    # 用比較保守方式：直接搜 conv_{layer}_w_new[...] 的宣告行
    m = re.search(rf"const ap_uint<\d+>conv_{layer}_w_new\[{pe}\]\[3\]\[\d+\]\s*=\s*\{{", text)
    if not m:
        raise KeyError(f"cannot find conv_{layer}_w_new declaration")
    decl = text[m.start():m.end()]
    # 抓整段 array block
    block = text[m.end(): text.find(";", m.end())]
    hexes = _re_hex.findall(block)

    simdnum = in_ch // simd
    infold = 3 * simdnum
    cols = out_ch // pe
    K = infold * cols
    expect = pe * 3 * K
    if len(hexes) != expect:
        raise ValueError(f"conv{layer} hex count mismatch: got {len(hexes)} expect {expect}")

    w = np.zeros((out_ch, in_ch, 3, 3), dtype=np.int8)

    idx = 0
    for p in range(pe):
        for kc in range(3):            # 注意：HLS/export 用的是 [p][kc][idx]，kc 在第二維
            for flat in range(K):
                word = int(hexes[idx], 16); idx += 1
                vals = _unpack_int4_word(word, simd, mode=mode)  # (simd,)
                po = flat // infold
                t = flat % infold
                kr = t // simdnum
                s  = t % simdnum
                oc = po * pe + p
                ic0 = s * simd
                ic1 = ic0 + simd
                w[oc, ic0:ic1, kr, kc] = vals

    return torch.from_numpy(w)


def _load_head_w(text: str, pe: int = 2, in_ch: int = 64, out_ch: int = 72, simd: int = 8) -> torch.Tensor:
    """
    conv_8_w_new[2][288], each ap_uint<32> packing 8*nibble
    HLS head 用 ap_int<4> reinterpret bits => two's complement
    return: (72,64,1,1)
    """
    m = re.search(r"const ap_uint<32>conv_8_w_new\[2\]\[288\]\s*=\s*\{", text)
    if not m:
        raise KeyError("cannot find conv_8_w_new")
    block = text[m.end(): text.find(";", m.end())]
    hexes = _re_hex.findall(block)
    if len(hexes) != 2 * 288:
        raise ValueError(f"conv8 hex count mismatch: {len(hexes)}")

    simdnum = in_ch // simd  # 8
    K = simdnum * (out_ch // pe)  # 8 * 36 = 288

    w = np.zeros((out_ch, in_ch, 1, 1), dtype=np.int8)
    idx = 0
    for p in range(pe):
        for flat in range(K):
            word = int(hexes[idx], 16); idx += 1
            vals = _unpack_int4_word(word, simd, mode="twos")
            po = flat // simdnum
            s  = flat % simdnum
            oc = po * pe + p
            ic0 = s * simd
            ic1 = ic0 + simd
            w[oc, ic0:ic1, 0, 0] = vals
    return torch.from_numpy(w)


def _load_vec_int(text: str, name_pat: str, out_ch: int, pe: int) -> torch.Tensor:
    """
    讀 inc/bias 這類 ap_int<...> 二維 [pe][out_ch/pe]
    """
    m = re.search(rf"{name_pat}\s*=\s*\{{", text)
    if not m:
        raise KeyError(f"cannot find: {name_pat}")
    block = text[m.end(): text.find(";", m.end())]
    # 支援 "0x.." / "-0x.."
    nums = re.findall(r"-?0x[0-9a-fA-F]+", block)
    if len(nums) != out_ch:
        raise ValueError(f"{name_pat} count mismatch: {len(nums)} != {out_ch}")
    v = np.zeros((out_ch,), dtype=np.int32)
    for i,s in enumerate(nums):
        v[i] = int(s, 16)
    # reshape to [pe][out_ch/pe] then map back to [out_ch]
    v2 = np.zeros((out_ch,), dtype=np.int32)
    cols = out_ch // pe
    k = 0
    for p in range(pe):
        for po in range(cols):
            oc = po * pe + p
            v2[oc] = v[k]; k += 1
    return torch.from_numpy(v2)


def load_champ_weights_into_model(model, weights_hpp_path: str):
    text = _read_text(weights_hpp_path)

    # ---- conv weights ----
    model.conv0.conv.weight.data.copy_(_load_conv0_w(text).to(model.conv0.conv.weight.data.dtype))

    # conv1~7: OFFSET-BINARY !!
    model.conv1.conv.weight.data.copy_(_load_conv3x3_w(text, 1, pe=8,  in_ch=16, out_ch=32, simd=16, mode="offset").to(model.conv1.conv.weight.data.dtype))
    model.conv2.conv.weight.data.copy_(_load_conv3x3_w(text, 2, pe=4,  in_ch=32, out_ch=64, simd=32, mode="offset").to(model.conv2.conv.weight.data.dtype))
    model.conv3.conv.weight.data.copy_(_load_conv3x3_w(text, 3, pe=1,  in_ch=64, out_ch=64, simd=64, mode="offset").to(model.conv3.conv.weight.data.dtype))
    model.conv4.conv.weight.data.copy_(_load_conv3x3_w(text, 4, pe=1,  in_ch=64, out_ch=64, simd=16, mode="offset").to(model.conv4.conv.weight.data.dtype))
    model.conv5.conv.weight.data.copy_(_load_conv3x3_w(text, 5, pe=1,  in_ch=64, out_ch=64, simd=16, mode="offset").to(model.conv5.conv.weight.data.dtype))
    model.conv6.conv.weight.data.copy_(_load_conv3x3_w(text, 6, pe=1,  in_ch=64, out_ch=64, simd=16, mode="offset").to(model.conv6.conv.weight.data.dtype))
    model.conv7.conv.weight.data.copy_(_load_conv3x3_w(text, 7, pe=1,  in_ch=64, out_ch=64, simd=16, mode="offset").to(model.conv7.conv.weight.data.dtype))

    # head conv8: two's complement
    model.head.conv.weight.data.copy_(_load_head_w(text).to(model.head.conv.weight.data.dtype))

    # ---- bn inc/bias ----
    # 這些 name_pat 要跟 weights.hpp 宣告行一致（用比較寬鬆 regex）
    model.bnq0.inc_f.data.copy_(_load_vec_int(text, r"const ap_int<15>\s*conv_0_inc_new\[16\]\[1\]", out_ch=16, pe=16).to(model.bnq0.inc_f.data.dtype))
    model.bnq0.bias_f.data.copy_(_load_vec_int(text, r"const ap_int<25>\s*conv_0_bias_new\[16\]\[1\]", out_ch=16, pe=16).to(model.bnq0.bias_f.data.dtype))

    model.bnq1.inc_f.data.copy_(_load_vec_int(text, r"const ap_int<13>\s*conv_1_inc_new\[8\]\[4\]", out_ch=32, pe=8).to(model.bnq1.inc_f.data.dtype))
    model.bnq1.bias_f.data.copy_(_load_vec_int(text, r"const ap_int<21>\s*conv_1_bias_new\[8\]\[4\]", out_ch=32, pe=8).to(model.bnq1.bias_f.data.dtype))

    model.bnq2.inc_f.data.copy_(_load_vec_int(text, r"const ap_int<12>\s*conv_2_inc_new\[4\]\[16\]", out_ch=64, pe=4).to(model.bnq2.inc_f.data.dtype))
    model.bnq2.bias_f.data.copy_(_load_vec_int(text, r"const ap_int<21>\s*conv_2_bias_new\[4\]\[16\]", out_ch=64, pe=4).to(model.bnq2.bias_f.data.dtype))

    model.bnq3.inc_f.data.copy_(_load_vec_int(text, r"const ap_int<11>\s*conv_3_inc_new\[1\]\[64\]", out_ch=64, pe=1).to(model.bnq3.inc_f.data.dtype))
    model.bnq3.bias_f.data.copy_(_load_vec_int(text, r"const ap_int<21>\s*conv_3_bias_new\[1\]\[64\]", out_ch=64, pe=1).to(model.bnq3.bias_f.data.dtype))

    for i, bnq in zip([4,5,6,7], [model.bnq4, model.bnq5, model.bnq6, model.bnq7]):
        bnq.inc_f.data.copy_(_load_vec_int(text, rf"const ap_int<12>\s*conv_{i}_inc_new\[1\]\[64\]", out_ch=64, pe=1).to(bnq.inc_f.data.dtype))
        bnq.bias_f.data.copy_(_load_vec_int(text, rf"const ap_int<21>\s*conv_{i}_bias_new\[1\]\[64\]", out_ch=64, pe=1).to(bnq.bias_f.data.dtype))

    # ---- head bias ----
    # conv_8_bias_new[2][36]
    b8 = _load_vec_int(text, r"const ap_int<11>\s*conv_8_bias_new\[2\]\[36\]", out_ch=72, pe=2)
    if model.head.conv.bias is None:
        model.head.conv.bias = torch.nn.Parameter(torch.zeros((72,), dtype=model.head.conv.weight.dtype, device=model.head.conv.weight.device))
    model.head.conv.bias.data.copy_(b8.to(model.head.conv.bias.data.dtype))

    return model
