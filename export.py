import os, json
import numpy as np
import torch
import structure

# ----------------------------
# 4-bit packing helpers
# ----------------------------

def _u4_twos(v: int) -> int:
    """
    two's complement bits:
      signed int in [-8..7] -> 4-bit two's complement (v & 0xF)
    """
    return int(v) & 0xF

def _u4_offset(v: int) -> int:
    """
    offset-binary bits for HLS conv1~7:
      signed int in [-8..7] -> stored nibble in [0..15] via (v + 8)
    """
    v = int(v)
    if v < -8:
        v = -8
    elif v > 7:
        v = 7
    return (v + 8) & 0xF

def _pack_int4(vals, mode: str = "twos"):  # vals length = SIMD
    """
    mode:
      - "twos"   : pack as two's complement bits (v & 0xF)
      - "offset" : pack as offset-binary bits (v+8)
    """
    word = 0
    if mode == "twos":
        fn = _u4_twos
    elif mode == "offset":
        fn = _u4_offset
    else:
        raise ValueError(f"unknown mode={mode}")

    for i, v in enumerate(vals):
        word |= (fn(v) << (4 * i))
    return word

def _hex_u(word: int) -> str:
    return f"\"0x{int(word):x}\""

def _hex_s(x: int) -> str:
    x = int(x)
    return f"\"-0x{(-x):x}\"" if x < 0 else f"\"0x{x:x}\""

def _format_nested(obj, indent=0):
    # 原本的 formatter 保留（寬鬆）
    if isinstance(obj, list):
        inner = ",\n".join(_format_nested(x, indent + 0) for x in obj)
        return "{"+inner+"}"
    return str(obj)

# ----------------------------
# Main exporter
# ----------------------------

@torch.no_grad()
def write_weights_hpp_from_model(model: structure.HLSLikeQATDetector, out_path: str):
    model.eval()

    # layer configs (對齊 DAC-SDC 2023 ultra_speed 這份設計)
    CFG = {
        0: dict(k=3, in_ch=3,  out_ch=16, pe=16, simd=3,  word=12,  kind="l0"),
        1: dict(k=3, in_ch=16, out_ch=32, pe=8,  simd=16, word=64,  kind="3x3"),
        2: dict(k=3, in_ch=32, out_ch=64, pe=4,  simd=32, word=128, kind="3x3"),
        3: dict(k=3, in_ch=64, out_ch=64, pe=1,  simd=64, word=256, kind="3x3"),
        4: dict(k=3, in_ch=64, out_ch=64, pe=1,  simd=16, word=64,  kind="3x3"),
        5: dict(k=3, in_ch=64, out_ch=64, pe=1,  simd=16, word=64,  kind="3x3"),
        6: dict(k=3, in_ch=64, out_ch=64, pe=1,  simd=16, word=64,  kind="3x3"),
        7: dict(k=3, in_ch=64, out_ch=64, pe=1,  simd=16, word=64,  kind="3x3"),
        8: dict(k=1, in_ch=64, out_ch=72, pe=2,  simd=8,  word=32,  kind="1x1"),
    }

    def get_w_int4(conv: structure.QuantConv2dInt4):
        # signed int4 in [-8..7] (torch.int8)
        return structure.fake_quant_signed(conv.conv.weight, conv.w_bits).to(torch.int8).cpu()  # [OC,IC,KH,KW]

    def get_b_int(conv: structure.QuantConv2dInt4):
        if conv.conv.bias is None:
            return None
        if conv.bias_bits is None:
            return structure.round_ste(conv.conv.bias).to(torch.int32).cpu()
        return structure.fake_quant_signed(conv.conv.bias, conv.bias_bits).to(torch.int32).cpu()

    def get_inc_bias(bnq: structure.HLSBnQuReLU):
        inc  = structure.fake_quant_signed(bnq.inc_f,  bnq.inc_bits).to(torch.int32).cpu()   # [C]
        bias = structure.fake_quant_signed(bnq.bias_f, bnq.bias_bits).to(torch.int32).cpu()  # [C]
        return inc, bias

    # conv0: 3 channels packed into 12b, bits are treated like ap_int<4> in HLS => two's complement bits
    def pack_conv0_l0(w):  # [16,3,3,3]
        out = [[[None for _ in range(3)] for _ in range(3)] for _ in range(16)]
        for oc in range(16):
            for kr in range(3):
                for kc in range(3):
                    v0 = _u4_twos(w[oc,0,kr,kc].item())
                    v1 = _u4_twos(w[oc,1,kr,kc].item())
                    v2 = _u4_twos(w[oc,2,kr,kc].item())
                    word = v0 | (v1<<4) | (v2<<8)
                    out[oc][kr][kc] = _hex_u(word)
        return out

    # conv1~7: **HLS DSP conv uses offset-binary semantics** (stored nibble = w+8)
    # so export must pack weights with mode="offset" for layers 1..7
    def pack_conv3x3(w, out_ch, in_ch, pe, simd, mode="twos"):
        simdnum = in_ch // simd
        infold = 3 * simdnum
        out = [[[None for _ in range(infold*(out_ch//pe))] for _ in range(3)] for _ in range(pe)]
        for oc in range(out_ch):
            p = oc % pe
            po = oc // pe
            for kr in range(3):
                for s in range(simdnum):
                    idx = po*infold + kr*simdnum + s
                    ic0, ic1 = s*simd, (s+1)*simd
                    for kc in range(3):
                        vals = w[oc, ic0:ic1, kr, kc].tolist()
                        out[p][kc][idx] = _hex_u(_pack_int4(vals, mode=mode))
        return out

    # head 1x1: bits are used like ap_int<4> => two's complement bits
    def pack_head_1x1(w, out_ch, in_ch, pe, simd):
        simdnum = in_ch // simd
        out = [[None for _ in range(simdnum*(out_ch//pe))] for _ in range(pe)]
        for oc in range(out_ch):
            p = oc % pe
            po = oc // pe
            for s in range(simdnum):
                idx = po*simdnum + s
                ic0, ic1 = s*simd, (s+1)*simd
                vals = w[oc, ic0:ic1, 0, 0].tolist()
                out[p][idx] = _hex_u(_pack_int4(vals, mode="twos"))
        return out

    def pack_vec_to_pe(vec, pe):  # vec: [OUT_CH]
        out_ch = vec.numel()
        assert out_ch % pe == 0
        cols = out_ch // pe
        arr = [[None for _ in range(cols)] for _ in range(pe)]
        for oc in range(out_ch):
            p = oc % pe
            po = oc // pe
            arr[p][po] = _hex_s(vec[oc].item())
        return arr

    # --- build all arrays ---
    w0 = pack_conv0_l0(get_w_int4(model.conv0))

    # conv1~7 MUST be offset
    w1 = pack_conv3x3(get_w_int4(model.conv1), **{k:CFG[1][k] for k in ["out_ch","in_ch","pe","simd"]}, mode="offset")
    w2 = pack_conv3x3(get_w_int4(model.conv2), **{k:CFG[2][k] for k in ["out_ch","in_ch","pe","simd"]}, mode="offset")
    w3 = pack_conv3x3(get_w_int4(model.conv3), **{k:CFG[3][k] for k in ["out_ch","in_ch","pe","simd"]}, mode="offset")
    w4 = pack_conv3x3(get_w_int4(model.conv4), **{k:CFG[4][k] for k in ["out_ch","in_ch","pe","simd"]}, mode="offset")
    w5 = pack_conv3x3(get_w_int4(model.conv5), **{k:CFG[5][k] for k in ["out_ch","in_ch","pe","simd"]}, mode="offset")
    w6 = pack_conv3x3(get_w_int4(model.conv6), **{k:CFG[6][k] for k in ["out_ch","in_ch","pe","simd"]}, mode="offset")
    w7 = pack_conv3x3(get_w_int4(model.conv7), **{k:CFG[7][k] for k in ["out_ch","in_ch","pe","simd"]}, mode="offset")

    # head stays twos
    w8 = pack_head_1x1(get_w_int4(model.head), **{k:CFG[8][k] for k in ["out_ch","in_ch","pe","simd"]})

    inc0, bias0 = get_inc_bias(model.bnq0)
    inc1, bias1 = get_inc_bias(model.bnq1)
    inc2, bias2 = get_inc_bias(model.bnq2)
    inc3, bias3 = get_inc_bias(model.bnq3)
    inc4, bias4 = get_inc_bias(model.bnq4)
    inc5, bias5 = get_inc_bias(model.bnq5)
    inc6, bias6 = get_inc_bias(model.bnq6)
    inc7, bias7 = get_inc_bias(model.bnq7)

    b8 = get_b_int(model.head)  # [72]

    # --- write file ---
    lines = []
    lines.append('#ifndef _WEIGHTS_HPP_')
    lines.append('#define _WEIGHTS_HPP_')
    lines.append('#include <ap_int.h>')

    # weights
    lines.append('const ap_uint<12>conv_0_w_new[16][3][3]=')
    lines.append(_format_nested(w0) + ';')

    lines.append('const ap_uint<64>conv_1_w_new[8][3][12]=')
    lines.append(_format_nested(w1) + ';')

    lines.append('const ap_uint<128>conv_2_w_new[4][3][48]=')
    lines.append(_format_nested(w2) + ';')

    lines.append('const ap_uint<256>conv_3_w_new[1][3][192]=')
    lines.append(_format_nested(w3) + ';')

    lines.append('const ap_uint<64>conv_4_w_new[1][3][768]=')
    lines.append(_format_nested(w4) + ';')

    lines.append('const ap_uint<64>conv_5_w_new[1][3][768]=')
    lines.append(_format_nested(w5) + ';')

    lines.append('const ap_uint<64>conv_6_w_new[1][3][768]=')
    lines.append(_format_nested(w6) + ';')

    lines.append('const ap_uint<64>conv_7_w_new[1][3][768]=')
    lines.append(_format_nested(w7) + ';')

    lines.append('const ap_uint<32>conv_8_w_new[2][288]=')
    lines.append(_format_nested(w8) + ';')

    # inc/bias (按 PE 分組)
    lines.append('const ap_int<15> conv_0_inc_new[16][1]=')
    lines.append(_format_nested(pack_vec_to_pe(inc0, 16)) + ';')
    lines.append('const ap_int<25> conv_0_bias_new[16][1]=')
    lines.append(_format_nested(pack_vec_to_pe(bias0, 16)) + ';')

    lines.append('const ap_int<13> conv_1_inc_new[8][4]=')
    lines.append(_format_nested(pack_vec_to_pe(inc1, 8)) + ';')
    lines.append('const ap_int<21> conv_1_bias_new[8][4]=')
    lines.append(_format_nested(pack_vec_to_pe(bias1, 8)) + ';')

    lines.append('const ap_int<12> conv_2_inc_new[4][16]=')
    lines.append(_format_nested(pack_vec_to_pe(inc2, 4)) + ';')
    lines.append('const ap_int<21> conv_2_bias_new[4][16]=')
    lines.append(_format_nested(pack_vec_to_pe(bias2, 4)) + ';')

    lines.append('const ap_int<11> conv_3_inc_new[1][64]=')
    lines.append(_format_nested(pack_vec_to_pe(inc3, 1)) + ';')
    lines.append('const ap_int<21> conv_3_bias_new[1][64]=')
    lines.append(_format_nested(pack_vec_to_pe(bias3, 1)) + ';')

    for i,(inc,bias) in enumerate([(inc4,bias4),(inc5,bias5),(inc6,bias6),(inc7,bias7)], start=4):
        lines.append(f'const ap_int<12> conv_{i}_inc_new[1][64]=')
        lines.append(_format_nested(pack_vec_to_pe(inc, 1)) + ';')
        lines.append(f'const ap_int<21> conv_{i}_bias_new[1][64]=')
        lines.append(_format_nested(pack_vec_to_pe(bias, 1)) + ';')

    # head bias
    lines.append('const ap_int<11> conv_8_bias_new[2][36]=')
    lines.append(_format_nested(pack_vec_to_pe(b8, 2)) + ';')

    lines.append('#endif')

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


@torch.no_grad()
def export_all_outputs(model, out_dir="export_out"):
    os.makedirs(out_dir, exist_ok=True)

    # 1) 軟體可用：PyTorch state_dict（可推論、可續訓）
    torch.save(model.state_dict(), os.path.join(out_dir, "qat_model_state.pth"))

    # 2) 軟體也可用：量化後整數參數（不打包，方便 numpy/C++ 自己吃）
    q = structure.export_hls_params(model)
    torch.save(q, os.path.join(out_dir, "quant_params.pt"))

    # npz 版本（更通用）
    np_dict = {}
    for k,v in q.items():
        if torch.is_tensor(v):
            np_dict[k] = v.detach().cpu().numpy()
        else:
            np_dict[k] = np.array(v)
    np.savez(os.path.join(out_dir, "quant_params.npz"), **np_dict)

    # 3) HLS 可直接替換的 weights.hpp
    write_weights_hpp_from_model(model, os.path.join(out_dir, "output__150_200_250_nhn.hpp"))
    print(f"[OK] exported to: {out_dir}/ (weights.hpp, qat_model_state.pth, quant_params.npz/.pt)")
