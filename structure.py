import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# -----------------------------
# 1) STE (Straight-Through Estimator) rounding
# -----------------------------
class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through: pretend round() is identity in backward
        return grad_output


def round_ste(x):
    return RoundSTE.apply(x)


# -----------------------------
# 1.5) STE clamp (forward == clamp, backward == identity)
#
# PyTorch's torch.clamp has zero gradient once values saturate.
# With hard-integer QAT, this often makes weights / BN-fold params
# get stuck at the quantization limits. We keep the *forward* exactly
# the same as hardware (saturating clamp), but pass gradients through.
# -----------------------------
class ClampSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, min_val, max_val):
        # min_val / max_val may be None
        if min_val is None and max_val is None:
            return x
        if max_val is None:
            return torch.clamp(x, min=min_val)
        if min_val is None:
            return torch.clamp(x, max=max_val)
        return torch.clamp(x, min=min_val, max=max_val)

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through: pretend clamp is identity
        return grad_output, None, None


def clamp_ste(x, min_val=None, max_val=None):
    return ClampSTE.apply(x, min_val, max_val)


# -----------------------------
# 2) Fake quant helpers
# -----------------------------
def fake_quant_signed(x, bits: int):
    """
    Signed integer quantization with STE rounding:
      range: [-2^(bits-1), 2^(bits-1)-1]
    returns float tensor containing integer values
    """
    qmin = -(2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1
    # STE clamp: keep forward identical to saturating clamp, but allow
    # gradients to flow even when saturated.
    xq = clamp_ste(round_ste(x), qmin, qmax)
    return xq


def fake_quant_unsigned(x, bits: int):
    """
    Unsigned integer quantization with STE rounding:
      range: [0, 2^bits-1]
    returns float tensor containing integer values
    """
    qmin = 0
    qmax = (2 ** bits) - 1
    # STE clamp for the same reason as above.
    xq = clamp_ste(round_ste(x), qmin, qmax)
    return xq


# -----------------------------
# 3) Weight int4 fake-quant wrapper for Conv2d
# -----------------------------
class QuantConv2dInt4(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1, bias=False,
                 act_bits_in: int = 4,
                 w_bits: int = 4,
                 bias_bits: int | None = None):  # <-- NEW
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=k, stride=s, padding=p, bias=bias)
        self.act_bits_in = act_bits_in
        self.w_bits = w_bits
        self.bias_bits = bias_bits  # <-- NEW

    def forward(self, x):
        xq = fake_quant_unsigned(x, self.act_bits_in)
        wq = fake_quant_signed(self.conv.weight, self.w_bits)

        b = self.conv.bias
        if b is not None and self.bias_bits is not None:
            b = fake_quant_signed(b, self.bias_bits)  # <-- NEW: quantize bias

        y = F.conv2d(xq, wq, bias=b, stride=self.conv.stride, padding=self.conv.padding)
        return y



# -----------------------------
# 4) HLS-like BN folding + Quantized ReLU (0..15)
#    y = clamp( round( (x * inc + bias) / 2^SHIFT ), 0..15 )
# -----------------------------
# class HLSBnQuReLU(nn.Module):
#     def __init__(self,
#                  ch: int,
#                  out_bits: int = 4,     # activation out bits (0..15)
#                  inc_bits: int = 15,    # inc integer bits (signed)
#                  bias_bits: int = 20,   # bias integer bits (signed) (tune per layer)
#                  shift: int = 15):      # SHIFT = (W_BIT-1)+DATA_BIT+L_SHIFT (per layer)
#         super().__init__()
#         self.ch = ch
#         self.out_bits = out_bits
#         self.inc_bits = inc_bits
#         self.bias_bits = bias_bits
#         self.shift = shift

#         # learnable float params; forward will quantize them into integer inc/bias via STE
#         # init to something reasonable (inc near 1*2^shift scale, bias near 0)
#         self.inc_f  = nn.Parameter(torch.ones(ch))
#         self.bias_f = nn.Parameter(torch.zeros(ch))

#     # def forward(self, x):
#     #     """
#     #     x: (B, C, H, W) float tensor representing integer-like accumulations
#     #     returns: (B, C, H, W) float tensor representing 0..15 integers
#     #     """
#     #     B, C, H, W = x.shape
#     #     assert C == self.ch

#     #     # quantize inc/bias into integers (STE)
#     #     inc_q  = fake_quant_signed(self.inc_f,  self.inc_bits).view(1, C, 1, 1)
#     #     bias_q = fake_quant_signed(self.bias_f, self.bias_bits).view(1, C, 1, 1)

#     #     # affine: x*inc + bias (integer domain, but stored in float)
#     #     v = x * inc_q + bias_q

#     #     # scale back by 2^shift with rounding:
#     #     # HLS: (v + 2^(shift-1)) >> shift  <=> round(v / 2^shift)
#     #     scale = float(2 ** self.shift)
#     #     v_scaled = round_ste(v / scale)

#     #     # ReLU + clamp to [0, 2^out_bits-1]
#     #     # Keep forward == ReLU, but use STE clamp so gradients can still
#     #     # update upstream params even if activations are mostly negative.
#     #     v_relu = clamp_ste(v_scaled, min_val=0.0, max_val=None)
#     #     y = fake_quant_unsigned(v_relu, self.out_bits)
#     #     return y
#     def forward(self, x):
#         """
#         Bit-accurate to HLS bn_qurelu_fixed:
#         bn_res = in*inc + bias
#         if bn_res > 0:
#             y = (bn_res + 2^(shift-1)) >> shift
#             y = min(y, 2^out_bits-1)
#         else:
#             y = 0
#         """
#         # conv 輸出在 torch 裡是 float，但值本質是整數；轉成 int64 做位移才會準
#         x_i = x.to(torch.int64)

#         inc_i  = fake_quant_signed(self.inc_f,  self.inc_bits).to(torch.int64).view(1, -1, 1, 1)
#         bias_i = fake_quant_signed(self.bias_f, self.bias_bits).to(torch.int64).view(1, -1, 1, 1)

#         v = x_i * inc_i + bias_i  # bn_res

#         half = 1 << (self.shift - 1)  # 2^(shift-1)
#         y = torch.where(v > 0, (v + half) >> self.shift, torch.zeros_like(v))

#         y = torch.clamp(y, 0, (1 << self.out_bits) - 1)
#         return y.to(torch.float32)

class FloorSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def floor_ste(x):
    return FloorSTE.apply(x)


class HLSBnQuReLU(nn.Module):
    def __init__(self,
                 ch: int,
                 out_bits: int = 4,
                 inc_bits: int = 15,
                 bias_bits: int = 20,
                 shift: int = 15):
        super().__init__()
        self.ch = ch
        self.out_bits = out_bits
        self.inc_bits = inc_bits
        self.bias_bits = bias_bits
        self.shift = shift
        self.inc_f  = nn.Parameter(torch.ones(ch))
        self.bias_f = nn.Parameter(torch.zeros(ch))

    def forward(self, x):
        """
        Train: float+STE (可反傳)
        Eval : int64 bit-exact (對齊硬體)
        """

        # ===== Eval: bit-exact (你拿去跟硬體比時用) =====
        if (not self.training):
            x_i = x.to(torch.int64)
            inc_i  = fake_quant_signed(self.inc_f,  self.inc_bits).to(torch.int64).view(1, -1, 1, 1)
            bias_i = fake_quant_signed(self.bias_f, self.bias_bits).to(torch.int64).view(1, -1, 1, 1)

            v = x_i * inc_i + bias_i
            half = 1 << (self.shift - 1)
            y = torch.where(v > 0, (v + half) >> self.shift, torch.zeros_like(v))
            y = torch.clamp(y, 0, (1 << self.out_bits) - 1)
            return y.to(torch.float32)

        # ===== Train: STE-friendly =====
        inc_q  = fake_quant_signed(self.inc_f,  self.inc_bits).view(1, -1, 1, 1)
        bias_q = fake_quant_signed(self.bias_f, self.bias_bits).view(1, -1, 1, 1)

        # v 是「整數語意」但用 float 存，方便反傳
        v = x * inc_q + bias_q

        scale = float(2 ** self.shift)
        half  = float(2 ** (self.shift - 1))

        # HLS: (v + half) >> shift  對正數相當於 floor((v+half)/2^shift)
        y_pos = floor_ste((v + half) / scale)
        y = torch.where(v > 0, y_pos, torch.zeros_like(v))

        # clamp to [0, 2^out_bits-1] (forward 同硬體；backward 走 STE)
        y = clamp_ste(y, 0.0, float((1 << self.out_bits) - 1))
        y = fake_quant_unsigned(y, self.out_bits)
        return y



# -----------------------------
# 5) Full HLS-like QAT model
#    input: (B,3,320,640) in [0,1] float
#    internal: quantize to uint8 (0..255) for conv0 input
# -----------------------------
class HLSLikeQATDetector(nn.Module):
    def __init__(self,
                 shifts,        # list length 8: conv0..conv7
                 inc_bits_new,  # list length 8
                 bias_bits_new, # list length 8
                 head_bias_bits_new=11):
        super().__init__()
        assert len(shifts) == 8 and len(inc_bits_new) == 8 and len(bias_bits_new) == 8

        # conv0: input is uint8 pixel domain
        self.conv0 = QuantConv2dInt4(3, 16, k=3, s=1, p=1, bias=False, act_bits_in=8, w_bits=4)
        self.bnq0  = HLSBnQuReLU(16, out_bits=4, inc_bits=inc_bits_new[0], bias_bits=bias_bits_new[0], shift=shifts[0])
        self.pool0 = nn.MaxPool2d(2, 2)

        self.conv1 = QuantConv2dInt4(16, 32, k=3, s=1, p=1, bias=False, act_bits_in=4, w_bits=4)
        self.bnq1  = HLSBnQuReLU(32, out_bits=4, inc_bits=inc_bits_new[1], bias_bits=bias_bits_new[1], shift=shifts[1])
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = QuantConv2dInt4(32, 64, k=3, s=1, p=1, bias=False, act_bits_in=4, w_bits=4)
        self.bnq2  = HLSBnQuReLU(64, out_bits=4, inc_bits=inc_bits_new[2], bias_bits=bias_bits_new[2], shift=shifts[2])
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = QuantConv2dInt4(64, 64, k=3, s=1, p=1, bias=False, act_bits_in=4, w_bits=4)
        self.bnq3  = HLSBnQuReLU(64, out_bits=4, inc_bits=inc_bits_new[3], bias_bits=bias_bits_new[3], shift=shifts[3])
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = QuantConv2dInt4(64, 64, k=3, s=1, p=1, bias=False, act_bits_in=4, w_bits=4)
        self.bnq4  = HLSBnQuReLU(64, out_bits=4, inc_bits=inc_bits_new[4], bias_bits=bias_bits_new[4], shift=shifts[4])

        self.conv5 = QuantConv2dInt4(64, 64, k=3, s=1, p=1, bias=False, act_bits_in=4, w_bits=4)
        self.bnq5  = HLSBnQuReLU(64, out_bits=4, inc_bits=inc_bits_new[5], bias_bits=bias_bits_new[5], shift=shifts[5])

        self.conv6 = QuantConv2dInt4(64, 64, k=3, s=1, p=1, bias=False, act_bits_in=4, w_bits=4)
        self.bnq6  = HLSBnQuReLU(64, out_bits=4, inc_bits=inc_bits_new[6], bias_bits=bias_bits_new[6], shift=shifts[6])

        self.conv7 = QuantConv2dInt4(64, 64, k=3, s=1, p=1, bias=False, act_bits_in=4, w_bits=4)
        self.bnq7  = HLSBnQuReLU(64, out_bits=4, inc_bits=inc_bits_new[7], bias_bits=bias_bits_new[7], shift=shifts[7])

        # head conv8: bias needs to match CONV_8_BIAS_BIT_NEW=11
        self.head = QuantConv2dInt4(64, 72, k=1, s=1, p=0,
                                    bias=True, act_bits_in=4, w_bits=4,
                                    bias_bits=head_bias_bits_new)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()

        # map normalized [0,1] -> [0,255] if needed
        if x.max() <= 1.5:
            x = x * 255.0

        x = fake_quant_unsigned(x, 8)  # uint8 pixels

        x = self.pool0(self.bnq0(self.conv0(x)))
        x = self.pool1(self.bnq1(self.conv1(x)))
        x = self.pool2(self.bnq2(self.conv2(x)))
        x = self.pool3(self.bnq3(self.conv3(x)))

        x = self.bnq4(self.conv4(x))
        x = self.bnq5(self.conv5(x))
        x = self.bnq6(self.conv6(x))
        x = self.bnq7(self.conv7(x))

        y = self.head(x)  # (B,72,20,40)
        return y



# -----------------------------
# 6) Export helpers: get int4 weights + int inc/bias (for packing weights.hpp)
# -----------------------------
@torch.no_grad()
def export_hls_params(model):
    out = {}

    def q_w(conv: QuantConv2dInt4):
        wq = fake_quant_signed(conv.conv.weight, conv.w_bits)
        return wq.to(torch.int8)

    def q_b(conv: QuantConv2dInt4):
        if conv.conv.bias is None:
            return None
        if conv.bias_bits is None:
            return round_ste(conv.conv.bias).to(torch.int32)
        return fake_quant_signed(conv.conv.bias, conv.bias_bits).to(torch.int32)

    def q_inc_bias(bnq: HLSBnQuReLU):
        inc_q  = fake_quant_signed(bnq.inc_f,  bnq.inc_bits).to(torch.int32)
        bias_q = fake_quant_signed(bnq.bias_f, bnq.bias_bits).to(torch.int32)
        return inc_q, bias_q, bnq.shift

    for i in range(8):
        conv = getattr(model, f"conv{i}")
        bnq  = getattr(model, f"bnq{i}")
        out[f"conv{i}.weight_int4"] = q_w(conv)
        out[f"conv{i}.bias_int32"]  = q_b(conv)  # conv0..7 是 None
        inc_q, bias_q, shift = q_inc_bias(bnq)
        out[f"conv{i}.inc_int"]  = inc_q
        out[f"conv{i}.bias_int"] = bias_q
        out[f"conv{i}.shift"]    = shift

    out["head.weight_int4"] = q_w(model.head)
    out["head.bias_int32"]  = q_b(model.head)   # ✅ 會量化到 11-bit

    return out


