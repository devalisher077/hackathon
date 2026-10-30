"""Original fused Qwen2 rotary embedding and its transpose derivative.

The two halves rotate as (a, b) -> (-b, a); arithmetic stays fp32 and
separate multiply/add rounding is preserved with enable_fp_fusion=False.
"""

import triton
import triton.language as tl


@triton.jit
def rope_pair_forward_kernel(
    q_ptr, k_ptr, cos_ptr, sin_ptr, oq_ptr, ok_ptr,
    q_batch, q_time, k_batch, k_time,
    cos_batch, cos_time, sin_batch, sin_time,
    T, D: tl.constexpr, Q: tl.constexpr, K: tl.constexpr,
    COS_BROADCAST: tl.constexpr, SIN_BROADCAST: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    batch, time = row // T, row % T
    column = tl.arange(0, BLOCK)
    lane = column % D
    partner = tl.where(lane < D // 2, column + D // 2, column - D // 2)
    is_q = column < Q
    is_k = (column >= Q) & (column < Q + K)
    q = tl.load(q_ptr + batch * q_batch + time * q_time + column, is_q, 0)
    qr = tl.load(q_ptr + batch * q_batch + time * q_time + partner, is_q, 0)
    k = tl.load(k_ptr + batch * k_batch + time * k_time + column - Q, is_k, 0)
    kr = tl.load(k_ptr + batch * k_batch + time * k_time + partner - Q, is_k, 0)
    value, rotated = tl.where(is_q, q, k), tl.where(is_q, qr, kr)
    rotated = tl.where(lane < D // 2, -rotated, rotated)
    cb = 0 if COS_BROADCAST else batch
    sb = 0 if SIN_BROADCAST else batch
    cosine = tl.load(cos_ptr + cb * cos_batch + time * cos_time + lane)
    sine = tl.load(sin_ptr + sb * sin_batch + time * sin_time + lane)
    result = value * cosine + rotated * sine
    tl.store(oq_ptr + row * Q + column, result, is_q)
    tl.store(ok_ptr + row * K + column - Q, result, is_k)


@triton.jit
def rope_pair_backward_kernel(
    gq_ptr, gk_ptr, cos_ptr, sin_ptr, dq_ptr, dk_ptr,
    gqb, gqh, gqt, gqd, gkb, gkh, gkt, gkd,
    cos_batch, cos_time, sin_batch, sin_time,
    T, D: tl.constexpr, Q: tl.constexpr, K: tl.constexpr,
    COS_BROADCAST: tl.constexpr, SIN_BROADCAST: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    batch, time = row // T, row % T
    column = tl.arange(0, BLOCK)
    lane = column % D
    partner_lane = tl.where(lane < D // 2, lane + D // 2, lane - D // 2)
    qhead, khead = column // D, (column - Q) // D
    is_q = column < Q
    is_k = (column >= Q) & (column < Q + K)
    qp = gq_ptr + batch * gqb + qhead * gqh + time * gqt
    kp = gk_ptr + batch * gkb + khead * gkh + time * gkt
    q = tl.load(qp + lane * gqd, is_q, 0)
    qr = tl.load(qp + partner_lane * gqd, is_q, 0)
    k = tl.load(kp + lane * gkd, is_k, 0)
    kr = tl.load(kp + partner_lane * gkd, is_k, 0)
    value, rotated = tl.where(is_q, q, k), tl.where(is_q, qr, kr)
    cb = 0 if COS_BROADCAST else batch
    sb = 0 if SIN_BROADCAST else batch
    cosine = tl.load(cos_ptr + cb * cos_batch + time * cos_time + lane)
    other_sine = tl.load(sin_ptr + sb * sin_batch + time * sin_time + partner_lane)
    other = rotated * other_sine
    result = value * cosine + tl.where(lane < D // 2, other, -other)
    tl.store(dq_ptr + row * Q + column, result, is_q)
    tl.store(dk_ptr + row * K + column - Q, result, is_k)
