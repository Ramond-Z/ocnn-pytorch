import triton
import triton.language as tl
import torch


from typing import *


@triton.jit
def ocnn_forward_implicit_gemm_kernel(
    data,
    weight,
    bias,
    neighbour,
    output,
    n, c_in, c_out, neighbour_size: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
    BK: tl.constexpr
):
    block_id = tl.program_id(axis=0)
    n_blocks_c_out = tl.cdiv(c_out, B2)
    block_id_c_out = block_id % n_blocks_c_out
    block_id_n = block_id // n_blocks_c_out

    num_k = tl.cdiv(c_in, BK)
    offset_n = (block_id_n * B1 + tl.arange(0, B1)) % n
    offset_c_out = (block_id_c_out * B2 + tl.arange(0, B2)) % c_out
    offset_k = tl.arange(0, BK)

    accumulator = tl.zeros((B1, B2), dtype=tl.float32)

    weight_ptr = weight + (offset_c_out[None, :] * neighbour_size * c_in + offset_k[:, None])

    for k in range(num_k * neighbour_size):
        neighbour_id = k // num_k
        block_id_k = k % num_k

        neighbour_offset_n = tl.load(neighbour + offset_n * neighbour_size + neighbour_id)
        mask = neighbour_offset_n != -1
        input_ptr = data + block_id_k * BK + (neighbour_offset_n[:, None] * c_in + offset_k[None, :])
        input_block = tl.load(input_ptr, mask=mask[:, None] & (offset_k[None, :] < c_in - block_id_k * BK), other=0.0)
        weight_block = tl.load(weight_ptr, mask=offset_k[:, None] < c_in - block_id_k * BK, other=0.0)
        accumulator = tl.dot(input_block, weight_block, accumulator)
        weight_ptr += min(BK, c_in - block_id_k * BK)
    c = accumulator.to(data.type.element_ty)

    if bias is not None:
        bias_block = tl.load(bias + offset_c_out)
        c += bias_block[None, :]

    out_offset_n = block_id_n * B1 + tl.arange(0, B1)
    out_offset_c_out = block_id_c_out * B2 + tl.arange(0, B2)
    out_ptr = output + (out_offset_n[:, None] * c_out + out_offset_c_out[None, :])
    out_mask = (out_offset_n[:, None] < n) & (out_offset_c_out[None, :] < c_out)
    tl.store(out_ptr, c, mask=out_mask)


def ocnn_forward_implicit_gemm(
    data,
    weight,
    bias,
    neighbour
):
    N = neighbour.shape[0]
    c_in = data.shape[1]
    c_out = weight.shape[0]
    n_neighbours = weight.shape[1]
    output = torch.empty((N, c_out), device=data.device, dtype=data.dtype)
    grid = lambda META: (triton.cdiv(c_out, META['B2']) * triton.cdiv(N, META['B1']), )
    # print(data.shape, weight.shape, neighbour.shape, output.shape)
    ocnn_forward_implicit_gemm_kernel[grid](
        data, weight, bias, neighbour, output,
        N, c_in, c_out, n_neighbours,
        64, 64, 64
    )
    return output


if __name__ == '__main__':
    data = torch.arange(128, dtype=torch.float32, device='cuda').reshape(1, -1)
    neighbour = torch.tensor([[0, ]], dtype=torch.long, device='cuda')
    weight = torch.ones((1, 1, 128), dtype=torch.float32, device='cuda')
    print(ocnn_forward_implicit_gemm(data, weight, None, neighbour))
    print(ocnn_forward_implicit_gemm(data * 10, weight * .1, None, neighbour))
    print(data.sum())
