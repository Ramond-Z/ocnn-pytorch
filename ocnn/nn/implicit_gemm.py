import triton
import triton.language as tl
import torch


from typing import *
from .autotune import triton_autotune, autotune_config


@triton_autotune(
    configs=autotune_config,
    key=['c_in', 'c_out', 'neighbour_size']
)
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
    data: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    neighbour: torch.Tensor
):
    assert data.is_contiguous() and weight.is_contiguous and neighbour.is_contiguous()
    if bias is not None:
        assert bias.is_contiguous()
    N = neighbour.shape[0]
    c_in = data.shape[1]
    c_out = weight.shape[0]
    n_neighbours = weight.shape[1]
    output = torch.empty((N, c_out), device=data.device, dtype=data.dtype)
    def grid(META): return (triton.cdiv(c_out, META['B2']) * triton.cdiv(N, META['B1']), )
    ocnn_forward_implicit_gemm_kernel[grid](
        data, weight, bias, neighbour, output,
        N, c_in, c_out, n_neighbours,
        # 64, 64, 64
    )
    return output


heuristics = {
    'BV': lambda meta: max(1, meta['B2'] // meta['Ci']),
    'BCi': lambda meta: min(meta['Ci'], meta['B2']),
}


@triton_autotune(
    configs=autotune_config,
    key=['LOGN', 'Ci', 'Co', 'V'],
)
@triton.heuristics(heuristics)
@triton.jit
def ocnn_backward_weight_implicit_gemm_kernel(
    grad_output,
    input,
    neighbor,
    grad_weight,
    # Tensor dimensions
    N, LOGN, Ci, Co, V: tl.constexpr,
    # Meta-parameters
    B1: tl.constexpr,   # Block size for Co dimension
    B2: tl.constexpr,   # Block size for V * Ci dimension
    BK: tl.constexpr,   # Block size for K dimension (N)
    BV: tl.constexpr,   # Block size for V dimension
    BCi: tl.constexpr,  # Block size for Ci dimension
):
    block_id_co = tl.program_id(axis=0)
    block_id_vci = tl.program_id(axis=1)

    # Create pointers for submatrices of A and B.
    num_k = tl.cdiv(N, BK)  # Number of blocks in K dimension
    offset_co = (block_id_co * B1 + tl.arange(0, B1)) % Co                          # (B1,)
    offset_v = (tl.arange(0, BV) + (block_id_vci // (Ci // BCi)) * BV) % V          # (BV,)
    offset_ci = (tl.arange(0, BCi) + (block_id_vci % (Ci // BCi)) * BCi) % Ci       # (BCi,)
    offset_k = tl.arange(0, BK)                                                     # (BK,)
    neighbor_ptr = neighbor + (offset_k[:, None] * V + offset_v[None, :])           # (BK, BV)
    grad_output_ptr = grad_output + (offset_k[None, :] * Co + offset_co[:, None])   # (B1, BK)

    # Create a block of the output matrix C.
    accumulator = tl.zeros((B1, BV * BCi), dtype=tl.float32)

    # Iterate along V*Ci dimension.
    for k in range(num_k):
        # Calculate pointers to input matrix.
        input_offset_n = tl.load(neighbor_ptr, mask=offset_k[:, None] < N - k * BK, other=-1)   # (BK, BV)
        input_ptr = input + (input_offset_n[:, :, None] * Ci + offset_ci[None, None, :])                # (BK, BV, BCi)
        # Load the next block of input and weight.
        grad_output_block = tl.load(grad_output_ptr, mask=offset_k[None, :] < N, other=0.0)
        input_block = tl.load(input_ptr, mask=input_offset_n[:, :, None] != -1, other=0.0).reshape(BK, BV * BCi)
        # Accumulate along the K dimension.
        accumulator = tl.dot(grad_output_block, input_block, accumulator, 'tf32')           # (B1, BV * BCi)
        # Advance pointers.
        grad_output_ptr += BK * Co
        neighbor_ptr += BK * V
    c = accumulator.to(grad_output.type.element_ty)

    # Write back the block of the output matrix with masks.
    grad_weight_offset_co = block_id_co * B1 + tl.arange(0, B1)
    grad_weight_offset_vci = block_id_vci * BV * BCi + tl.arange(0, BV * BCi)
    grad_weight_ptr = grad_weight + (grad_weight_offset_co[:, None] * V * Ci + grad_weight_offset_vci[None, :])
    grad_weight_mask = (grad_weight_offset_co[:, None] < Co) & (grad_weight_offset_vci[None, :] < V * Ci)
    tl.store(grad_weight_ptr, c, mask=grad_weight_mask)


def ocnn_backward_weight_implicit_gemm(
    grad: torch.Tensor,
    data: torch.Tensor,
    neighbour: torch.Tensor
):
    assert grad.is_contiguous() and data.is_contiguous() and neighbour.is_contiguous()
    n = grad.shape[0]
    c_in = data.shape[1]
    c_out = grad.shape[1]
    n_neighbours = neighbour.shape[1]
    weight_grad = torch.empty((c_out, n_neighbours, c_in), device=grad.device, dtype=grad.dtype)
    def grid(META): return (triton.cdiv(c_out, META['B1']), triton.cdiv(n_neighbours * c_in, META['BV'] * META['BCi']))
    ocnn_backward_weight_implicit_gemm_kernel[grid](
        grad,
        data,
        neighbour,
        weight_grad,
        n, None, c_in, c_out, n_neighbours,
        # 16, 16, 16,
    )
    return weight_grad.permute(1, 2, 0)


if __name__ == '__main__':
    data = torch.arange(128, dtype=torch.float32, device='cuda').reshape(1, -1)
    neighbour = torch.tensor([[0, ]], dtype=torch.long, device='cuda')
    weight = torch.ones((1, 1, 128), dtype=torch.float32, device='cuda')
    print(ocnn_forward_implicit_gemm(data, weight, None, neighbour))
    print(ocnn_forward_implicit_gemm(data * 10, weight * .1, None, neighbour))
    print(data.sum())
