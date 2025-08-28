import triton
import triton.language as tl
import torch
import math


from typing import *
from flex_gemm.kernels.triton.spconv import sparse_submanifold_conv_fwd_implicit_gemm
from flex_gemm.kernels.triton.spconv.sparse_submanifold_conv_fwd_implicit_gemm import sparse_submanifold_conv_fwd_implicit_gemm_kernel
from flex_gemm.kernels.triton.spconv.sparse_submanifold_conv_bwd_implicit_gemm_splitk import sparse_submanifold_conv_bwd_weight_implicit_gemm_splitk


def ocnn_forward_implicit_gemm(data: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, neighbour: torch.Tensor):
    return sparse_submanifold_conv_fwd_implicit_gemm(data, weight, bias, neighbour, -1)


def ocnn_backward_weight_implicit_gemm(grad: torch.Tensor, data: torch.Tensor, neighbour: torch.Tensor):
    return sparse_submanifold_conv_bwd_weight_implicit_gemm_splitk(grad, data, neighbour, -1).permute(1, 2, 0)


def flex_gemm_forward_implicit(data: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, neighbour: torch.Tensor):
    return sparse_submanifold_conv_fwd_implicit_gemm(data, weight, bias, neighbour, -1)


def flex_gemm_backward_weight_implicit(grad: torch.Tensor, data: torch.Tensor, neighbour: torch.Tensor):
    return sparse_submanifold_conv_bwd_weight_implicit_gemm_splitk(grad, data, neighbour, -1)


def flex_gemm_forward_implicit_zero_init(data: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, neighbour: torch.Tensor):
    in_channel = data.shape[1]
    n_neighbour = neighbour.shape[1]
    n_output_octants = neighbour.shape[0]
    n_input_octants = data.shape[0]
    out_channel = weight.shape[0]
    assert weight.shape[1] == n_neighbour and weight.shape[2] == in_channel, "The shape of weight tensor shoule be (out_channel, neighbour_size, in_channel)"
    assert data.is_contiguous(), "data should be contiguous"
    assert weight.is_contiguous(), "weight should be contiguous"
    assert bias is None or bias.is_contiguous(), "bias shouled be contiguous"
    assert neighbour.is_contiguous(), "neighbour should be contiguous"
    result = torch.zeros((n_output_octants, out_channel), device=data.device, dtype=data.dtype)
    grid = lambda META: (triton.cdiv(out_channel, META['B2']) * triton.cdiv(n_output_octants, META['B1']), )
    sparse_submanifold_conv_fwd_implicit_gemm_kernel[grid](
        data, weight, bias, neighbour, result, -1,
        n_output_octants, int(math.log2(n_output_octants)), in_channel, out_channel, n_neighbour
    )
    return result



if __name__ == '__main__':
    data = torch.arange(128, dtype=torch.float32, device='cuda').reshape(1, -1)
    neighbour = torch.tensor([[0, ]], dtype=torch.long, device='cuda')
    weight = torch.ones((1, 1, 128), dtype=torch.float32, device='cuda')
    print(ocnn_forward_implicit_gemm(data, weight, None, neighbour))
    print(ocnn_forward_implicit_gemm(data * 10, weight * .1, None, neighbour))
    print(data.sum())
