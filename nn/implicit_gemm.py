import triton
import triton.language as tl
import torch
import math


from typing import *
from flex_gemm.kernels.triton.spconv import sparse_submanifold_conv_fwd_implicit_gemm
from flex_gemm.kernels.triton.spconv.sparse_submanifold_conv_bwd_implicit_gemm_splitk import sparse_submanifold_conv_bwd_weight_implicit_gemm_splitk


def ocnn_forward_implicit_gemm(data: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, neighbour: torch.Tensor):
    return sparse_submanifold_conv_fwd_implicit_gemm(data, weight, bias, neighbour, -1)

def ocnn_backward_weight_implicit_gemm(grad: torch.Tensor, data: torch.Tensor, neighbour: torch.Tensor):
    return sparse_submanifold_conv_bwd_weight_implicit_gemm_splitk(grad, data, neighbour, -1).permute(1, 2, 0)



if __name__ == '__main__':
    data = torch.arange(128, dtype=torch.float32, device='cuda').reshape(1, -1)
    neighbour = torch.tensor([[0, ]], dtype=torch.long, device='cuda')
    weight = torch.ones((1, 1, 128), dtype=torch.float32, device='cuda')
    print(ocnn_forward_implicit_gemm(data, weight, None, neighbour))
    print(ocnn_forward_implicit_gemm(data * 10, weight * .1, None, neighbour))
    print(data.sum())
