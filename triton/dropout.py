# https://triton-lang.org/main/getting-started/tutorials/04-low-memory-dropout.html#sphx-glr-getting-started-tutorials-04-low-memory-dropout-py
import torch

import triton
import triton.language as tl

@triton.jit
def dropout_kernel(
    x_ptr, dropout_mask_ptr, out_ptr, n_elements, p, BLOCK_SIZE: tl.constexpr
):
    pid=tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets<n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    dropout_mask=tl.load(dropout_mask_ptr + offsets, mask=mask)
    
    output=tl.where(dropout_mask, x/(1-p), 0.0)
    tl.store(out_ptr + offsets, output, mask=mask)

def dropout(x, dropout_mask, p):
    output=torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    dropout_kernel[grid](x, dropout_mask, output, n_elements, p, BLOCK_SIZE=1024)
    return output

def test():
    device="cuda"
    x = torch.randn(4,4).to(device)
    p = 0.5
    dropout_mask = (torch.rand(size=(4, 4)) > p).to(torch.int32).to(device)
    output = dropout(x, dropout_mask, p=p)
    print(output)

if __name__ == "__main__":
    test() 
    
    
    