import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(input_ptr, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = input_ptr + pid*n_cols
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    
    input = tl.load(block_start+offsets, mask=mask, other=-float('inf'))
    
    casual_mask = offsets > pid
    input += tl.where(casual_mask, -float('inf'), 0.0)
    
    input -= tl.max(input, axis=0)
    numerator = tl.exp(input)
    denominator = tl.sum(numerator, axis=0)
    output= numerator/denominator
    
    output_start = output_ptr + pid*n_cols
    tl.store(output_start+offsets, output, mask=mask)

def softmax(input):
    output = torch.empty_like(input)
    n_rows, n_cols = input.shape
    softmax_kernel[(n_rows, )](input, output, n_cols, BLOCK_SIZE=triton.next_power_of_2(n_cols))
    return output


    
    
    

