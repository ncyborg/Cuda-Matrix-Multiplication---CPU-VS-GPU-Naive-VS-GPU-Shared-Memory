# Cuda-Matrix-Multiplication---CPU-VS-GPU-Naive-VS-GPU-Shared-Memory

This Program essentially was testing the speed of matrix multiplication when done on the following ways

CPU - Does matrix multiplication on arrays A and B on the CPU and inserts result into array C

GPU Naive - Does matrix multiplication on GPU where each thread in a block calculates a respective cell in the result array

GPU optimized - Does matrix multiplication on GPU where each thread in a block calculates a respective cell in the result array, 
the difference here is that all threads within the same block share memory with each other.

First test is to test how the calculation speed compares as the array size increases but the number of blocks stays at 4. 

Second test is to test how the calculation speed compares as the array size increases but the block size stays the same 4 * 4 
with a total of 16 threads per block

Third test is to test how the calculation speed compares as the array size increases but the block size stays at the max threads per block 32 * 32. 

Results: we found that CPU is faster with small vector sizes in general whereas with very large data sets GPU is much faster
