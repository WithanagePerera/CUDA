// Starter CUDA Intro
// Based on Fireship's CUDA in 100 seconds
// 9/30/2024 (Happy Birthday :D)

#include <stdio.h>

// Use global specifier to define function or CUDA kernel that runs on GPU
__global__ void add(int* a, int* b, int* c)
{
    // Adds 2 vectors (a and b) together

    // Because we're doing calculations in parallel, we need to calculate the
    // global index of the thread and the block that we're working on 
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    c[i] = a[i] + b[i];
}

// Use managed specifier to tell CUDA that this data can be accessed from the 
// CPU and GPU without having to manually copy data between them
__managed__ int vector_a[400], vector_b[400], vector_c[400];

// main function for the CPU that runs the CUDA kernel
int main()
{
    // Populates our vectors with data
    for (int i = 0; i < 400; i++)
    {
        vector_a[i] = i;

        vector_b[i] = 400+i;
    }

    // Triple brackets allow us to configure the CUDA kernel launch to control how
    // many blocks and how many threads per block to use
    // Syntax: <<<blocks, threads per block>>>
    add<<<4, 100>>>(vector_a, vector_b, vector_c);

    // Pauses execution and waits for it to complete on the GPU
    cudaDeviceSynchronize();

    // Once completed on the GPU, we add up all of the vectors in vector c
    int result_sum = 0;
    for (int i = 0; i < 400; i++)
    {
        result_sum += vector_c[i];
        printf("Value at index %d: %d\n", i, vector_c[i]);
    }

    printf("Results: sum = % d", result_sum);
}