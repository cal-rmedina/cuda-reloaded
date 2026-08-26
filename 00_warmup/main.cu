#include <cassert>
#include <iostream>

#include <cuda_runtime.h>

__global__ void print_threads_kernel() {

    printf("block-%u, thread(%u, %u, %u)\n",
        blockIdx.x, threadIdx.x, threadIdx.y, threadIdx.z);
}

__global__ void global_id_kernel(const unsigned int elements,
    unsigned int* d_array_id) {

    const unsigned int tid =  blockIdx.x * blockDim.x + threadIdx.x;
        d_array_id[tid] = tid;
}

// TODO 0: Run the program, check 1st kernel lauch, modify to see output
//
//         How many threads per thread-block are launched?
//         How many blocks are launched?
//         How many output lines are expected (from the kernel)?
//         What are the thread-block & grid dimensions?
//         What is the usage of cudaDeviceSynchronize?

int main() {

    constexpr unsigned int threads = 1<<5;

    // Get GPU 0 properties (list available devices if > 1)
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const unsigned int max_threads_per_block = prop.maxThreadsPerBlock;
    printf("Max thread-block size: %u\n", max_threads_per_block);

// TODO 1: Change the thread-block (1D -> 2D) & check the output
//
//         How many threads per thread-block are launched (2D)?
//         How many output lines are expected (from the kernel)?
//         What are the thread-block & grid dimensions?

    dim3 thread_block(threads);
//    dim3 thread_block(threads, threads);

    // 1st kernel launch
    print_threads_kernel<<<1, thread_block>>>();
//    cudaDeviceSynchronize();


// TODO 2: Uncomment the following part & run the code with 1D thread-blocks
//
//         Compute the active threads for the 2nd kernel launch
//         How many active threads (total) are launched?
//         Are all the elements of the array covered?
//         Are we accessing elements out of bounds?
//
// Change the number of elements to 1000000, modify if needed to work
// regardless the array size
//
//         Are there idle threads after the kernel modification?
//         Are we accessing elements out of bounds?
//         Do we need cudaDeviceSynchronize after the 2nd kernel launch?
/*
    constexpr unsigned int elements = 1 << 20;
    constexpr size_t elements_size = elements * sizeof(unsigned int);

    // GPU memory allocation
    unsigned int *d_array_id;
    cudaMalloc(&d_array_id, elements_size);

    // Allocating enough threads to cover the array
    const unsigned int blocks = (elements + threads - 1) / threads;

    printf("Active threads: %u, array size: %u\n", blocks * threads, elements);

    // 2nd kernel launch
    global_id_kernel<<<blocks, thread_block>>>(elements, d_array_id);

    unsigned int *h_array_id;
    cudaMallocHost(&h_array_id, elements_size);

    cudaMemcpy(h_array_id, d_array_id, elements_size,
        cudaMemcpyDeviceToHost);

    // Checking that the device array is correctly set
    for (unsigned int i = 0; i < elements; ++i) {
        assert(h_array_id[i] == i);
    }

    // GPU memory free
    cudaFree(d_array_id);
    cudaFreeHost(h_array_id);
*/
    return 0;
}

