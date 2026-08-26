#include <iostream>
#include <cassert>

#include <cuda_runtime.h>
#include <cooperative_groups.h>

void filling_square_image_chessboard_pattern_block_width(
    const int image_width,
    const int block_width,
    int* block_average,
    unsigned char* pixel_value)
{
    // Random numbers seed
    srand(time(nullptr));

    const int blocks_x = image_width / block_width;

    for (int y = 0; y < blocks_x; y++) {
        for (int x = 0; x < blocks_x; ++x) {

            // Average chessboard block color: even black 0, odd white 255
            constexpr int color_half = 128;
            const int color = ((x + y) % 2 == 0) ? 0 : color_half;

            // Accumulating the pixel value per block
            int block_px_value = 0;

            // Fill square block (block_width) with the same color
            for (int dy = 0; dy < block_width; ++dy) {
                for (int dx = 0; dx < block_width; ++dx) {
                    const int px = (y * block_width + dy) * image_width
                                            + (x * block_width + dx);

                    const int current_px = rand() % color_half + color; 

                    // Store current pixel value on global memory
                    pixel_value[px] = static_cast<unsigned char>(current_px);

                    block_px_value += current_px;
                }
            }

            // Computing block id given the block coordinates (x,y)
            const int block_1d_id = y * blocks_x + x;
            block_average[block_1d_id] = block_px_value;
        }
    }
}

// --- TEST HOST FUNCTIONS ---
void test_cumulative_blocks(const int image_width,
                            const int block_width,
                            const int* h_block_average,
                            const int* d_block_average)
{
    const int blocks_x = image_width / block_width;
    const int total_blocks = blocks_x * blocks_x;

    auto* block_average_temp = new int[total_blocks];

    cudaMemcpy( block_average_temp, d_block_average, total_blocks * sizeof(int),
        cudaMemcpyDeviceToHost);

    // Loop over the blocks
    for (int i = 0; i < total_blocks; i++) {
        assert(block_average_temp[i] == h_block_average[i]);
    }

    free(block_average_temp);
}


void test_chessboard_colors(const int image_width,
                            const int block_width,
                            const bool inverted,
                            const unsigned char* d_pixel)
{
    const int blocks_x = image_width / block_width;
    const int total_pixels = image_width * image_width;

    auto* h_pixel = new unsigned char[total_pixels];

    cudaMemcpy( h_pixel, d_pixel, total_pixels * sizeof(unsigned char),
        cudaMemcpyDeviceToHost);

    // Loop over the blocks
    for (int y = 0; y < blocks_x; y++) {
        for (int x = 0; x < blocks_x; ++x) {

            // Corresponding color for the current block
            unsigned char color = ((x + y) % 2 == 0) ? 0: 255;

            if (inverted) color = (color == 0) ? 255: 0;

            // Loop over the block pixels
            for (int dy = 0; dy < block_width; ++dy) {
                for (int dx = 0; dx < block_width; ++dx) {

                    // Computing global index
                    const int px = (y * block_width + dy) * image_width
                                            + (x * block_width + dx);
                    assert(h_pixel[px] == color);
                }
            }
        }
    }
    free(h_pixel);
}

// --- DEVICE FUNCTIONS ---
__global__ void block_cumulative_warp_reduction_kernel(
        const unsigned char *d_pixel,
        int *d_block_average,
        const int image_width,
        const int block_width)
{
    const int block_1d_id = blockIdx.x;
    const int blocks_x = image_width / block_width;
    const int pixels_per_block = block_width * block_width;

    // Accumulated pixel value per thread
    int thread_px_sum = 0;

    // Stride loop over the number of pixels per block  
    for (int px = threadIdx.x; px < pixels_per_block; px += blockDim.x) {
        const int local_x = px % block_width;
        const int local_y = px / block_width;

        // Computing 1st pixel coordinates (x, y) + local coordinates    
        const int curr_px_x = (block_1d_id % blocks_x) * block_width + local_x;
        const int curr_px_y = (block_1d_id / blocks_x) * block_width + local_y;
   
        // Computing global 1D pixel id 
        const int px_1d_id = curr_px_y * image_width + curr_px_x;

        // Adding pixel value per thread 
        thread_px_sum += d_pixel[px_1d_id];
    }

    // Warp reduction using shuffle operations; all threads active
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_px_sum += __shfl_down_sync(0xffffffff, thread_px_sum, offset);
     }

    // 1st thread of each warp adds the value to the average array
    if (threadIdx.x % 32 == 0) {
       atomicAdd(&d_block_average[block_1d_id], thread_px_sum);
    }
}

__global__ void block_cumulative_multi_read_kernel(
                    const unsigned char *d_pixel,
                    int *d_block_average,
                    const int image_width,
                    const int block_width)
{
    const int block_1d_id = blockIdx.x;
    const int blocks_x = image_width / block_width;
    const int pixels_per_block = block_width * block_width;

    // Dividing by 4 due to multi-element memory read 
    const int group_transactions = pixels_per_block >> 2;

    // Stride loop over the number of reading transactions
    for (int group = threadIdx.x; group < group_transactions; group += blockDim.x) {

        // Current linear pixel given by: group * 4
        const int px = group << 2;
        const int local_x = px % block_width;
        const int local_y = px / block_width;

        // Computing 1st pixel coordinates (x, y) + local coordinates    
        const int curr_px_x = (block_1d_id % blocks_x) * block_width + local_x;
        const int curr_px_y = (block_1d_id / blocks_x) * block_width + local_y;
   
        // Computing global 1D pixel id 
        const int px_1d_id = curr_px_y * image_width + curr_px_x;

        const uchar4 v = *reinterpret_cast<const uchar4*>(d_pixel + px_1d_id);
        const int sum_4_values = v.x + v.y + v.z + v.w;

        // Accumulate pixel value on average array
        atomicAdd(&d_block_average[block_1d_id], sum_4_values);
    }
}

__global__ void assign_average_block_kernel(unsigned char* d_pixel,
                                            const int* d_block_average,
                                            const int image_width,
                                            const int block_width)
{
    const int thread_1d_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_pixels = image_width * image_width;

    if (thread_1d_id < total_pixels) {
        const int blocks_x = image_width / block_width;

        const int px_x = thread_1d_id % image_width;
        const int px_y = thread_1d_id / image_width;

        const int block_x = px_x / block_width;
        const int block_y = px_y / block_width;

        const int block_1d_id = block_y * blocks_x + block_x;
        const int pixels_per_block = block_width * block_width;

        // Computing block average divining by the number of pixels per block
        const int block_value = d_block_average[block_1d_id] / pixels_per_block;

        d_pixel[thread_1d_id] = static_cast<unsigned char>(block_value);
    }
}

__global__ void round_color_kernel(unsigned char* d_pixel,
                                   const int image_width)
{
    const int thread_1d_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_pixels = image_width * image_width;

    if (thread_1d_id < total_pixels) {
        // Pixexl either black (0) or white (255) depending on average value
        d_pixel[thread_1d_id] = (d_pixel[thread_1d_id] < 128) ? 0 : 255;
    }
}

__global__ void invert_color_kernel(unsigned char* d_pixel,
                                    const int image_width)
{
    const int thread_1d_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_pixels = image_width * image_width;

    if (thread_1d_id < total_pixels) {
        // Change color pixel: black <--> white
        d_pixel[thread_1d_id] = (d_pixel[thread_1d_id] == 0) ? 255 : 0;
    }
}

int main()
{
    constexpr int image_width = 1<<13;      // 8192 x 8192 px image
    constexpr int block_width = 1<<10;      // 1024 x 1024 px block
    constexpr int threads = 1<<8;

    constexpr int total_pixels = image_width * image_width;
    constexpr int blocks_x = image_width / block_width;
    constexpr int total_blocks = blocks_x * blocks_x;

    auto* h_pixel = new unsigned char[total_pixels];
    auto* h_block_average = new int[total_blocks];


    // Creating the noisy chessboard (Host)
    filling_square_image_chessboard_pattern_block_width(
          image_width, block_width, h_block_average, h_pixel);

    unsigned char *d_pixel;
    cudaMalloc(&d_pixel, total_pixels *sizeof(unsigned char));
    cudaMemcpy( d_pixel, h_pixel, total_pixels * sizeof (unsigned char),
                cudaMemcpyHostToDevice);

    // GPU allocation and initialization (all elements set to 0)
    int *d_block_average;
    cudaMalloc(&d_block_average, total_blocks *sizeof(int));
    cudaMemset(d_block_average, 0, total_blocks * sizeof(int));

    // Compute block pixel average; one t-b per block
    block_cumulative_warp_reduction_kernel<<<total_blocks, threads>>>(d_pixel,
                                                       d_block_average,
                                                       image_width,
                                                       block_width);

//    block_cumulative_multi_read_kernel<<<total_blocks, threads>>>(d_pixel,
//                                                       d_block_average,
//                                                       image_width,
//                                                       block_width);

    // TEST: compare device and host accumulated pixel value
    test_cumulative_blocks(image_width, block_width, h_block_average,
                           d_block_average);

    // Assign block average to al the block pixels; one pixel per thread
    constexpr int blocks_covering_pixels = (total_pixels + threads -1)/ threads;
    assign_average_block_kernel<<<blocks_covering_pixels, threads>>>(d_pixel,
            d_block_average,
            image_width,
            block_width);

    // Rounding the color to either black or white depending on average value
    round_color_kernel<<<blocks_covering_pixels, threads>>>(d_pixel, image_width);

    // TEST: avoiding OpenCV functions, check array colors
    test_chessboard_colors(image_width, block_width, false, d_pixel);

    invert_color_kernel<<<blocks_covering_pixels, threads>>>(d_pixel, image_width);

    // TEST: avoiding OpenCV functions, check array inverted colors
    test_chessboard_colors(image_width, block_width, true, d_pixel);

    cudaFree(d_pixel);
    cudaFree(d_block_average);

    free(h_pixel);
    free(h_block_average);

    return 0;
}

