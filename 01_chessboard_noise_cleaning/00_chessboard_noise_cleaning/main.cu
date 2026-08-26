#include <iostream>
#include <cuda_runtime.h>
#include <cassert>

// TEST: openCV2 just for visualization
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

void filling_square_image_chessboard_pattern_block_width(
    const int image_width,
    const int block_width,
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

            // Fill square block (block_width) with the same color
            for (int dy = 0; dy < block_width; ++dy) {
                for (int dx = 0; dx < block_width; ++dx) {
                    const int px = (y * block_width + dy) * image_width
                                            + (x * block_width + dx);
                    pixel_value[px] = rand() % color_half + color;
                }
            }
        }
    }
}

void test_chessboard_inverted_colors(const int image_width,
                                     const int block_width,
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
            const unsigned char color = ((x + y) % 2 == 0) ? 255: 0;

            // Loop over the block pixels
            for (int dy = 0; dy < block_width; ++dy) {
                for (int dx = 0; dx < block_width; ++dx) {
                    const int px = (y * block_width + dy) * image_width
                                            + (x * block_width + dx);
                    //TODO: Add assert to avoid if condition
                    if(h_pixel[px] != color){
                        printf("Wrong block: (%d, %d)\n", x, y);
                        exit(1);
                    }
                }
            }
        }
    }
    free(h_pixel);
}

// TODO:
//  What is the atomic operation doing? 
//  Is the 1D array an easy way to address each pixel?
//  Is there any better way to accumulate values? 
//  Is it efficient modifying one "unsigned char" element per thread?
__global__ void block_cumulative_kernel(const unsigned char *d_pixel,
                                        int *d_block_average,
                                        const int image_width,
                                        const int block_width)
{
    const int block_1d_id = blockIdx.x;
    const int blocks_x = image_width / block_width;
    const int pixels_per_block = block_width * block_width;

    // Stride loop over the number of pixels per block  
    for (int px = threadIdx.x; px < pixels_per_block; px += blockDim.x) {
        const int local_x = px % block_width;
        const int local_y = px / block_width;

        // Computing 1st pixel coordinates (x, y) + local coordinates    
        const int curr_px_x = (block_1d_id % blocks_x) * block_width + local_x;
        const int curr_px_y = (block_1d_id / blocks_x) * block_width + local_y;
   
        // Computing global 1D pixel id 
        const int px_1d_id = curr_px_y * image_width + curr_px_x;

        // Accumulate pixel value on average array
        atomicAdd(&d_block_average[block_1d_id], d_pixel[px_1d_id]);
    }
}

// TODO:
//  Is the thread-per-pixel approach the optimal?
//  How many times is the division / pixels_per_block computed?
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

// TODO:
//  Is the thread-per-pixel approach the optimal?
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

// TODO:
//  Is the thread-per-pixel approach the optimal?
//  Can we combine invert_color_kernel and round_color_kernel?
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

int main() {

    // 
    constexpr int image_width = 1<<13;      // 8192x8192 px per image
    constexpr int block_width = 1<<10;      // 1024x1024 px per block
    constexpr int threads = 1<<8;           // threads per thread-block

    constexpr int total_pixels = image_width * image_width;
    constexpr int blocks_x = image_width / block_width;
    constexpr int total_blocks = blocks_x * blocks_x;

    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const int max_threads_per_block = prop.maxThreadsPerBlock;

    assert(threads <= max_threads_per_block);
    assert(threads <= block_width);


    // TODO: Once understood, use unsigned char array and check if it is faster
    // auto* h_pixel = new unsigned char[total_pixels];
    // ...
    // free(h_pixel);
    cv::Mat h_pixel(image_width, image_width, CV_8UC1);

    // Creating the noisy chessboard on Host
    filling_square_image_chessboard_pattern_block_width(
          image_width, block_width, h_pixel.ptr());

    // TEST: checking output noisy chessboard
    cv::imwrite("0_noisy_chessboard.png", h_pixel);

    unsigned char *d_pixel;
    cudaMalloc(&d_pixel, total_pixels *sizeof(unsigned char));

    int *d_block_average;
    cudaMalloc(&d_block_average, total_blocks *sizeof(int));
    cudaMemset(d_block_average, 0, total_blocks * sizeof(int));

    cudaMemcpy( d_pixel, h_pixel.ptr(), total_pixels * sizeof (unsigned char),
                cudaMemcpyHostToDevice);

    // Compute block pixel average; one t-b per block
    block_cumulative_kernel<<<total_blocks, threads>>>(d_pixel,
                                                       d_block_average,
                                                       image_width,
                                                       block_width);

    // Assign block average to al the block pixels; one pixel per thread
    constexpr int blocks_covering_pixels = (total_pixels + threads -1)/ threads;
    assign_average_block_kernel<<<blocks_covering_pixels, threads>>>(d_pixel,
            d_block_average,
            image_width,
            block_width);

    // TEST: DtH copy: checking output average chessboard image
    cudaMemcpy( h_pixel.ptr(), d_pixel, total_pixels * sizeof(unsigned char),
        cudaMemcpyDeviceToHost);
    cv::imwrite("1_average_chessboard.png", h_pixel);

    // Rounding the color to either black or white depending on average value
    round_color_kernel<<<blocks_covering_pixels, threads>>>(d_pixel, image_width);

    // TEST: DtH copy: checking output rounded chessboard image
    cudaMemcpy( h_pixel.ptr(), d_pixel, total_pixels * sizeof(unsigned char),
        cudaMemcpyDeviceToHost);
    cv::imwrite("2_rounded_chessboard.png", h_pixel);

    invert_color_kernel<<<blocks_covering_pixels, threads>>>(d_pixel, image_width);

    // TEST: DtH copy: checking output inverted chessboard image
    cudaMemcpy( h_pixel.ptr(), d_pixel, total_pixels * sizeof(unsigned char),
        cudaMemcpyDeviceToHost);
    cv::imwrite("3_inverted_chessboard.png", h_pixel);

    // Checking output array without creating the image avoiding OpenCV functions
    test_chessboard_inverted_colors(image_width, block_width, d_pixel);

    cudaFree(d_pixel);
    cudaFree(d_block_average);

    return 0;
}

