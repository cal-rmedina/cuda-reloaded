#include <cuda_runtime.h>
#include <stdio.h>

void filling_square_image_with_row_ascending_block_values(
    const unsigned int block_width,
    const unsigned int image_width,
    unsigned char* pixel)
{
    for (unsigned int col = 0; col < image_width; col++) {
        for (unsigned int row = 0; row < image_width; ++row) {
            const unsigned int px = col * image_width + row;
            pixel[px] = row % block_width;
        }
    }
}

__device__ __forceinline__ unsigned int row_element_sum(
    const unsigned char* __restrict__ d_pixel,
    const unsigned int block_width,
    const unsigned int initial_row_pixel) {

    unsigned int sum = 0;
    for (unsigned int px = 0; px < block_width; px++) {
        sum += d_pixel[initial_row_pixel + px];
    }
    return sum;
}

__device__ __forceinline__ unsigned int sum_8_bits_int(const int val)
{
    int shifted_val = val;

    // Explicit sum expression
    // sum = (p0 & 0xFF) + ((p0 >> 8)  & 0xFF) + ((p0 >> 16) & 0xFF) + ((p0 >> 24) & 0xFF);

    // When 4 contiguous uchar (01 02 03 04) are read as an int, the int
    // containing the 4 values looks like: 0x04030201. To recover each byte we
    // shift 1 byte (8 bits) + bitwise AND to obtain and each original byte.
    // The bitwise AND with 0xFFu (11111111) keeps the byte we need.

    // Explicit loop to avoid last shifted_val
    unsigned int sum = shifted_val & 0xFFu; // byte 0
    shifted_val >>= 8;          

    sum += shifted_val & 0xFFu; // byte 1
    shifted_val >>= 8;          

    sum += shifted_val & 0xFFu; // byte 2
    shifted_val >>= 8;          

    sum += shifted_val & 0xFFu; // byte 3

    return sum;
}

__device__ __forceinline__ unsigned int load_sum_20_bits_int(
    const unsigned char* __restrict__ d_pixel,
    unsigned int initial_row_pixel) {

    // 4‑byte load (int)
    const auto p0 = *reinterpret_cast<const int*>(d_pixel + initial_row_pixel);     // 0‑3
    const auto p1 = *reinterpret_cast<const int*>(d_pixel + initial_row_pixel + 4); // 4‑7
    const auto p2 = *reinterpret_cast<const int*>(d_pixel + initial_row_pixel + 8); // 8‑11
    const auto p3 = *reinterpret_cast<const int*>(d_pixel + initial_row_pixel + 12); // 12-15
    const auto p4 = *reinterpret_cast<const int*>(d_pixel + initial_row_pixel + 16); // 16-19

    const unsigned int sum_0 = sum_8_bits_int(p0);
    const unsigned int sum_1 = sum_8_bits_int(p1);
    const unsigned int sum_2 = sum_8_bits_int(p2);
    const unsigned int sum_3 = sum_8_bits_int(p3);
    const unsigned int sum_4 = sum_8_bits_int(p4);
    return sum_0 + sum_1 + sum_2 + sum_3 + sum_4;
}

__device__ __forceinline__ unsigned int load_sum_20_bits_uchar4(
    const unsigned char* __restrict__ d_pixel,
    unsigned int initial_row_pixel) {

    // 4‑byte load (uchar4)
    const auto p0 = *reinterpret_cast<const uchar4*>(d_pixel + initial_row_pixel);      // 0-3
    const auto p1 = *reinterpret_cast<const uchar4*>(d_pixel + initial_row_pixel + 4);  // 4-7
    const auto p2 = *reinterpret_cast<const uchar4*>(d_pixel + initial_row_pixel + 8);  // 8-11
    const auto p3 = *reinterpret_cast<const uchar4*>(d_pixel + initial_row_pixel + 12); // 12-15
    const auto p4 = *reinterpret_cast<const uchar4*>(d_pixel + initial_row_pixel + 16); // 16-19

    unsigned int sum = 0;
    sum += p0.x + p0.y + p0.z + p0.w;
    sum += p1.x + p1.y + p1.z + p1.w;
    sum += p2.x + p2.y + p2.z + p2.w;
    sum += p3.x + p3.y + p3.z + p3.w;
    sum += p4.x + p4.y + p4.z + p4.w;

    return sum;
}

__device__ __forceinline__ void check_row_sum(const unsigned int sum,
    const unsigned int block_width) {

     // Gauss sum: 0 + 1 + 2 + ... + n-1 = n (n-1) / 2
     const unsigned int expected_sum = (block_width * (block_width - 1)) >> 1;

     if ( sum != expected_sum) {
         printf("Wrong value: %d, instead of %d\n", sum, expected_sum);
     }
}

__global__ void row_loop_per_thread_kernel(
    const unsigned int block_width,
    const unsigned int blocks_in_x,
    const unsigned int image_width,
    const unsigned char *d_pixel)
{
    if (threadIdx.x < block_width) {
        const unsigned int block_id = blockIdx.x;
        const unsigned int x = block_id % blocks_in_x;
        const unsigned int y = block_id / blocks_in_x;
        const unsigned int initial_row_pixel = (threadIdx.x + y) * 
                        image_width + x * block_width;

//        unsigned int sum = row_element_sum(d_pixel, block_width,
//            initial_row_pixel);

        unsigned int sum = load_sum_20_bits_int(d_pixel, initial_row_pixel);

        check_row_sum(sum, block_width);
    }
}

int main() {

    constexpr unsigned int block_width = 20;
    constexpr unsigned int blocks_in_x = 1 << 10;
    constexpr unsigned int total_blocks = blocks_in_x * blocks_in_x;

    constexpr unsigned int image_width = block_width * blocks_in_x;
    constexpr unsigned int total_pixels = image_width * image_width;

    auto* h_pixel = new unsigned char[total_pixels];

    filling_square_image_with_row_ascending_block_values( block_width,
        image_width, h_pixel);

    unsigned char *d_pixel;
    cudaMalloc(&d_pixel, total_pixels *sizeof(unsigned char));

    cudaMemcpy( d_pixel, h_pixel, total_pixels, cudaMemcpyHostToDevice);
   
    constexpr unsigned int threads = 1 << 5;
    row_loop_per_thread_kernel<<< total_blocks, threads>>>(
                                        block_width, blocks_in_x,
                                        image_width, d_pixel);
    cudaDeviceSynchronize();

    cudaFree(d_pixel);
    free(h_pixel);

    return 0;
}
