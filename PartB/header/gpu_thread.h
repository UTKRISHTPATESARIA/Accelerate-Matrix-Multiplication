#include <math.h>
#include <unistd.h>
#include <cuda.h>

#define TILE_DIM 32
#define TILE_DIM_FACTOR 5
#define BLOCK_ROWS 8
#define thread_block_factor 4 // 16 threads
#define max_threads_per_block (1 << thread_block_factor)

#define CUDA_SAFECALL(call)                                                 \
    {                                                                       \
        call;                                                               \
        cudaError err = cudaGetLastError();                                 \
        if (cudaSuccess != err) {                                           \
            fprintf(                                                        \
                stderr,                                                     \
                "Cuda error in function '%s' file '%s' in line %i : %s.\n", \
                #call, __FILE__, __LINE__, cudaGetErrorString(err));        \
            fflush(stderr);                                                 \
            _exit(EXIT_FAILURE);                                            \
        }                                                                   \
    }


__global__ void multiply_gpu_less_threads(int *A, int *B, int *C, int N, int max_threads, int shift_factor, int output_size)
{
    int row_value = (blockIdx.x << shift_factor);
    int col_value = (threadIdx.y << shift_factor);
    int output_value = blockIdx.x * max_threads + threadIdx.y;

    if (output_value < output_size)
    {
        int sum_value = 0;
        for (int i = 0; i < N; ++i)
        {
            sum_value += A[row_value + i] * B[col_value + i];
        }
        C[output_value] = sum_value;
    }
}

__global__ void multiply_gpu(int *A, int *B, int *C, int N, int tb_factor, int col_blocks, int shift_factor, int output_size)
{
    int row_value = blockIdx.x;
    int col_value = (blockIdx.y << (tb_factor << 1)) + (threadIdx.x << tb_factor) + threadIdx.y;
    int output_value = (row_value << (tb_factor << 1)) * col_blocks + col_value;
    row_value = (row_value << shift_factor);
    col_value = (col_value << shift_factor);
    if (output_value < output_size)
    {
        int sum_value = 0;
        for (int i = 0; i < N; ++i)
        {
            sum_value += A[row_value + i] * B[col_value + i];
        }
        // C[output_value] = sum_value;
        atomicAdd(&C[output_value], sum_value);
    }
}

__global__ void add_gpu(int N, int *A, int *C, int mt, int st)
{
    int output_row = ((blockIdx.x << mt) + (threadIdx.y)) << st;
    int first_row = (((blockIdx.x << mt) << 1) + (threadIdx.y << 1)) << st;
    int second_row = first_row + N;
    for (int i = 0; i < N; ++i)
    {
        C[output_row + i] = A[i + first_row] + A[i + second_row];
    }
}

__global__ void transpose_gpu(int *matTran, int *matIn, int N, int shift_factor)
{
    __shared__ int tile[TILE_DIM][TILE_DIM + 1];
    int i_n = (blockIdx.x << TILE_DIM_FACTOR) + threadIdx.x;
    int i_m = (blockIdx.y << TILE_DIM_FACTOR) + threadIdx.y;
    int i;
    for (i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        if (i_n < N && (i_m + i) < N)
        {
            tile[threadIdx.y + i][threadIdx.x] = matIn[((i_m + i) << shift_factor) + i_n];
        }
    }
    __syncthreads();

    i_n = (blockIdx.y << TILE_DIM_FACTOR) + threadIdx.x;
    i_m = (blockIdx.x << TILE_DIM_FACTOR) + threadIdx.y;

    for (i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        if (i_n < N && (i_m + i) < N)
        {
            matTran[((i_m + i) << shift_factor) + i_n] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
}

void transpose_cpu(int *new_matrix, int *old_matrix, int size)
{
    int diagnol_index, index2, length, shift_factor = (int)(log10(size) / log10(2)), index = 0;
    diagnol_index = (index << shift_factor) + index;
    while (index < size - 1)
    {
        length = size - index;
        index2 = index * (size + 1);
        for (int i = 1; i < length; ++i)
        {
            new_matrix[index2 + (i << shift_factor)] = old_matrix[i + index2];
            new_matrix[i + index2] = old_matrix[index2 + (i << shift_factor)];
        }
        new_matrix[diagnol_index] = old_matrix[diagnol_index];
        ++index;
        diagnol_index = (index << shift_factor) + index;
    }
    new_matrix[diagnol_index] = old_matrix[diagnol_index];
}

void gpuThread(int N, int *matA, int *matB, int *output)
{
    int shift_factor, shift_factor_minus_one, size_using_shift, half_size;
    int *A, *B, *BT, *C, *D, *E;
    half_size = (N >> 1); // N/2
    shift_factor = (int)(log10(N) / log10(2));
    shift_factor_minus_one = shift_factor - 1;
    size_using_shift = (sizeof(int) << (shift_factor << 1)); // N*N*sizeof(int)
    int max_threads = (1 << (thread_block_factor << 1));     // 16*16

    CUDA_SAFECALL(cudaMalloc((void **)&A, size_using_shift));
    CUDA_SAFECALL(cudaMalloc((void **)&B, size_using_shift));
    CUDA_SAFECALL(cudaMalloc((void **)&BT, size_using_shift));

    cudaMemcpy(A, matA, size_using_shift, cudaMemcpyHostToDevice);
    cudaMemcpy(B, matB, size_using_shift, cudaMemcpyHostToDevice);

    if (N > 32)
    {
        dim3 threads(TILE_DIM, BLOCK_ROWS, 1);
        dim3 grid(ceil(N >> TILE_DIM_FACTOR), ceil(N >> TILE_DIM_FACTOR), 1);
        transpose_gpu<<<grid, threads>>>(BT, B, N, shift_factor);
    }
    else
    {
        int *bT = new int[size_using_shift];
        transpose_cpu(bT, matB, N);
        cudaMemcpy(BT, bT, size_using_shift, cudaMemcpyHostToDevice);
    }

    cudaMalloc((void **)&C, size_using_shift >> 1);
    cudaMalloc((void **)&D, size_using_shift >> 1);
    cudaMalloc((void **)&E, size_using_shift >> 2);

    dim3 threadsPerBlock, blocksPerGrid;
    if (half_size <= max_threads)
    {
        threadsPerBlock.x = 1;
        threadsPerBlock.y = half_size;
        blocksPerGrid.x = 1;
        blocksPerGrid.y = 1;
        max_threads = shift_factor_minus_one;
    }
    else
    {
        threadsPerBlock.x = 1;
        threadsPerBlock.y = max_threads;
        blocksPerGrid.x = (half_size >> (thread_block_factor << 1));
        blocksPerGrid.y = 1;
        max_threads = (thread_block_factor << 1);
    }
    add_gpu<<<blocksPerGrid, threadsPerBlock>>>(N, A, C, max_threads, shift_factor);
    add_gpu<<<blocksPerGrid, threadsPerBlock>>>(N, BT, D, max_threads, shift_factor);

    max_threads = (1 << (thread_block_factor << 1)); // 16*16
    if (half_size < max_threads)
    {
        threadsPerBlock.x = 1;
        threadsPerBlock.y = half_size;
        blocksPerGrid.x = half_size;
        blocksPerGrid.y = 1;
        max_threads = half_size;
        multiply_gpu_less_threads<<<blocksPerGrid, threadsPerBlock>>>(C, D, E, N, max_threads, shift_factor, size_using_shift >> 2);
    }
    else
    {
        threadsPerBlock.x = max_threads_per_block;
        threadsPerBlock.y = max_threads_per_block;
        int blocks = (half_size >> (thread_block_factor << 1));
        blocksPerGrid.x = blocks;
        blocksPerGrid.y = 1;
        if (N > max_threads)
        {
            blocksPerGrid.x = half_size;
            blocksPerGrid.y = half_size >> (thread_block_factor << 1);
        }
        multiply_gpu<<<blocksPerGrid, threadsPerBlock>>>(C, D, E, N, thread_block_factor, blocksPerGrid.y, shift_factor, size_using_shift >> 2);
    }
    cudaMemcpy(output, E, size_using_shift >> 2, cudaMemcpyDeviceToHost);

    cudaFree(A);
    cudaFree(B);
    cudaFree(BT);
    cudaFree(C);
    cudaFree(D);
    cudaFree(E);
}
