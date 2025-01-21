#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <omp.h>

#define INF ((1 << 30) - 1)
#define DEV_NO 0
#define B 64  // Block size

int original_n, m;
int *Dist;
cudaDeviceProp prop;

// Phase 1 Kernel with 2x2 tiling
__global__ void phase1_optimized(int *Dist, int Round, int n) {
    __shared__ int sharedDist[B][B];

    // Each thread handles a 2x2 block
    int tx = threadIdx.x;  // 0 to (B/2 - 1)
    int ty = threadIdx.y;  // 0 to (B/2 - 1)

    int x = tx * 2;
    int y = ty * 2;

    int i_base = Round * B;
    int j_base = Round * B;

    int i = i_base + y;
    int j = j_base + x;

    // Load 2x2 elements into shared memory
    #pragma unroll
    for (int dy = 0; dy < 2; ++dy) {
        #pragma unroll
        for (int dx = 0; dx < 2; ++dx) {
            sharedDist[y + dy][x + dx] = Dist[(i + dy) * n + (j + dx)];
        }
    }
    __syncthreads();

    // Compute Phase 1
    for (int k = 0; k < B; ++k) {
        #pragma unroll
        for (int dy = 0; dy < 2; ++dy) {
            #pragma unroll
            for (int dx = 0; dx < 2; ++dx) {
                int y_idx = y + dy;
                int x_idx = x + dx;
                int new_dist = sharedDist[y_idx][k] + sharedDist[k][x_idx];
                if (new_dist < sharedDist[y_idx][x_idx]) {
                    sharedDist[y_idx][x_idx] = new_dist;
                }
            }
        }
        __syncthreads();
    }

    // Write back to global memory
    #pragma unroll
    for (int dy = 0; dy < 2; ++dy) {
        #pragma unroll
        for (int dx = 0; dx < 2; ++dx) {
            Dist[(i + dy) * n + (j + dx)] = sharedDist[y + dy][x + dx];
        }
    }
}

// Phase 2 Kernel with 2x2 tiling
__global__ void phase2_optimized(int *Dist, int Round, int n) {
    __shared__ int sharedPivot[B][B];
    __shared__ int sharedBlock[B][B];

    int tx = threadIdx.x;  // 0 to (B/2 - 1)
    int ty = threadIdx.y;  // 0 to (B/2 - 1)

    int x = tx * 2;
    int y = ty * 2;

    int i, j;

    if (blockIdx.y == 0) { // Pivot Row
        if (blockIdx.x == Round) return;

        int i_base = Round * B;
        int j_base = blockIdx.x * B;

        i = i_base + y;
        j = j_base + x;

        // Load sharedPivot and sharedBlock
        for (int dy = 0; dy < 2; ++dy) {
            for (int dx = 0; dx < 2; ++dx) {
                sharedPivot[y + dy][x + dx] = Dist[(i + dy) * n + (Round * B + x + dx)];
                sharedBlock[y + dy][x + dx] = Dist[(i + dy) * n + (j + dx)];
            }
        }
    } else { // Pivot Column
        if (blockIdx.x == Round) return;

        int i_base = blockIdx.x * B;
        int j_base = Round * B;

        i = i_base + y;
        j = j_base + x;

        // Load sharedPivot and sharedBlock
        for (int dy = 0; dy < 2; ++dy) {
            for (int dx = 0; dx < 2; ++dx) {
                sharedPivot[y + dy][x + dx] = Dist[(Round * B + y + dy) * n + (j + dx)];
                sharedBlock[y + dy][x + dx] = Dist[(i + dy) * n + (j + dx)];
            }
        }
    }

    __syncthreads();

    // Compute Phase 2
    for (int k = 0; k < B; ++k) {
        for (int dy = 0; dy < 2; ++dy) {
            for (int dx = 0; dx < 2; ++dx) {
                int y_idx = y + dy;
                int x_idx = x + dx;

                int new_dist;
                if (blockIdx.y == 0) {
                    new_dist = sharedPivot[y_idx][k] + sharedBlock[k][x_idx];
                } else {
                    new_dist = sharedBlock[y_idx][k] + sharedPivot[k][x_idx];
                }
                if (new_dist < sharedBlock[y_idx][x_idx]) {
                    sharedBlock[y_idx][x_idx] = new_dist;
                }
            }
        }
        __syncthreads();
    }

    // Write back to global memory
    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            Dist[(i + dy) * n + (j + dx)] = sharedBlock[y + dy][x + dx];
        }
    }
}

// Phase 3 Kernel with 2x2 tiling
__global__ void phase3_optimized(int *Dist, int Round, int n, int start_y) {
    if (blockIdx.x == Round || blockIdx.y + start_y == Round) return;

    __shared__ int sharedRow[B][B];
    __shared__ int sharedCol[B][B];

    int tx = threadIdx.x;  // 0 to (B/2 - 1)
    int ty = threadIdx.y;  // 0 to (B/2 - 1)

    int x = tx * 2;
    int y = ty * 2;

    int i_base = (blockIdx.y + start_y) * B;
    int j_base = blockIdx.x * B;

    int i = i_base + y;
    int j = j_base + x;

    // Load sharedRow and sharedCol
    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            sharedRow[y + dy][x + dx] = Dist[(i + dy) * n + (Round * B + x + dx)];
            sharedCol[y + dy][x + dx] = Dist[(Round * B + y + dy) * n + (j + dx)];
        }
    }
    __syncthreads();

    // Load current distances
    int current[2][2];
    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            current[dy][dx] = Dist[(i + dy) * n + (j + dx)];
        }
    }

    // Compute Phase 3
    for (int k = 0; k < B; ++k) {
        for (int dy = 0; dy < 2; ++dy) {
            for (int dx = 0; dx < 2; ++dx) {
                int y_idx = y + dy;
                int x_idx = x + dx;

                int new_dist = sharedRow[y_idx][k] + sharedCol[k][x_idx];
                if (new_dist < current[dy][dx]) {
                    current[dy][dx] = new_dist;
                }
            }
        }
    }

    // Write back to global memory
    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            Dist[(i + dy) * n + (j + dx)] = current[dy][dx];
        }
    }
}

// Input function with padding
void input(char *infile, int *padded_n) {
    FILE *file = fopen(infile, "rb");
    fread(&original_n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    // Calculate padded_n as the smallest multiple of B >= original_n
    *padded_n = ((original_n + B - 1) / B) * B;

    Dist = (int *)malloc((*padded_n) * (*padded_n) * sizeof(int));

    // Initialize the entire matrix to INF
    for (int i = 0; i < (*padded_n) * (*padded_n); ++i)
        Dist[i] = INF;

    // Set Dist[i][i] = 0 for original nodes
    for (int i = 0; i < original_n; ++i)
        Dist[i * (*padded_n) + i] = 0;

    // Read m edges and set the corresponding distances
    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        int a = pair[0], b = pair[1], w = pair[2];
        Dist[a * (*padded_n) + b] = w;
    }
    fclose(file);
}

// Output function writing only original_n x original_n
void output_result(char *outFileName, int padded_n) {
    FILE *outfile = fopen(outFileName, "wb");
    for (int i = 0; i < original_n; ++i)
        fwrite(&Dist[i * padded_n], sizeof(int), original_n, outfile);
    fclose(outfile);
}

int main(int argc, char *argv[]) {
    int padded_n;
    input(argv[1], &padded_n);
    int n = padded_n;  // Update local n to padded_n

    int device_count;
    cudaGetDeviceCount(&device_count);  // Get the number of CUDA devices

    int rounds = n / B;

    // Limit the number of devices to the number of rounds
    int active_device_count = device_count;
    if (rounds < device_count) {
        active_device_count = rounds;
    }

    // Allocate array of device pointers for active devices
    int **d_Dist = (int **)malloc(active_device_count * sizeof(int*));

    // Adjust the number of OpenMP threads to the active number of devices
    omp_set_num_threads(active_device_count);

    // Define thread and grid dimensions
    dim3 threads(B / 2, B / 2);  // Each thread handles a 2x2 block
    dim3 grid_phase1(1, 1);
    dim3 grid_phase2(rounds, 2);
    dim3 grid_phase3(rounds, rounds);

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        cudaSetDevice(thread_id);

        // Allocate device memory for each GPU
        cudaMalloc((void**)&d_Dist[thread_id], n * n * sizeof(int));

        // Copy the entire Dist matrix to each GPU
        cudaMemcpy(d_Dist[thread_id], Dist, n * n * sizeof(int), cudaMemcpyHostToDevice);

        for (int r = 0; r < rounds; ++r) {
            // Phase 1 - Only one device needs to perform this
            if (thread_id == 0) {
                phase1_optimized<<<grid_phase1, threads>>>(d_Dist[thread_id], r, n);
                cudaDeviceSynchronize();
            }
            #pragma omp barrier  // Ensure phase 1 is completed before phase 2

            // Phase 2
            phase2_optimized<<<grid_phase2, threads>>>(d_Dist[thread_id], r, n);

            // Phase 3
            phase3_optimized<<<grid_phase3, threads>>>(d_Dist[thread_id], r, n, 0);
        }

        // Synchronize before copying data back
        cudaDeviceSynchronize();

        // Copy results back to host from device 0
        if (thread_id == 0) {
            cudaMemcpy(Dist, d_Dist[thread_id], n * n * sizeof(int), cudaMemcpyDeviceToHost);
        }

        // Free device memory
        cudaFree(d_Dist[thread_id]);
    }

    // Write the result to the output file
    output_result(argv[2], n);

    // Free host memory
    free(Dist);
    free(d_Dist);

    return EXIT_SUCCESS;
}
