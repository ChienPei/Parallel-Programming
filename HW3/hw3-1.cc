#include <stdio.h>
#include <stdlib.h>
#include <sched.h>
#include <omp.h>
#include <emmintrin.h> 
#define CHUNK_SIZE 10

const int INF = ((1 << 30) - 1);
const int V = 50010;
void input(char* inFileName);
void output(char* outFileName);

void block_FW(int B);
int ceil(int a, int b);
void cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height);

int n, m, NUM_THREADS;
static int Dist[V][V];

int main(int argc, char* argv[]) {
    input(argv[1]);
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    NUM_THREADS = CPU_COUNT(&cpu_set);
    int B = 512;
    block_FW(B);
    output(argv[2]);
    return 0;
}

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    printf("n: %d, m: %d\n", n, m);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                Dist[i][j] = 0;
            } else {
                Dist[i][j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]][pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (Dist[i][j] >= INF) Dist[i][j] = INF;
        }
        fwrite(Dist[i], sizeof(int), n, outfile);
    }
    fclose(outfile);
}

int ceil(int a, int b) { return (a + b - 1) / b; }

void block_FW(int B) {
    int round = ceil(n, B);
    for (int r = 0; r < round; ++r) {
        printf("%d %d\n", r, round);
        fflush(stdout);
        
        /* Phase 1 */
        cal(B, r, r, r, 1, 1);

        /* Phase 2 */
        #pragma omp parallel sections
        {
            #pragma omp section
            cal(B, r, r, 0, r, 1);  // Left
            #pragma omp section
            cal(B, r, r, r + 1, round - r - 1, 1);  // Right
            #pragma omp section
            cal(B, r, 0, r, 1, r);  // Top
            #pragma omp section
            cal(B, r, r + 1, r, 1, round - r - 1);  // Bottom
        }

        /* Phase 3 */
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int i = 0; i < round; ++i) {
            for (int j = 0; j < round; ++j) {
                if (i != r && j != r) {
                    cal(B, r, i, j, 1, 1);
                }
            }
        }
    }
}

#include <emmintrin.h>  // SSE2 header

// Define a helper function for min operation using SSE2
__m128i _mm_min_epi32_sse2(__m128i a, __m128i b) {
    __m128i mask = _mm_cmplt_epi32(a, b);  // Compare a and b
    return _mm_or_si128(_mm_and_si128(mask, a), _mm_andnot_si128(mask, b));  // Select min values
}

void cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height) {
    int block_end_x = block_start_x + block_height;
    int block_end_y = block_start_y + block_width;

    for (int b_i = block_start_x; b_i < block_end_x; ++b_i) {
        for (int b_j = block_start_y; b_j < block_end_y; ++b_j) {
            for (int k = Round * B; k < (Round + 1) * B && k < n; ++k) {
                int block_internal_start_x = b_i * B;
                int block_internal_end_x = (b_i + 1) * B;
                int block_internal_start_y = b_j * B;
                int block_internal_end_y = (b_j + 1) * B;

                if (block_internal_end_x > n) block_internal_end_x = n;
                if (block_internal_end_y > n) block_internal_end_y = n;

                #pragma omp parallel for schedule(dynamic, 4)
                for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
                    for (int j = block_internal_start_y; j < block_internal_end_y; j += 4) {
                        // Load Dist[i][j] and Dist[k][j] values into SSE2 registers
                        __m128i dist_ij = _mm_loadu_si128((__m128i*)&Dist[i][j]);
                        __m128i dist_ik = _mm_set1_epi32(Dist[i][k]);
                        __m128i dist_kj = _mm_loadu_si128((__m128i*)&Dist[k][j]);

                        // Compute dist_ik + dist_kj
                        __m128i sum_dist = _mm_add_epi32(dist_ik, dist_kj);

                        // Get the minimum of Dist[i][j] and sum_dist using the custom min function
                        __m128i new_dist = _mm_min_epi32_sse2(dist_ij, sum_dist);

                        // Store the result back to Dist[i][j]
                        _mm_storeu_si128((__m128i*)&Dist[i][j], new_dist);
                    }
                }
            }
        }
    }
}
