#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define BR 8         // 每個 block 負責的 row 數量(區塊尺寸)
#define BC 8         // 每個 block 負責的 column 數量(區塊尺寸)
#define BK 64         // K 和 V 的 tiling 大小(一次處理的維度片段)
#define THREAD_ROWS 2 // 每個 thread 處理的 row 數量
#define THREAD_COLS 2 // 每個 thread 處理的 column 數量

int batchSize, seqLen, dim;
float *h_Q, *h_K, *h_V, *h_O;

// 從檔案中讀取 B, N, d (batchSize, seqLen, dim)，以及 Q, K, V 資料
void input(char *input_filename) {
    FILE *file = fopen(input_filename, "rb");
    fread(&batchSize, sizeof(int), 1, file);
    fread(&seqLen, sizeof(int), 1, file);
    fread(&dim, sizeof(int), 1, file);

    h_Q = (float *)malloc(batchSize * seqLen * dim * sizeof(float));
    h_K = (float *)malloc(batchSize * seqLen * dim * sizeof(float));
    h_V = (float *)malloc(batchSize * seqLen * dim * sizeof(float));
    h_O = (float *)malloc(batchSize * seqLen * dim * sizeof(float));

    for (int b = 0; b < batchSize; b++) {
        fread(h_Q + b * seqLen * dim, sizeof(float), seqLen * dim, file);
        fread(h_K + b * seqLen * dim, sizeof(float), seqLen * dim, file);
        fread(h_V + b * seqLen * dim, sizeof(float), seqLen * dim, file);
    }
    memset(h_O, 0, batchSize * seqLen * dim * sizeof(float));
    fclose(file);

    // printf("B: %d, N: %d, d: %d\n", batchSize, seqLen, dim);
}

// 輸出結果 O 到檔案中
void output(char *output_filename) {
    FILE *file = fopen(output_filename, "wb");
    fwrite(h_O, sizeof(float), batchSize * seqLen * dim, file);
    fclose(file);

    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_O);
}

// flash_attention_kernel：執行 flash attention 的核心計算
// 輸入: Q, K, V 張量以及相關維度
// 計算過程：
// 1. 將 Q、K、V 分塊(tiling)載入 shared memory。
// 2. 計算注意力分數 score = Q * K^T / sqrt(dim)。
// 3. 使用 softmax 將 score 轉為注意力權重。
// 4. 使用這些權重對 V 做加權求和，得到輸出 O。
__global__ void flash_attention_kernel(
    float *d_Q, float *d_K, float *d_V, float *d_O,
    int batchSize, int seqLen, int dim
) {
    int tidX = threadIdx.x;
    int tidY = threadIdx.y;
    int threadIdFlat = tidY * blockDim.x + tidX;

    // blockIdx.x: batch的索引, blockIdx.y: sequence中分塊的索引
    int batchIdx = blockIdx.x;
    int blockRow = blockIdx.y;
    int batchOffset = batchIdx * seqLen * dim;

    // 使用 shared memory 儲存分塊後的 Q, K, V, 以及中間的分數計算結果
    __shared__ float qTile[BR][BK + 1];
    __shared__ float kTile[BC][BK + 1]; 
    __shared__ float vTile[BC][BK];
    __shared__ float scoreTile[BR][BC + 1];

    // numKVTiles: 我們將 d 維度分成若干 BK 大小的片段 (tile)
    // int numKVTiles = (dim + BK - 1) / BK;
    // 每個 tile 有多少個 thread 來處理 (由 BR、BC 和 THREAD_ROWS、THREAD_COLS 決定)
    int numThreadsPerTile = BR * BC / (THREAD_ROWS * THREAD_COLS);

    // 計算該 thread 對應到 tile 中的 row、col 索引
    int tileColIdx = threadIdFlat % (BC / THREAD_COLS);
    int tileRowIdx = threadIdFlat / (BC / THREAD_COLS);

    // rowSumExp: 記錄 softmax 分母 (sum of exponentials)
    // rowMax: 用於儲存目前 row 的最大值 (softmax 稍後要用來做數值穩定性)
    // prevRowMax: 儲存更新前的 max 值
    // outAcc: 用於累積加權後的 V 值 (最後除上 rowSumExp 得到 O)
    float rowSumExp[THREAD_ROWS];  
    float rowMax[THREAD_ROWS];  
    float prevRowMax[THREAD_ROWS];
    float outAcc[THREAD_ROWS][BK];

    // 初始化這些陣列
    #pragma unroll
    for (int i = 0; i < THREAD_ROWS; i++) {
        rowSumExp[i] = 0.0f;
        rowMax[i] = -INFINITY;
        #pragma unroll
        for (int j = 0; j < BK; j++) {
            outAcc[i][j] = 0.0f;
        }
    }

    // Step1 : 載入 Q 的分塊到 shared memory (qTile)
    // qStart: 該 batch 的起始位置 + blockRow 對應的 Q 區段起點
    int qStart = batchOffset + blockRow * BR * dim;
    // 所有 BR * BK 的元素都被分攤到 numThreadsPerTile 個 thread 處理。
    #pragma unroll
    for (int i = threadIdFlat; i < BR * BK; i += numThreadsPerTile) {
        int row = i / BK; // row within the block
        int col = i % BK; // col within the tile segment
        // int globalRow = blockRow * BR + row;
        // 若超出範圍，則填 0
        if (row < BR && col < dim) {
            qTile[row][col] = d_Q[qStart + row * dim + col];
        } else {
            qTile[row][col] = 0.0f;
        }
    }

    __syncthreads();

    // colTiles: 我們在 sequence length (N) 方向上也分成多個 BC 大小的 tile
    // 到目前為止我們已經拿到要計算的 qTile 了，所以接著要拿 qTile 去跟每一個對應的 kTile 還有 vTile 做運算
    int colTiles = (seqLen + BC - 1) / BC;
    #pragma unroll
    for (int tileIdx = 0; tileIdx < colTiles; tileIdx++) {
        // kStart: 對應該 tile 的 K, V 起始位置
        int kStart = batchOffset + tileIdx * BC * dim;
        // 載入 K、V 的分塊到 shared memory
        // Step2 : 載入 K, V  的分塊到 shared memory (kTile, vTile)
        #pragma unroll
        for (int i = threadIdFlat; i < BC * BK; i += numThreadsPerTile) {
            int row = i / BK;
            int col = i % BK;
            int globalCol = tileIdx * BC + row;
            if (row < BC && col < dim && globalCol < seqLen) {
                kTile[row][col] = d_K[kStart + row * dim + col];
                vTile[row][col] = d_V[kStart + row * dim + col];
            } else {
                kTile[row][col] = 0.0f;
                vTile[row][col] = 0.0f;
            }
        }

        __syncthreads();

        // threadResults: 暫存本 thread 計算的 (Q_i * K_j^T) 部分結果
        float threadResults[THREAD_ROWS * THREAD_COLS] = {0.0f};

        // Setp3 : 計算 Q * K^T, 計算當前 thread 負責的部分，並且因為矩陣相乘涉及共同維度 d 的逐元素相乘與加總，
        // 所以使用 threadResults 來累積每一輪的乘積結果。

        // 將 dim 分成 numKVTiles 個 BK 大小的區段
        // #pragma unroll
        // for (int t = 0; t < numKVTiles; t++) {
        // int t = 0;
        #pragma unroll
        for (int d = 0; d < BK; d++) {
            float qVals[THREAD_ROWS];
            float kVals[THREAD_COLS];

            // 取出對應 row 的 Q 值
            for (int i = 0; i < THREAD_ROWS; i++) {
                int row = tileRowIdx * THREAD_ROWS + i;
                qVals[i] = qTile[row][d];
            }

            // 取出對應 col 的 K 值
            for (int j = 0; j < THREAD_COLS; j++) {
                int col = tileColIdx * THREAD_COLS + j;
                kVals[j] = kTile[col][d];
            }

            // 矩陣乘法累加
            for (int i = 0; i < THREAD_ROWS; i++) {
                for (int j = 0; j < THREAD_COLS; j++) {
                    threadResults[i * THREAD_COLS + j] += qVals[i] * kVals[j];
                }
            }
        }
        // }

        // Setp4 : 對分數進行 scaling: score = QK^T / sqrt(dim)
        float scaling = 1.0f / sqrtf((float)dim);
        #pragma unroll
        for (int i = 0; i < THREAD_ROWS; i++) {
            for (int j = 0; j < THREAD_COLS; j++) {
                threadResults[i * THREAD_COLS + j] *= scaling;
            }
        }

        // Setp5 : 將 threadResults 寫回 shared memory 的 scoreTile
        #pragma unroll
        for (int i = 0; i < THREAD_ROWS; i++) {
            int row = tileRowIdx * THREAD_ROWS + i;
            for (int j = 0; j < THREAD_COLS; j++) {
                int col = tileColIdx * THREAD_COLS + j;
                scoreTile[row][col] = threadResults[i * THREAD_COLS + j];
            }
        }

        __syncthreads();

        // Setp5 : softmax(1), 拿到 rowMax
        #pragma unroll
        for (int i = 0; i < THREAD_ROWS; i++) {
            int row = tileRowIdx * THREAD_ROWS + i;
            int globalRow = blockRow * BR + row;
            if (globalRow < seqLen) {
                prevRowMax[i] = rowMax[i];
                float currentMax = rowMax[i];
                for (int j = 0; j < BC; j++) {
                    int globalCol = tileIdx * BC + j;
                    if (globalCol < seqLen) {
                        currentMax = fmaxf(currentMax, scoreTile[row][j]);
                    }
                }
                rowMax[i] = currentMax;
            }
        }

        __syncthreads();

        // Setp6 : softmax(2), 計算 outAcc
        // 將之前的結果 O, l 與新 max 做 exponent scaling 調整
        // 以保持數值穩定度 (將上一輪的計算結果重新 scale)
        #pragma unroll
        for (int i = 0; i < THREAD_ROWS; i++) {
            int globalRow = blockRow * BR + tileRowIdx * THREAD_ROWS + i;
            if (globalRow < seqLen) {
                float expScale = expf(prevRowMax[i] - rowMax[i]);
                rowSumExp[i] *= expScale;
                for (int j = 0; j < BK; j++) {
                    outAcc[i][j] *= expScale;
                }
            }
        }

        // Setp7 : 得到最終的 rowSumExp
        // 計算本輪新加入的 score 區段的 exponent，加總到 rowSumExp 並更新 outAcc
        #pragma unroll
        for (int i = 0; i < THREAD_ROWS; i++) {
            int row = tileRowIdx * THREAD_ROWS + i;
            int globalRow = blockRow * BR + row;
            if (globalRow < seqLen) {
                for (int j = 0; j < BC; j++) {
                    int globalCol = tileIdx * BC + j;
                    if (globalCol < seqLen) {
                        float scoreVal = scoreTile[row][j];
                        float p = expf(scoreVal - rowMax[i]);
                        rowSumExp[i] += p;
                        // 使用 p 加權累加 V 的值
                        for (int k = 0; k < BK; k++) {
                            outAcc[i][k] += p * vTile[j][k];
                        }
                    }
                }
            }
        }

        __syncthreads();
    }

    // Step8
    // 完整 softmax 的條件：只有當所有分塊完成後，softmax 的結果 outAcc 才能最終計算完成。
    // 將最終結果 outAcc / rowSumExp 寫回 global memory 的 O
    int oStart = batchOffset + blockRow * BR * dim;
    #pragma unroll
    for (int i = 0; i < THREAD_ROWS; i++) {
        int row = tileRowIdx * THREAD_ROWS + i;
        int globalRow = blockRow * BR + row;
        if (globalRow < seqLen) {
            float invSumExp = 1.0f / rowSumExp[i];
            for (int k = 0; k < BK; k++) {
                if (k < dim) {
                    d_O[oStart + row * dim + k] = outAcc[i][k] * invSumExp;
                }
            }
        }
    }
}

void flash_attention(float *d_Q, float *d_K, float *d_V, float *d_O, int seqLen, int dim) {
    // 計算 block 維度: 在 row 路徑上 BR/THREAD_ROWS, 在 column 路徑上 BC/THREAD_COLS
    dim3 blockDim(BC / THREAD_COLS, BR / THREAD_ROWS);
    // 計算 grid 維度: batchSize 在 x 方向，sequence length 分成 (seqLen + BR - 1)/BR 塊
    dim3 gridDim(batchSize, (seqLen + BR - 1) / BR);
    flash_attention_kernel<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_O, batchSize, seqLen, dim);
    cudaDeviceSynchronize();
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_filename> <output_filename>\n", argv[0]);
        return 1;
    }

    // 載入輸入資料
    input(argv[1]);

    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc((void **)&d_Q, batchSize * seqLen * dim * sizeof(float));
    cudaMalloc((void **)&d_K, batchSize * seqLen * dim * sizeof(float));
    cudaMalloc((void **)&d_V, batchSize * seqLen * dim * sizeof(float));
    cudaMalloc((void **)&d_O, batchSize * seqLen * dim * sizeof(float));

    // 將 Q, K, V 資料從 host 傳到 device
    cudaMemcpy(d_Q, h_Q, batchSize * seqLen * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, batchSize * seqLen * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, batchSize * seqLen * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_O, 0, batchSize * seqLen * dim * sizeof(float));

    // double start = getTimeStamp();
    // 呼叫 flash_attention 核心計算
    flash_attention(d_Q, d_K, d_V, d_O, seqLen, dim);
    // double end = getTimeStamp();

    // printf("(B, N, d): (%d, %d, %d)\n", batchSize, seqLen, dim);
    // printf("Time: %.3f seconds\n", end - start);

    // 將結果 O 從 device 拷貝回 host
    cudaMemcpy(h_O, d_O, batchSize * seqLen * dim * sizeof(float), cudaMemcpyDeviceToHost);
    // 輸出結果
    output(argv[2]);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);

    return 0;
}


