## HW 4 Report

# 1. Implementation
## 1.a Describe how you implemented the FlashAttention forward pass using CUDA.
**Mention the algorithm's key steps, such as matrix blocking, SRAM usage, and how intermediate results like scaling factors (ℓ and 𝑚) were calculated.**

**(1) Matrix Blocking**  
- 將 Q 分為 BR×BK 的區塊，每個 thread 負責部分載入資料。
計算公式及對應的程式碼如下：
![Attention Formula](https://github.com/ChienPei/Parallel-Programming-HW4/blob/main/pics/Implementation-math-1.png?raw=true)

```cpp
// 載入 Q 的分塊到 shared memory
int qStart = batchOffset + blockRow * BR * dim; 
for (int i = threadIdFlat; i < BR * BK; i += numThreadsPerTile) {
    int row = i / BK; // 確定在分塊內的 row 索引
    int col = i % BK; // 確定在分塊內的 column 索引
    if (row < BR && col < dim) {
        qTile[row][col] = d_Q[qStart + row * dim + col];
    } else {
        qTile[row][col] = 0.0f; // 超出範圍補零
    }
}
```

**(2) 計算 attention scores 並使用 scaling factors**
- Q 的每一 row 與 K 的每一 column 相乘，形成 attention scores 。
- 使用 scaling factors  避免 overflow，提升穩定性。
計算公式及對應的程式碼如下：
![Attention Formula](https://github.com/ChienPei/Parallel-Programming-HW4/blob/main/pics/Implementation-math-2.png?raw=true)

```cpp
// 計算 Q * K^T，並逐步累加
for (int d = 0; d < BK; d++) {
    ...
    for (int i = 0; i < THREAD_ROWS; i++) {
        for (int j = 0; j < THREAD_COLS; j++) {
            // 每個 thread 負責計算部分的 Q * K^T 
            threadResults[i * THREAD_COLS + j] += qVals[i] * kVals[j];
        }
    }
}
// 進行縮放
float scaling = 1.0f / sqrtf((float)dim);
for (int i = 0; i < THREAD_ROWS; i++) {
    for (int j = 0; j < THREAD_COLS; j++) {
        // 每個 thread 負責計算部分的 Q * K^T 後的 scaling
        threadResults[i * THREAD_COLS + j] *= scaling;
    }
}
```

**(3)  Softmax 計算 (避免分母為0)**
- Softmax 的計算容易因數值過大導致 overflow 的問題，透過減去最大值 (row-wise max) 解決數值不穩定性。具體的做法是：
- `rowMax[i]` 存目前 row 的最大值，避免指數運算中出現 overflow。
- `outAcc` 累積 softmax 分子，並乘上 V。
- `rowSumExp[i]` 累積 softmax 分母。
計算公式及對應的程式碼如下：
![Attention Formula](https://github.com/ChienPei/Parallel-Programming-HW4/blob/main/pics/Implementation-math-3.png?raw=true)

```cpp
#pragma unroll
for (int i = 0; i < THREAD_ROWS; i++) {
    int row = tileRowIdx * THREAD_ROWS + i;
    float currentMax = rowMax[i];
    for (int j = 0; j < BC; j++) {
        currentMax = fmaxf(currentMax, scoreTile[row][j]);  // 更新最大值 (row-wise max)
    }
    rowMax[i] = currentMax;
}
// 計算 Softmax
for (int i = 0; i < THREAD_ROWS; i++) {
    for (int j = 0; j < BC; j++) {
        float scoreVal = scoreTile[row][j];
        float p = expf(scoreVal - rowMax[i]); // 減去最大值
        rowSumExp[i] += p; // 累積計算 softamx 所需要的分母
        for (int k = 0; k < BK; k++) {
            outAcc[i][k] += p * vTile[j][k]; // 分子的部分先乘上對應的 V
        }
    }
}
```

**(4) 完成 softmax 並寫回全局記憶體 O**
- 將 `outAcc` 除以計算完的 `rowSumExp` 得到 softmax 後的 Q * K^T * V 的最終答案。
- 將最終結果寫回全域記憶體 O。
計算公式及對應的程式碼如下：
![Attention Formula](https://github.com/ChienPei/Parallel-Programming-HW4/blob/main/pics/Implementation-math-4.png?raw=true)

```cpp
// 將 outAcc 進行標準化並寫回全局記憶體
for (int i = 0; i < THREAD_ROWS; i++) {
    float invSumExp = 1.0f / rowSumExp[i];
    for (int k = 0; k < BK; k++) {
        if (k < dim) {
            // 除以上一部所計算出的分母（即完成 softmax 的計算）
            d_O[oStart + row * dim + k] = outAcc[i][k] * invSumExp; 
        }
    }
}
```

---

## 1.b Explain how matrices Q, K, and V are divided into blocks and processed in parallel.
- 我將 Q、K、V 分塊為 BR×BC 的區塊，使每個 CUDA block 能夠同時負責一部分的 row 與 column。這樣就可以在有限的 shared memory 空間中處理維度 d 的片段。
下圖是具體的分配方式的示意圖：
![Division](https://github.com/ChienPei/Parallel-Programming-HW4/blob/main/pics/Implementation-division-1.png?raw=true)

---

## 1.c Describe how you chose the block sizes B_r and B_c and why.
我選擇 `BR=BC=8`，這是做完實驗後發現最佳的配置。
當 `BK=64` 時，`qTile` 和 `kTile` 的記憶體需求分別為 `BR×BK` 和 `BC×BK`。若選擇 `BR=BC=32`，共享記憶體的壓力會大幅增加，可能導致 block 無法同時執行，降低平行度。此外，`BR=BC=8` 讓每個 CUDA block 處理較小範圍的資料，能分配給更多 block 同時執行，提升 GPU 的平行運算效能。相比之下，`BR=BC=32` 雖然減少了 block 總數，但每個 block 占用更多資源，可能導致 GPU 核心閒置，無法充分發揮硬體的能力。因此 `BR=BC=8` 的設定或許比較能在計算與記憶體訪問間取得最佳平衡，同時確保資源利用率最大化。

---

## 1.d Specify the configurations for CUDA kernel launches, such as the number of threads per block, shared memory allocation, and grid dimensions.
- Threads per block 和 grid dimensions 的配置
    - 每個 block 會分到 `(BC/THREAD_COLS)* (R/THREAD_ROWS)` 個 threads，因為每個 thread 會負責處理 `THREAD_COLS*THREAD_ROWS` 個元素
    - 每個 grid 會有 `(batchSize)*((seqLen+BR-1)/BR)` 個 blocks, 其中根據 BR 的大小來決定要將 seqLen 分成幾塊。

    ```CPP
    dim3 blockDim(BC / THREAD_COLS, BR / THREAD_ROWS);
    dim3 gridDim(batchSize, (seqLen + BR - 1) / BR);
    ```

- Shared memory allocation 的配置
    - 每個 Q, K, V 的 column 長度都是 BK ，負責處理每個維度的計算。
    - Q 一次會處理 BR 個 rows, 而 K, V 會處理 BC 個 rows。
    - 由於 O 存的是相乘後的結果，又因為 (BR×BK) * (BK×BC) = (BR×BC)，所以 O 的配置是 BR×BC。
    ```cpp
    // 使用 shared memory 儲存分塊後的 Q, K, V, 以及 O
    __shared__ float qTile[BR][BK + 1]; // Q
    __shared__ float kTile[BC][BK + 1];  // K
    __shared__ float vTile[BC][BK]; // V
    __shared__ float scoreTile[BR][BC + 1]; // O
    ```
---

## 1.e Justify your choices and how they relate to the blocking factors and the SRAM size.
我是根據實驗結果來選擇BR和BC的大小，而 `BK`，是為 dimension 最多是 64，所以設定成64。在實驗中，當分塊大小從 `BR=BC=16` 減小至 `BR=BC=8 `時，執行時間從 `4.475` 秒降至 `3.526` 秒，表示出較小的分塊設計能顯著提升效能。這可能是因為 `BR=BC=8` 的分塊大小能更有效地利用共享記憶體，在避免共享記憶體壓力的同時，允許更多 CUDA blocks 平行運行，進一步提升 GPU 資源的利用率。

此外，這樣的設計也可以減少了 thread 間的同步與資源競爭問題，提高整體效率。從下面的實驗結果中可以看到，除了 Block Size 調整外，其他技術（如 Tiling 和解決 Bank Conflict）也對效能有進一步優化作用，但 Block Size 的調整是提升效能的最關鍵因素之一，充分展示了這是不錯的分塊策略！

---

# 2. Profiling Results
下面分別為重點摘要的和詳細的輸出結果：

| Metric Name                 | Value           |
| -------------------------   | --------------- |
| **Achieved Occupancy**      | **12.5%**       |
| **SM Efficiency**           | **99.41%**      |
| **Shared Load Throughput**  | **1528.0 GB/s** |
| **Shared Store Throughput** | **13.956 GB/s** |
| **Global Load Throughput**  | **4.03 GB/s**   |
| **Global Store Throughput** | **81.002 MB/s** |

```bash
Profiling results for kernel: flash_attention_kernel
Profiling metric: achieved_occupancy
==3423229== NVPROF is profiling process 3423229, command: ./hw4 testcases/t28 t28.out
B: 4, N: 16384, d: 64
(B, N, d): (4, 16384, 64)
Time: 1.560 seconds
==3423229== Profiling application: ./hw4 testcases/t28 t28.out
==3423229== Profiling result:
==3423229== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1080 (0)"
    Kernel: flash_attention_kernel(float*, float*, float*, float*, int, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.125000    0.125000    0.125000
-------------------------------
Profiling metric: sm_efficiency
==505524== NVPROF is profiling process 505524, command: ./hw4 testcases/t28 t28.out
B: 4, N: 16384, d: 64
(B, N, d): (4, 16384, 64)
Time: 1.558 seconds
==505524== Profiling application: ./hw4 testcases/t28 t28.out
==505524== Profiling result:
==505524== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1080 (0)"
    Kernel: flash_attention_kernel(float*, float*, float*, float*, int, int, int)
          1                             sm_efficiency                   Multiprocessor Activity      99.41%      99.41%      99.41%
-------------------------------
Profiling metric: shared_load_throughput
==3423305== NVPROF is profiling process 3423305, command: ./hw4 testcases/t28 t28.out
B: 4, N: 16384, d: 64
(B, N, d): (4, 16384, 64)
Time: 1.558 seconds
==3423305== Profiling application: ./hw4 testcases/t28 t28.out
==3423305== Profiling result:
==3423305== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1080 (0)"
    Kernel: flash_attention_kernel(float*, float*, float*, float*, int, int, int)
          1                    shared_load_throughput             Shared Memory Load Throughput  1528.0GB/s  1528.0GB/s  1528.0GB/s
-------------------------------
Profiling metric: shared_store_throughput
==794295== NVPROF is profiling process 794295, command: ./hw4 testcases/t28 t28.out
B: 4, N: 16384, d: 64
(B, N, d): (4, 16384, 64)
Time: 1.585 seconds
==794295== Profiling application: ./hw4 testcases/t28 t28.out
==794295== Profiling result:
==794295== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1080 (0)"
    Kernel: flash_attention_kernel(float*, float*, float*, float*, int, int, int)
          1                   shared_store_throughput            Shared Memory Store Throughput  13.956GB/s  13.956GB/s  13.956GB/s
-------------------------------
Profiling metric: gld_throughput
==3423343== NVPROF is profiling process 3423343, command: ./hw4 testcases/t28 t28.out
B: 4, N: 16384, d: 64
(B, N, d): (4, 16384, 64)
Time: 3.991 seconds
==3423343== Profiling application: ./hw4 testcases/t28 t28.out
==3423343== Profiling result:
==3423343== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1080 (0)"
    Kernel: flash_attention_kernel(float*, float*, float*, float*, int, int, int)
          1                            gld_throughput                    Global Load Throughput  4.0298GB/s  4.0298GB/s  4.0298GB/s
-------------------------------
Profiling metric: gst_throughput
==1576528== NVPROF is profiling process 1576528, command: ./hw4 testcases/t28 t28.out
B: 4, N: 16384, d: 64
(B, N, d): (4, 16384, 64)
Time: 1.590 seconds
==1576528== Profiling application: ./hw4 testcases/t28 t28.out
==1576528== Profiling result:
==1576528== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1080 (0)"
    Kernel: flash_attention_kernel(float*, float*, float*, float*, int, int, int)
          1                            gst_throughput                   Global Store Throughput  81.002MB/s  81.002MB/s  81.002MB/s
-------------------------------

```
---

# 3. Experiment & Analysis
## 3.a System Spec 
- 使用的是 Apollo-GPU server

---

## 3.b Optimization
我使用 `t29` 這筆測資來做實驗，因為他是 hw4-judge 的所有測資中花最多時間執行的。
| Optimization Technique | Execution Time (s) |
| ---                    | ---                |
| GPU Baseline           | 4.475 seconds      |
| Block Size 16→8        | 3.526 seconds      |  
| Tiling                 | 3.376 seconds      |
| Bank Conflict          | 2.961 seconds      |
| Unroll                 | 2.897 seconds      |

![Optimization](https://github.com/ChienPei/Parallel-Programming-HW4/blob/main/pics/Optimization.png?raw=true)

我使用了縮小 block size、tiling、unroll、coalesced memory access、shared memory、handle bank conflict 和 cuda 2d alignment 這幾個優化方法。圖中列出 block size、tiling、unroll，可以發現確實有優化效果。這些優化技術共同將執行時間從 baseline 的 4.475 秒縮短至最佳的 2.897 秒，整體效能提升約 35%。

---


# 4. Experience & conclusion
## 4.a What have you learned from this homework?
在這次作業中，我學到了以下幾點：
1. 深入理解 FlashAttention 運作：
了解了矩陣分塊、Softmax 數值穩定性處理，以及共享記憶體在提升效能中的重要角色。
2. 數值穩定性與效能提升：
使用 row-wise 最大值解決 Softmax 運算中的 overflow 問題，並透過 tiling、unroll 等技術進一步優化運算效率。
3. CUDA 程式設計實作：
學習到如何選擇合適的 block size，搭配共享記憶體最佳化，達成 GPU 資源最大化利用。
4. 結論：
這次作業讓我對 CUDA 的設計與效能優化有更深入的理解，透過實作與分析有效提升 GPU 的平行計算能力，這對未來處理高效能運算有很大的幫助！