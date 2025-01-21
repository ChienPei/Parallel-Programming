# PP HW 5 Report 

## 1. Overview
> In conjunction with the UCP architecture mentioned in the lecture, please read [ucp_hello_world.c](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/pp2024/examples/ucp_hello_world.c)

1. Identify how UCP Objects (`ucp_context`, `ucp_worker`, `ucp_ep`) interact through the API, including at least the following functions:
    - `ucp_init`
    - `ucp_worker_create`
    - `ucp_ep_create`

    - 1.1  `ucp_init`:  如下圖，`ucp_init` 負責初始化 `ucp_context`，並根據 `ucp_params`, 和 `config` 設定 UCP 所需的資源和參數。
    以下是`ucp_init`中 `ucp_context`, `ucp_params`, 和 `config` 的具體步驟： 

        **[example/ucp_hello_world.c]**
        [permalink](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/84e459e73df4f02aecd044c44e4584d88f4b9b0e/examples/ucp_hello_world.c#L563-L575)
            - `ucp_init` 主要是用來根據 `ucp_params` 和 `config` 初始化 `ucp_context`，建立 UCP 的運行環境。
            - `ucp_params.field_mask` 是用來指定要初始化哪些項目，像是功能設定（例如 `UCP_PARAM_FIELD_FEATURES`）。
            - `ucp_params.features` 則是設定啟用的功能，像是支援 `UCP_FEATURE_TAG` 或`UCP_FEATURE_WAKEUP`。
            - `request_size` 和 `request_init` 負責設定 request 的大小和初始化方式。
            - `ucp_context` 整合這些設定後，負責管理 UCP 的通訊資源，確保系統可以正常運行。
            ![1.1](https://github.com/ChienPei/Parallel-Programming/blob/main/HW5/pics/1.1.png?raw=true)

        **[src/ucp/api/ucp.h]**
        [permalink](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/84e459e73df4f02aecd044c44e4584d88f4b9b0e/src/ucp/api/ucp.h#L2057-L2063)
        - `ucp_init` 會呼叫 `ucp_init_version`
            permalink：
            ![1.2](https://github.com/ChienPei/Parallel-Programming/blob/main/HW5/pics/1.2.png?raw=true)

        **[src/ucp/core/ucp_context.c]**
        [permalink](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/84e459e73df4f02aecd044c44e4584d88f4b9b0e/src/ucp/core/ucp_context.c#L2123-L2212)
        - `ucp_init_version` 是用來初始化 `ucp_context` 的函數，具體執行步驟如下：
            1. 使用 `ucp_version_check` 來檢查 API 的版本，確保提供的 API 主版號和次版號符合要求。
            2. 如果 `config` 為空，則呼叫 `ucp_config_read` 生成一個預設配置；否則直接使用用戶提供的配置。
            3. 呼叫 `ucs_calloc` 為 `ucp_context` 分配記憶體空間，並初始化`ucp_context`結構中的成員，例如 `cached_key_list` 和 `mt_lock`。
            4. 使用 `ucp_fill_config` 和 `ucp_fill_resources`，根據 `params` 和 `config` 設置`ucp_context` 所需的功能（例如 `UCP_FEATURE_TAG` 和 `UCP_FEATURE_WAKEUP`）、資源以及傳輸層。
            5. 如果配置中啟用了 `rcache`，則初始化記憶體 registration cache，否則將其設置為空。
            6. 為上下文生成一個  **UUID**，用來追蹤資源。
            ![1.3](https://github.com/ChienPei/Parallel-Programming/blob/main/HW5/pics/1.3.png?raw=true)

    - 1.2 `ucp_worker_create` : 如下圖，`ucp_worker_create` 是用來建立一個 UCP worker 的函數，負責初始化並設置 UCP 的核心工作單元，將 `ucp_context`、`ucp_params` 和 `ucp_worker` 整合起來，以下是具體步驟：

        **[example/ucp_hello_world.c]**
        [permalink](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/84e459e73df4f02aecd044c44e4584d88f4b9b0e/examples/ucp_hello_world.c#L584-L588)
            1. 使用 `worker_params.field_mask` 指定初始化哪些參數，此處設定為多執行緒模式 (`UCS_THREAD_MODE_SINGLE`)，表示不需要額外的鎖定機制（mutex）。
            2. `UCP_WORKER_PARAM_FIELD_THREAD_MODE` 是用來告知 UCP 初始化 Worker 時需要設定 `thread_mode` 的 flag。
            3. 呼叫 `ucp_worker_create` 函數，根據 `ucp_context` 和 `worker_params` 初始化 Worker，並將結果存入 `ucp_worker` 指標。
            ![1.4](https://github.com/ChienPei/Parallel-Programming/blob/main/HW5/pics/1.4.png?raw=true)

        **[src/ucp/api/ucp.h]**
        [permalink](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/84e459e73df4f02aecd044c44e4584d88f4b9b0e/src/ucp/api/ucp.h#L2140-L2142)
            ![1.5](https://github.com/ChienPei/Parallel-Programming/blob/main/HW5/pics/1.5.png?raw=true)

        **[src/ucp/core/ucp_worker.c]**
        [permalink](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/84e459e73df4f02aecd044c44e4584d88f4b9b0e/src/ucp/core/ucp_worker.c#L2311-L2547)
        - `ucp_worker_create` 是 UCP 用於創建 Worker 的核心函數，負責初始化一個 UCP Worker，為後續通訊和資源管理提供基礎，具體執行步驟如下：
            1. 使用 `ucs_calloc` 分配 Worker 的記憶體空間，若分配失敗則返回錯誤 `UCS_ERR_NO_MEMORY`。
            2. 將 `context` 綁定到 Worker（`worker->context = context`），生成 UUID 作為 Worker 的識別碼，並初始化計數器（例如 `flush_ops_count`、`inprogress`）和通訊結構（例如 `rkey_config_count`、`am_message_id`）。
            3. 初始化內部數據，包括`rkey_ptr_reqs`、 `internal_eps`、 `stream_ready_eps`  、`rkey_config_hash`和`discard_uct_ep_hash` 等。
            4. 呼叫 `ucp_worker_keepalive_reset`，初始化 Worker 的`keepalive` 機制，確保 Worker 能夠正常運行。
            5. 成功初始化後，將指向 Worker 的指標存入 `worker_p`，以供後續使用。
            ![1.6](https://github.com/ChienPei/Parallel-Programming/blob/main/HW5/pics/1.6.png?raw=true)

    - 1.3  `ucp_ep_create`：如下圖，`ucp_ep_create` 的功能是建立通訊端點（endpoint），並使用這個端點來實現資料傳輸。具體而言，它依賴 `ucp_worker` 作為 local 的通訊資源，並透過指定遠端的地址（`peer_addr`）來與另一個 `ucp_worker` 連接。此外，可以透過 `ep_params` 設定錯誤處理模式。而創建成功的 endpoint 將存儲在輸出的指標 `ep_p` 中。
        
        在 `ucp_hello_world.c` 中，`ucp_ep_create` 的作用分別為：
        
        - **在 client 端**：創建 `client_ep`，用於連接伺服器（server）。
        - **在 server 端**：創建 `server_ep`，用於連接客戶端（client）
        
        以下是 `ucp_ep_crete` 中 `ucp_worker`, `ucp_params`, 和 `server_ep` 的具體步驟：
        
        **[example/ucp_hello_world.c]** 
        [permalink](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/84e459e73df4f02aecd044c44e4584d88f4b9b0e/examples/ucp_hello_world.c#L443-L458)
        [permalink](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/84e459e73df4f02aecd044c44e4584d88f4b9b0e/examples/ucp_hello_world.c#L240-L252)
            - 設定 `ep_params` 配置，使用 `field_mask` 指定有效的參數（如 `REMOTE_ADDRESS`、`ERR_HANDLING_MODE`、`ERR_HANDLER` 等）。設定通訊目標`address`、`err_mode` 以及 `err_handler.cb`。
            - 呼叫 `ucp_ep_create` ，將 `ucp_worker` 與`ep_params`結合，創建一個 Endpoint，並將結果存入對應的 Endpoint 指標（例如 `client_ep` 或 `server_ep`）。
            ![1.7](https://github.com/ChienPei/Parallel-Programming/blob/main/HW5/pics/1.7.png?raw=true)
            ![1.8](https://github.com/ChienPei/Parallel-Programming/blob/main/HW5/pics/1.8.png?raw=true)

        **[src/ucp/api/ucp.h]**
        [permalink](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/84e459e73df4f02aecd044c44e4584d88f4b9b0e/src/ucp/api/ucp.h#L2575-L2576)
            ![1.9](https://github.com/ChienPei/Parallel-Programming/blob/main/HW5/pics/1.9.png?raw=true)

        **[src/ucp/core/ucp_ep.c]**
        [permalink](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/84e459e73df4f02aecd044c44e4584d88f4b9b0e/src/ucp/core/ucp_ep.c#L1176-L1216)
            - `ucp_ep_create` 是 UCP 中用於創建 Endpoint 的核心函數，Endpoint 是 UCP 通訊模型中負責處理與遠端通訊的 entity。`ucp_ep_create` 會根據輸入參數進行設定，其中 `worker` 是提供通訊資源的 context，負責管理整個 Endpoint 的基礎操作；`params` 包含創建 Endpoint 的詳細設定資訊，包括 `flags`（指定特殊模式，例如 Client-Server 模式）和 `field_mask`（指定有效字段，例如 `REMOTE_ADDRESS`、`ERR_HANDLING_MODE` 或 `NAME`），以及其他選填屬性如遠端 `address` 、`err_mode`和 `name`；最後，創建成功的 Endpoint 將存儲在輸出的指標 `ep_p` 中，供後續的通訊操作使用。
            ![1.10](https://github.com/ChienPei/Parallel-Programming/blob/main/HW5/pics/1.10.png?raw=true)


2. UCX abstracts communication into three layers as below. Please provide a diagram illustrating the architectural design of UCX.
    - `ucp_context`
    - `ucp_worker`
    - `ucp_ep`
    > Please provide detailed example information in the diagram corresponding to the execution of the command `srun -N 2 ./send_recv.out` or `mpiucx --host HostA:1,HostB:1 ./send_recv.out`

    - 2.1 ucp_context
        - `ucp_context_t` 是 UCX 的 global context，負責管理整個 UCX 程式的資源和設定。它是 UCX 的核心起點，所有的通訊操作都必須根據這個結構。簡單來說，`ucp_context_t` 主要用來抽象底層硬體資源、設定應用程式層的需求，並提供通訊的基本功能。
        
        **[src/ucp/core/ucp_context.h]**
        [permalink](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/84e459e73df4f02aecd044c44e4584d88f4b9b0e/src/ucp/core/ucp_context.h#L267-L384)
        ![1.11](https://github.com/ChienPei/Parallel-Programming/blob/main/HW5/pics/1.11.png?raw=true)

    - 2.2 ucp_worker
        - `ucp_worker_t` 在 UCX 中負責執行通訊操作，對應到一個特定的 thread 或邏輯處理環境。它會根據 `ucp_context_t`，負責管理和操作通訊資源，例如 `ucp_ep`（Endpoint）。一個 UCX 應用可以擁有多個 `ucp_worker`，每個 `worker` 可以與不同的 thread 或硬體資源綁定。
        
        **[src/ucp/core/ucp_worker.h]**
        [permalink](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/84e459e73df4f02aecd044c44e4584d88f4b9b0e/src/ucp/core/ucp_worker.h#L266-L360)
        ![1.12](https://github.com/ChienPei/Parallel-Programming/blob/main/HW5/pics/1.12.png?raw=true)

    - 2.3 ucp_ep
        - `ucp_ep_t` 是 UCX 中的 **Endpoint** 結構，代表一個與遠端 `ucp_worker` 的通訊連接。UCX 通訊模型中，所有的數據傳輸操作都會透過 `ucp_ep_t` 完成。也就是說，`ucp_ep_t` 負責維護與遠端節點之間的連接狀態，包括連接的`conn_sn`和`am_lane`。底層的數據傳輸透過 `uct_eps` 完成，每個 `uct_ep` 對應一條物理傳輸通道（例如 RDMA 或 SHM）。
        
        **[src/ucp/core/ucp_ep.h]**
        [permalink](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/84e459e73df4f02aecd044c44e4584d88f4b9b0e/src/ucp/core/ucp_ep.h#L536-L582)
        ![1.13](https://github.com/ChienPei/Parallel-Programming/blob/main/HW5/pics/1.13.png?raw=true)

    - 2.4 執行結果：
        ![1.14](https://github.com/ChienPei/Parallel-Programming/blob/main/HW5/pics/1.14.png?raw=true)

        **A diagram illustrating the architectural design of UCX:**

        ```bash
        Host A                                Host B
        ┌─────────────────────────────────┐  ┌─────────────────────────────────┐
        │ ucp_context (Global Context)    │  │ ucp_context (Global Context)    │
        │ - 初始化全局通信資源             │  │ - 初始化全局通信資源             │
        │ ┌─────────────────────────────┐ │  │ ┌─────────────────────────────┐ │
        │ │ ucp_worker (Communication) │ │  │ │ ucp_worker (Communication) │ │
        │ │ - 管理本地通信資源           │ │  │ │ - 管理本地通信資源           │ │
        │ │ - 支援多線程操作             │ │  │ │ - 支援多線程操作             │ │
        │ │ ┌─────────────────────────┐ │  │ │ ┌─────────────────────────┐ │ │
        │ │ │ ucp_ep (Endpoint)       │ │  │ │ │ ucp_ep (Endpoint)       │ │ │
        │ │ │ - 發送數據到 Host B      │ │  │ │ - 接收數據來自 Host A     │ │ │
        │ │ │ - 收到 Host B 的確認     │ │  │ │ - 回傳確認到 Host A       │ │ │
        │ │ └─────────────────────────┘ │  │ │ └─────────────────────────┘ │ │
        │ └─────────────────────────────┘ │  │ └─────────────────────────────┘ │
        └─────────────────────────────────┘  └─────────────────────────────────┘

        Data Flow:
        1. Host A → Host B: "Hello from rank 0"
        2. Host B → Host A: Acknowledgment
        ```

        假設我們在兩個節點上執行 **`srun -N 2 ./send_recv.out`**：

        1. **初始化通訊資源**
            - Host A 和 Host B 上分別呼叫 `ucp_init`，創建各自的 `ucp_context`。
            - 每個 `ucp_context` 都會創建一個對應的 `ucp_worker`，用於管理本地通訊資源並支援多線程操作。（如圖所示）
        2. **建立通訊端點 (Endpoint)**
            - Host A 透過 `ucp_ep_create` 建立指向 Host B 的通訊端點。
            - Host B 同樣透過 `ucp_ep_create` 建立指向 Host A 的端點。
        3. **數據傳輸**
            - Host A 使用 `ucp_ep` 發送訊息：「Hello from rank 0」到 Host B。
            - Host B 接收後，透過 `ucp_ep` 回傳確認訊息給 Host A。

3. Based on the description in HW5, where do you think the following information is loaded/created?
    - `UCX_TLS`
    - TLS selected by UCX
    
    `UCX_TLS` 在初始化時由 `ucp_config_read` 從環境變數中讀取。 
    `TLS selected by UCX` 在 `ucp_init` 初始化過程中，根據硬體資源與設定，自動選擇。

## 2. Implementation
> Please complete the implementation according to the [spec](https://docs.google.com/document/d/1fmm0TFpLxbDP7neNcbLDn8nhZpqUBi9NGRzWjgxZaPE/edit?usp=sharing)
> Describe how you implemented the two special features of HW5.
1. Which files did you modify, and where did you choose to print Line 1 and Line 2?
    **[src/ucs/config/parser.c → ucs_config_parser_print_opts]** 
    [permalink](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/84e459e73df4f02aecd044c44e4584d88f4b9b0e/src/ucs/config/types.h#L85-L94)
    在 **`src/ucs/config/types.h`** 的 ucs_config_print_flags_t 中加入 **`UCS_CONFIG_PRINT_TLS  = UCS_BIT(5)`**
    ![2.1](https://github.com/ChienPei/Parallel-Programming/blob/main/HW5/pics/2.1.png?raw=true)

    在 src/ucs/config/parser.c 的 ucs_config_parser_print_opts 印出 line 1 的資訊：
    [permalink](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/84e459e73df4f02aecd044c44e4584d88f4b9b0e/src/ucs/config/parser.c#L1880-L1883)
    ![2.2](https://github.com/ChienPei/Parallel-Programming/blob/main/HW5/pics/2.2.png?raw=true)

    在 src/ucp/core/ucp_worker.c 的 ucp_worker_print_used_tls 印出 line 2 ，此外也呼叫 ucp_config_print 來印出 line 1 。
    [permalink](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/84e459e73df4f02aecd044c44e4584d88f4b9b0e/src/ucp/core/ucp_worker.c#L1855-L1856)
    ![2.3](https://github.com/ChienPei/Parallel-Programming/blob/main/HW5/pics/2.3.png?raw=true)

2. How do the functions in these files call each other? Why is it designed this way?
    以下是檔案之間的互動：

    1. **[types.h]**
        - 定義了 `ucs_config_print_flags_t`，用來表示不同的配置要印出什麼的選項。
        - 新增了 `UCS_CONFIG_PRINT_TLS`，專門用來印環境變數 `UCX_TLS`。
    2. **[parser.c]**
        - **`ucs_config_parser_print_opts`**
            - 負責根據傳入的 `flags` 進行對應的配置輸出。
            - 當 `flags` 包含 `UCS_CONFIG_PRINT_TLS` 時，函數會檢查並印出環境變數 `UCX_TLS` 的值；若未設置，則輸出 "UCX_TLS=Not Set"。
        - 這個函數提供靈活的配置要印什麼的功能，並被其他模組（如 `ucp_worker.c`）呼叫以實現相對應的需求。
    3. **[ucp_worker.c]**
        - **`ucp_worker_print_used_tls`**
            - 負責計算與分類資料通道（例如 `tag_lanes_map`、`rma_lanes_map` 等），並最終呼叫 `ucp_config_print` 和 `ucs_config_parser_print_opts`。
            - 該函數設計的核心目的是整合資料處理和配置要印什麼，確保輸出內容包含所選的 TLS（傳輸層安全協定）。
    
    設計原因：
    1. **模組化設計**
        - 在要印出什麼的邏輯集中在 `parser.c` 中，確保邏輯獨立並可重複使用。
        - `ucp_worker.c` 專注於處理 worker 層級的資料結構，將打印邏輯與資料計算分離，降低模組之間的連結。
    2. 使用 `flags`（如 `UCS_CONFIG_PRINT_TLS`）作為控制開關，讓我們可以輕鬆定制要印出的內容，而不需修改核心邏輯，增加靈活性和可擴展性。
    3. 增加對 `UCX_TLS` 要印什麼的選項，有助於快速檢查環境變數的設置狀態，並提供對應的錯誤資訊。
    4. `parser.c` 中的 `ucs_config_parser_print_opts` 可被不同模組呼叫，避免重複實現，提升代碼可維護性，也增加可讀性和可重複使用性。
        

3. Observe when Line 1 and 2 are printed during the call of which UCP API?
    1. **Line 1**：目的是檢查環境變數 `UCX_TLS` 是否正確設置。
        - Printed 的時機：
            - 當呼叫 `ucp_worker_print_used_tls` 時，觸發 `ucp_config_print`，此函數印出環境變數 `UCX_TLS` 的設置。
        - 相關的 API：
            - `ucp_worker_create` -> `ucp_worker_print_used_tls` -> `ucp_config_print`
    2. **Line 2**：完成 `TLS` 資訊處理後，輸出分析結果，方便進行追蹤問題。
        - Printed 的時機：
            - 同樣在 `ucp_worker_print_used_tls` 函數中，當完成 `TLS` 資訊計算後，透過 `fprintf` 印出結果。
        - 相關的 API：
            - `ucp_worker_create` -> `ucp_worker_print_used_tls`

4. Does it match your expectations for questions **1-3**? Why?

    是的，我對問題 1-3 的回答符合我的預期，因為 UCX_TLS 的值在初始化時透過 ucp_config_read 從環境變數讀取，而 TLS selected by UCX 則在 ucp_init 階段根據硬體資源與設置自動選擇，這與實際執行的流程一致！

5. In implementing the features, we see variables like lanes, tl_rsc, tl_name, tl_device, bitmap, iface, etc., used to store different Layer's protocol information. Please explain what information each of them stores.

    - **`lanes`**: 代表通往遠端的多個邏輯通道，每個通道對應一個具體的傳輸層資源（如 TCP, InfiniBand）。
        - **儲存內容**：
            - 每個通道的功能分配（例如 TAG、RMA 功能）。
            - 資源的優先順序與資源 index。
    - **`tl_rsc`**：表示 UCX 中的傳輸層資源，目的是支援通訊的最佳化。
        - **儲存內容**：
            - 資源的詳細資訊，例如硬體名稱（如網路卡）與協議名稱（如 `rc`, `tcp`）。
            - 資源的能力與總數。
    - **`tl_name`**：用於標識特定傳輸層協議的名稱。
        - **儲存內容**：
            - 協議名稱，幫助區分不同的通訊方法。
    - **`tl_device`** ：代表資源所運行的硬體裝置，例如傳輸層所依賴的網路介面卡。
        - **儲存內容**：
            - 裝置的類型與名稱，幫助系統選擇相對應的硬體。
    - **`bitmap`** : 利用 bitmap 高效管理資源的分配與使用情況。
        - **儲存內容**：
            - 每個 lane 的啟用狀態，例如哪些 lanes 支援特定功能（如 RMA、AMO）。
    - **`iface`**: 負責處理與硬體或網路相關的通訊接口。
        - **儲存內容**：
            - 與每個資源相關的接口細節，例如數據傳輸操作的執行。

## 3. Optimize System 
1. Below are the current configurations for OpenMPI and UCX in the system. Based on your learning, what methods can you use to optimize single-node performance by setting UCX environment variables?

```
-------------------------------------------------------------------
/opt/modulefiles/openmpi/ucx-pp:

module-whatis   {OpenMPI 4.1.6}
conflict        mpi
module          load ucx/1.15.0
prepend-path    PATH /opt/openmpi-4.1.6/bin
prepend-path    LD_LIBRARY_PATH /opt/openmpi-4.1.6/lib
prepend-path    MANPATH /opt/openmpi-4.1.6/share/man
prepend-path    CPATH /opt/openmpi-4.1.6/include
setenv          UCX_TLS ud_verbs
setenv          UCX_NET_DEVICES ibp3s0:1
-------------------------------------------------------------------
```

1. Please use the following commands to test different data sizes for latency and bandwidth, to verify your ideas:
```bash
module load openmpi/ucx-pp
mpiucx -n 2 $HOME/UCX-lsalab/test/mpi/osu/pt2pt/osu_latency
mpiucx -n 2 $HOME/UCX-lsalab/test/mpi/osu/pt2pt/osu_bw
```
以下是執行的示意圖：
![3.1](https://github.com/ChienPei/Parallel-Programming/blob/main/HW5/pics/3.1.png?raw=true)
![3.2](https://github.com/ChienPei/Parallel-Programming/blob/main/HW5/pics/3.2.png?raw=true)

2. Please create a chart to illustrate the impact of different parameter options on various data sizes and the effects of different testsuite.
![3.3](https://github.com/ChienPei/Parallel-Programming/blob/main/HW5/pics/3.3.png?raw=true)
![3.4](https://github.com/ChienPei/Parallel-Programming/blob/main/HW5/pics/3.4.png?raw=true)


3. Based on the chart, explain the impact of different TLS implementations and hypothesize the possible reasons (references required).
**`ud_verbs` ：unreliable datagram over verbs**
**`sm` ：all shared memory transports**
**`posix` : portable operating system interface**
**`all` : use all the available transports**

**Latency 分析：**

1. **`ud_verbs` :** 在小資料量下延遲顯著高於其他 TLS，隨資料量增大雖仍最高，但差距縮小。這可能是因為 `ud_verbs` 為 RDMA 設計，啟動時涉及額外的初始化和通訊設置成本，對小資料量影響較大。
2. **`sm` :** 表現接近 `posix`，但在大資料量下延遲略高。這可能由於 `sm` 支援多進程通信，其內部邏輯增加了額外開銷，對大數據處理的性能造成輕微影響。
3. **`posix` :** 延遲表現最佳且穩定，原因可能是 POSIX 使用共享記憶體直接操作，實現簡單高效，避免同步和檢查的額外開銷，特別適合小數據傳輸。
4. **`all` :** 延遲通常介於 `ud_verbs` 與其他協議之間，可能由於 UCX 涵蓋多種傳輸協議，需進行協商與選擇，帶來一定的協議切換成本。

**Bandwidth 分析：**

1. **`ud_verbs`**：在小資料量下 Bandwidth 較小，大資料量時穩定，但表現仍舊不如其他協議，可能因初始化成本較高。
2. **`sm`**：在中小資料量下表現優異，Bandwidth 增長相對迅速。大資料量時稍有下降，可能由於同步或資源爭用。
3. **`all`**：穩定表現，Bandwith 高於 `ud_verbs`，雖然不是最好的，但也表現得還不錯，可能是因為協議切換或協商成本，在大資料量下略低於 `sm` 和 `posix`。
4. **`posix`**：雖然在小資料量下不是最好的，但是資料量越大，表現越好，Bandwidth 可達 10406.95 MB/s，可能是因為受益於共享記憶體的高效直接操作。

**references**: 

https://github.com/openucx/ucx/wiki/UCX-environment-parameters

https://blog.csdn.net/ssbandjl/article/details/133758302


### Advanced Challenge: Multi-Node Testing

This challenge involves testing the performance across multiple nodes. You can accomplish this by utilizing the sbatch script provided below. The task includes creating tables and providing explanations based on your findings. Notably, Writing a comprehensive report on this exercise can earn you up to 5 additional points.

- For information on sbatch, refer to the documentation at [Slurm's sbatch page](https://slurm.schedmd.com/sbatch.html).
- To conduct multi-node testing, use the following command:
```
cd ~/UCX-lsalab/test/
sbatch run.batch
```


## 4. Experience & Conclusion
1. What have you learned from this homework?
透過這次作業，我學習了 UCX 的基本架構與設計原理，深入理解了 ucp_context、ucp_worker 和 ucp_ep 的功能。此外，我學會如何透過調整環境變數（如 UCX_TLS）來測試不同傳輸協議在 latency 與 bandwidth 的性能表現，並針對測試數據進行分析。在實作過程中，我也更加熟悉 UCX 的應用！

2. How long did you spend on the assignment?
大約一週，包括回去看教學影片，以及 trace code，其中，花最多實驗在理解每個變數的意義。

3. Feedback (optional)
謝謝助教！