#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <boost/sort/spreadsort/spreadsort.hpp> 

void merge_arrays_min(float *local_data, int local_n, float *neighbor_data, int neighbor_n, float *merged_data) {
    int i = 0, j = 0, k = 0;
    while (i < local_n && j < neighbor_n && k < local_n) {
        if (local_data[i] <= neighbor_data[j]) {
            merged_data[k++] = local_data[i++];
        } else {
            merged_data[k++] = neighbor_data[j++];
        }
    }
    while (i < local_n && k < local_n) {
        merged_data[k++] = local_data[i++];
    }
    while (j < neighbor_n && k < local_n) {
        merged_data[k++] = neighbor_data[j++];
    }
}

void merge_arrays_max(float *local_data, int local_n, float *neighbor_data, int neighbor_n, float *merged_data) {
    int i = local_n - 1, j = neighbor_n - 1, k = local_n - 1;
    while (i >= 0 && j >= 0 && k >= 0) {
        if (local_data[i] >= neighbor_data[j]) {
            merged_data[k--] = local_data[i--];
        } else {
            merged_data[k--] = neighbor_data[j--];
        }
    }
    while (i >= 0 && k >= 0) {
        merged_data[k--] = local_data[i--];
    }
    while (j >= 0 && k >= 0) {
        merged_data[k--] = neighbor_data[j--];
    }
}

void exchange_and_merge(float **local_data, int local_n, float *neighbor_data, int neighbor_n,
                        int neighbor_rank, MPI_Op direction, float **merged_data, MPI_Comm comm) {
    MPI_Status status;
    MPI_Sendrecv(*local_data, local_n, MPI_FLOAT, neighbor_rank, 1,
                 neighbor_data, neighbor_n, MPI_FLOAT, neighbor_rank, 1, comm, &status);

    if ((direction == MPI_MIN && (*local_data)[local_n - 1] <= neighbor_data[0]) ||
        (direction == MPI_MAX && (*local_data)[0] >= neighbor_data[neighbor_n - 1])) {
        return;
    }

    // merge_arrays(*local_data, local_n, neighbor_data, neighbor_n, merged_data);
    if (direction == MPI_MIN) {
        // Keep the smaller half
        merge_arrays_min(*local_data, local_n, neighbor_data, neighbor_n, *merged_data);
    } else {
        // Keep the larger half
        merge_arrays_max(*local_data, local_n, neighbor_data, neighbor_n, *merged_data);
    }

    // Swap pointers to avoid copying
    // float *temp = *local_data;
    // *local_data = *merged_data;
    // *merged_data = temp;
    std::swap(*local_data, *merged_data);
}

int main(int argc, char *argv[]) {

    int n;
    char *input_filename, *output_filename;

    MPI_Init(&argc, &argv);

    // double start_time, end_time;
    // start_time = MPI_Wtime(); // Start timing
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    n = atoi(argv[1]);
    input_filename = argv[2];
    output_filename = argv[3];

    int elements_per_proc = n / size;
    int remainder = n % size;
    int start_idx, end_idx;
    if (rank < remainder) {
        start_idx = rank * (elements_per_proc + 1);
        end_idx = start_idx + elements_per_proc;
    } else {
        start_idx = rank * elements_per_proc + remainder;
        end_idx = start_idx + elements_per_proc - 1;
    }
    int local_n = end_idx - start_idx + 1;
    float *local_data = (float *)malloc(local_n * sizeof(float));

    MPI_File input_file;
    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);

    MPI_Offset offset = start_idx * sizeof(float);
    MPI_File_read_at(input_file, offset, local_data, local_n, MPI_FLOAT, MPI_STATUS_IGNORE);

    if (local_n >= 2) boost::sort::spreadsort::spreadsort(local_data, local_data + local_n);

    int max_neighbor_n = elements_per_proc + (remainder > 0 ? 1 : 0); 
    float *neighbor_data = (float *)malloc(max_neighbor_n * sizeof(float));
    float *merged_data = (float *)malloc(local_n * sizeof(float));

    // Odd-Even Transposition Sort
    for (int phase = 0; phase <= size; phase++) {
        if ((phase + rank) % 2 == 0) {
            if (rank < size - 1) {
                int neighbor_rank = rank + 1;
                int neighbor_n = (neighbor_rank < remainder) ? elements_per_proc + 1 : elements_per_proc;
                exchange_and_merge(&local_data, local_n, neighbor_data, neighbor_n, neighbor_rank,
                                   MPI_MIN, &merged_data, MPI_COMM_WORLD);
            }
        } else {
            if (rank > 0) {
                int neighbor_rank = rank - 1;
                int neighbor_n = (neighbor_rank < remainder) ? elements_per_proc + 1 : elements_per_proc;
                exchange_and_merge(&local_data, local_n, neighbor_data, neighbor_n, neighbor_rank,
                                   MPI_MAX, &merged_data, MPI_COMM_WORLD);
            }
        }
    }

    // free(neighbor_data);
    // free(merged_data);
    // free(temp_data);

    MPI_File output_file;
    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    MPI_File_write_at(output_file, offset, local_data, local_n, MPI_FLOAT, MPI_STATUS_IGNORE);

    // end_time = MPI_Wtime(); // End timing
    // if (rank == 0) {
    //     printf("Total execution time: %f seconds\n", end_time - start_time);
    // }

    // free(local_data);
    // MPI_Finalize();
}
