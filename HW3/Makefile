# Compiler settings
CC = mpicc
CXX = mpicxx
NVCC = nvcc
HIPCC = hipcc

# Flags
CFLAGS = -O3 -fopenmp
CXXFLAGS = -std=c++11 -O3 -fopenmp
NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61
HIPCCFLAGS = -std=c++11 -O3 --offload-arch=gfx90a -fopenmp
LDFLAGS = -lm
OMPFLAGS = -fopenmp
OMPLDFLAGS = -lgomp
LDLIBS = -pthread

# Targets
TARGETS = hw3-1 hw3-2 hw3-2-amd hw3-3 hw3-3-amd seq

.PHONY: all clean

# Build all targets
all: $(TARGETS)

# Clean target
clean:
	rm -f $(TARGETS) $(TARGETS:=.o) *.hip.o *.log

# hw3-1 target
hw3-1: hw3-1.cc
	$(CXX) $(CXXFLAGS) $(LDLIBS) -o $@ $<


# hw3-2 target for NVIDIA GPU
hw3-2: hw3-2.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $<

# hw3-2 target for AMD GPU
hw3-2-amd: hw3-2.hip
	$(HIPCC) $(HIPCCFLAGS) $(LDFLAGS) -o $@ $<

# hw3-3 target for NVIDIA GPU
hw3-3: hw3-3.cu
	$(NVCC) $(NVFLAGS) -Xcompiler "$(OMPFLAGS)" -Xlinker "$(OMPLDFLAGS)" -o $@ $<

# hw3-3 target for AMD GPU
hw3-3-amd: hw3-3.hip
	$(HIPCC) $(HIPCCFLAGS) $(OMPFLAGS) $(LDFLAGS) $(OMPLDFLAGS) -o $@ $<

# Sequential implementation
seq: seq.cc
	g++ $(CXXFLAGS) $(LDFLAGS) -o $@ $<

# Run hw3-2 with srun
.PHONY: run
run: hw3-2
	@echo "Running hw3-2 with srun..."
	srun -N1 -n1 ./hw3-2 | tee hw3-2-output.log
