CC = mpicc
CXX = mpicxx
CXXFLAGS = -O3 -lm -march=native
CFLAGS = -O3 -lm -march=native
export OMPI_CXX = icc
CXXFLAGS += -fp-model precise 
TARGETS = hw1

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS)