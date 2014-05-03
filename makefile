CXXFLAGS += -g -std=c++11 -Wall -Wextra
CUFLAGS += -g --compiler-options -Wall

ifeq "${config}" "Release"
	CXXFLAGS += -O3 #-mtune=native -march=native -s
	CUFLAGS += -O3 -DNDEBUG
	OUT = Release
else
	CXXFLAGS += -O0 -D_DEBUG=1
	CUFLAGS += -O0 -D_DEBUG=1
	OUT = Debug
endif

CUFLAGS += -arch=sm_35

LINKFLAGS += -L/usr/local/cuda-6.0/lib64/ -lcudart -lcudadevrt

CXXFLAGS += -I/home/igor/Development/qt-workspace/App/TetrisCube
CUFLAGS += -I/home/igor/Development/qt-workspace/App/TetrisCube

CU = /usr/local/cuda-6.0/bin/nvcc

TARGET = $(OUT)/cuda_cube

all: $(TARGET)

$(OUT)/cuda_cube: $(OUT)/main.o $(OUT)/CudaSolve.o $(OUT)/CudaLink.o makefile
	$(CXX) $(CXXFLAGS) $(OUT)/main.o $(OUT)/CudaSolve.o $(OUT)/CudaLink.o -o $@ $(LINKFLAGS)

$(OUT)/main.o: main.cpp SolverC.h makefile
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OUT)/CudaLink.o: $(OUT)/CudaSolve.o
	$(CU) $(CUFLAGS) $< -o $@ -dlink

$(OUT)/CudaSolve.o: CudaSolve.cu TimerC.h utils.h makefile
	$(CU) $(CUFLAGS) -c $< -o $@ -dc -lineinfo

clean:
	rm -f Debug/* && rm -f Release/*

.PHONY = all clean
