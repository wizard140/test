NVCC = nvcc
TARGET = matrix_mul_cuda
SRC = matrix_mul_cuda.cu
ARCH ?= sm_89
NVCCFLAGS = -O3 -std=c++17 -Wno-deprecated-gpu-targets -arch=$(ARCH)

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SRC)

test: $(TARGET)
	./$(TARGET) 256 256 256
	./$(TARGET) 512 256 128
	./$(TARGET) 1024 512 256
	./$(TARGET) 2048 1024 512
	./$(TARGET) 2048 256 2048

clean:
	rm -f $(TARGET) output.txt *.csv *.log
