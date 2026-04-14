
NVCC = nvcc
TARGET = matrix_mul_cuda
SRC = matrix_mul_cuda.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -O3 $(TARGET) $(SRC)

clean:
	rm -f $(TARGET) output.txt *.csv *.log

test: $(TARGET)
	./$(TARGET) 256 256 256 > test_square_256.log
	./$(TARGET) 512 256 128 > test_rect_512_256_128.log
	./$(TARGET) 1024 512 256 > test_rect_1024_512_256.log