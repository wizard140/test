NVCC = nvcc
TARGET = matrix_mul_cuda
SRC = matrix_mul_cuda.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -O3 -std=c++17 -o $(TARGET) $(SRC)

test: $(TARGET)
	./$(TARGET) 256 256 256
	./$(TARGET) 512 256 128
	./$(TARGET) 1024 512 256

clean:
	rm -f $(TARGET) output.txt *.csv *.log

testcsv: $(TARGET)
	python3 run_matrix_csv.py
