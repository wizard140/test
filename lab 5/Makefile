NVCC = nvcc
TARGET = vector_scale_add
SRC = vector_scale_add.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -O3 -o $(TARGET) $(SRC)

test: $(TARGET)
	./$(TARGET) 11

clean:
	rm -f $(TARGET) results.csv timing.png speedup.png comparison.png
