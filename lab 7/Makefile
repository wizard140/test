NVCC      := nvcc
TARGET    := convolution_lab
SRC       := convolution_lab.cu
OUT_DIR   := lab_output
REPEATS   := 7
PYTHON    := python3
NVCCFLAGS := -O3 -std=c++17

.PHONY: all clean test analyze

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
	rm -rf $(OUT_DIR)

# Generates all timing data and sample output images.
test: $(TARGET)
	mkdir -p $(OUT_DIR)
	./$(TARGET) $(REPEATS) $(OUT_DIR)

# Optional helper target for graphs and summary tables.
analyze: test
	$(PYTHON) analyze_results.py $(OUT_DIR)/results.csv $(OUT_DIR)
