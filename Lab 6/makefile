SHELL = /bin/bash

NVCC = nvcc
TARGET = matrix_mul_cuda
SRC = matrix_mul_cuda.cu
ARCH ?= sm_89
NVCCFLAGS = -O3 -std=c++17 -Wno-deprecated-gpu-targets -arch=$(ARCH)

CASES = "256 256 256" "512 256 128" "1024 512 256" "2048 1024 512" "2048 256 2048"
RUNS = 11
CSV = matrix_results.csv

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SRC)

test: $(TARGET)
	./$(TARGET) 256 256 256
	./$(TARGET) 512 256 128
	./$(TARGET) 1024 512 256
	./$(TARGET) 2048 1024 512
	./$(TARGET) 2048 256 2048

testcsv: $(TARGET)
	@echo "m,n,p,output_size,run,cpu_ms,cpu_gflops,naive_ms,naive_gflops,tiled_ms,tiled_gflops,naive_speedup,tiled_speedup,naive_max_abs_diff,tiled_max_abs_diff" > $(CSV)
	@for dims in $(CASES); do \
		set -- $$dims; \
		m=$$1; n=$$2; p=$$3; \
		for run in $$(seq 1 $(RUNS)); do \
			echo "Running $$m $$n $$p (run $$run/$(RUNS))"; \
			output=$$(./$(TARGET) $$m $$n $$p); \
			cpu_line=$$(echo "$$output" | grep '^CPU single'); \
			naive_line=$$(echo "$$output" | grep '^CUDA naive'); \
			tiled_line=$$(echo "$$output" | grep '^CUDA tiled'); \
			naive_diff=$$(echo "$$output" | awk -F': ' '/^Naive max abs diff/ {print $$2}'); \
			tiled_diff=$$(echo "$$output" | awk -F': ' '/^Tiled max abs diff/ {print $$2}'); \
			naive_speedup=$$(echo "$$output" | awk '/^Naive speedup:/ {gsub("x","",$$3); print $$3}'); \
			tiled_speedup=$$(echo "$$output" | awk '/^Tiled speedup:/ {gsub("x","",$$3); print $$3}'); \
			cpu_ms=$$(echo "$$cpu_line" | awk '{print $$3}'); \
			cpu_gflops=$$(echo "$$cpu_line" | awk '{print $$4}'); \
			naive_ms=$$(echo "$$naive_line" | awk '{print $$3}'); \
			naive_gflops=$$(echo "$$naive_line" | awk '{print $$4}'); \
			tiled_ms=$$(echo "$$tiled_line" | awk '{print $$3}'); \
			tiled_gflops=$$(echo "$$tiled_line" | awk '{print $$4}'); \
			echo "$$m,$$n,$$p,$$(($$m * $$p)),$$run,$$cpu_ms,$$cpu_gflops,$$naive_ms,$$naive_gflops,$$tiled_ms,$$tiled_gflops,$$naive_speedup,$$tiled_speedup,$$naive_diff,$$tiled_diff" >> $(CSV); \
		done; \
	done
	@echo "Wrote $(CSV)"

analysis:
	python3 analysis_results.py
	python3 plot_results.py
	python3 analysis_results_matrix.py
	python3 plot_results_matrix.py

clean:
	rm -f $(TARGET) output.txt *.csv *.log *.png *.txt
