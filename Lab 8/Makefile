# Commands:
#
#   make
#   make run-gigabit
#   make run-gigabyte
#   make clean


NVCC = nvcc
NVCCFLAGS = -O3

SRC = segmented_scan_multiblock_updated.cu
EXE = segmented_scan_multiblock

RESULT_DIR = results

# Test settings
REPEATS = 7

# Gigabit-size test:
# 2^20 through 2^24 elements
GIGABIT_MIN_POWER = 20
GIGABIT_MAX_POWER = 24

# Optional gigabyte size test:
# 2^27 elements
GIGABYTE_MIN_POWER = 27
GIGABYTE_MAX_POWER = 27

# CPU verification is only done up to this many elements.
# 4,194,304 = 2^22 elements so I dont waste time with the cpu.
# Bigger sizes still run, but verification columns will be -1.
VERIFY_LIMIT = 4194304

all: $(EXE)

$(EXE): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(EXE)

$(RESULT_DIR):
	mkdir -p $(RESULT_DIR)

run-gigabit: $(EXE) $(RESULT_DIR)
	@echo "Running segmented scan gigabit-size experiment..."
	@echo "Testing 2^$(GIGABIT_MIN_POWER) through 2^$(GIGABIT_MAX_POWER) elements"
	./$(EXE) $(REPEATS) $(GIGABIT_MIN_POWER) $(GIGABIT_MAX_POWER) $(VERIFY_LIMIT) | tee $(RESULT_DIR)/segmented_scan_gigabit_console.txt
	cp segmented_scan_multiblock_results.csv $(RESULT_DIR)/segmented_scan_gigabit_results.csv
	@echo "Saved CSV to $(RESULT_DIR)/segmented_scan_gigabit_results.csv"
	@echo "Saved console log to $(RESULT_DIR)/segmented_scan_gigabit_console.txt"

run-gigabyte: $(EXE) $(RESULT_DIR)
	@echo "Running optional segmented scan gigabyte-size experiment..."
	@echo "Testing 2^$(GIGABYTE_MIN_POWER) elements"
	@echo "This needs a lot more memory than the input size because the program uses output and temp buffers."
	./$(EXE) 3 $(GIGABYTE_MIN_POWER) $(GIGABYTE_MAX_POWER) $(VERIFY_LIMIT) | tee $(RESULT_DIR)/segmented_scan_gigabyte_console.txt
	cp segmented_scan_multiblock_results.csv $(RESULT_DIR)/segmented_scan_gigabyte_results.csv
	@echo "Saved CSV to $(RESULT_DIR)/segmented_scan_gigabyte_results.csv"
	@echo "Saved console log to $(RESULT_DIR)/segmented_scan_gigabyte_console.txt"

run-all: run-gigabit run-gigabyte
	@echo "All large segmented scan tests completed."

clean:
	rm -f $(EXE)
	rm -f segmented_scan_multiblock_results.csv
	rm -rf $(RESULT_DIR)

.PHONY: all run-gigabit run-gigabyte run-all clean
