# CUDA Convolution Lab Package

This package includes a full starter solution for the convolution homework:

- `convolution_lab.cu` - main CUDA program with 5 versions of convolution
- `Makefile` - `all`, `clean`, `test`, and `analyze` targets
- `analyze_results.py` - builds graphs, summaries, and break-even estimates from `results.csv`
- `Convolution_Lab_Report_Template.docx` - report template you can fill after running the program

## Versions implemented

1. `cpu_single` - single-threaded CPU convolution
2. `gpu_global` - tuned launch configuration, otherwise plain global-memory CUDA kernel
3. `gpu_tiled` - tiled shared-memory kernel
4. `gpu_constant` - global-memory kernel using constant memory for the filter
5. `gpu_tiled_constant` - tiled shared-memory kernel using constant memory for the filter

## Filters included

- `emboss_3x3`
- `sharpen_3x3`
- `gaussian_5x5`
- `edge_5x5`

## Test images

The program generates synthetic grayscale images in memory so file I/O is not part of the timed section:

- `gradient_512`
- `checker_1024`
- `rings_2048`
- `noise_3072`

## Build and run

```bash
make
make test
python3 analyze_results.py lab_output/results.csv lab_output
```

Or all at once:

```bash
make analyze
```

## Output files

After `make test`, the output folder contains:

- `results.csv` - one row per repeat for each image/filter/version
- `sample_*.pgm` - example transformed images for the report

After analysis, the output folder also contains:

- `timing_summary.csv`
- `speedup_summary.csv`
- `break_even_summary.csv`
- `speedup_vs_pixels.png`
- `runtime_vs_pixels.png`
- `filter_size_comparison.png`
- `stat_tests.txt`
- `analysis_summary.md`

## Notes

- Timing includes host-to-device and device-to-host transfers.
- Verification is run separately and is not included in the timed section.
- The constant-memory versions support filters up to `31 x 31` because of the constant-memory array size.
- The provided `Makefile` is aimed at a Linux/UNIX CUDA environment such as a university server or WSL setup.
