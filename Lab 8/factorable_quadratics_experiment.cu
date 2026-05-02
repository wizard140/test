#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

using ll = long long;
using ull = unsigned long long;

static bool hostPerfectSquare(ull value)
{
    ull root = static_cast<ull>(std::sqrt(static_cast<long double>(value)));

    while ((root + 1) * (root + 1) <= value)
    {
        root++;
    }

    while (root * root > value)
    {
        root--;
    }

    return root * root == value;
}

static bool hostIsFactorable(int a, int b, int c)
{
    ll disc = 1LL * b * b - 4LL * a * c;

    if (disc < 0)
    {
        return false;
    }

    return hostPerfectSquare(static_cast<ull>(disc));
}

static ll countFactorableCpu(int limit)
{
    ll count = 0;

    for (int a = -limit; a <= limit; a++)
    {
        if (a == 0)
        {
            continue;
        }

        for (int b = -limit; b <= limit; b++)
        {
            if (b == 0)
            {
                continue;
            }

            for (int c = -limit; c <= limit; c++)
            {
                if (c == 0)
                {
                    continue;
                }

                if (hostIsFactorable(a, b, c))
                {
                    count++;
                }
            }
        }
    }

    return count;
}

__device__ bool devicePerfectSquare(ull value)
{
    double rootAsDouble = sqrt(static_cast<double>(value));
    ull root = static_cast<ull>(rootAsDouble + 0.5);

    ull square = root * root;

    if (square == value)
    {
        return true;
    }

    if (root > 0)
    {
        ull lower = (root - 1) * (root - 1);

        if (lower == value)
        {
            return true;
        }
    }

    ull upper = (root + 1) * (root + 1);

    return upper == value;
}

__global__ void countFactorableKernel(int limit, ull total, ull *globalCount)
{
    ull tid = static_cast<ull>(blockIdx.x) * blockDim.x + threadIdx.x;
    ull stride = static_cast<ull>(blockDim.x) * gridDim.x;

    int span = 2 * limit;
    ull perAxis = static_cast<ull>(span);
    ull plane = perAxis * perAxis;

    ull localCount = 0;

    for (ull idx = tid; idx < total; idx += stride)
    {
        ull aIndex = idx / plane;
        ull remainder = idx % plane;
        ull bIndex = remainder / perAxis;
        ull cIndex = remainder % perAxis;

        int a = static_cast<int>(aIndex) - limit;
        int b = static_cast<int>(bIndex) - limit;
        int c = static_cast<int>(cIndex) - limit;

        if (a >= 0)
        {
            a++;
        }

        if (b >= 0)
        {
            b++;
        }

        if (c >= 0)
        {
            c++;
        }

        ll disc = 1LL * b * b - 4LL * a * c;

        if (disc >= 0 && devicePerfectSquare(static_cast<ull>(disc)))
        {
            localCount++;
        }
    }

    if (localCount > 0)
    {
        atomicAdd(globalCount, localCount);
    }
}

static bool cudaOk(cudaError_t code, const std::string &label)
{
    if (code != cudaSuccess)
    {
        std::cerr << label << " failed: " << cudaGetErrorString(code) << "\n";
        return false;
    }

    return true;
}

static ull totalQuadraticsForLimit(int limit)
{
    ull span = static_cast<ull>(2 * limit);
    return span * span * span;
}

static ull countFactorableGpu(int limit, int threadsPerBlock, int blockCount, double &milliseconds)
{
    ull total = totalQuadraticsForLimit(limit);

    ull *dCount = nullptr;
    ull hCount = 0;

    auto start = std::chrono::high_resolution_clock::now();

    if (!cudaOk(cudaMalloc(&dCount, sizeof(ull)), "cudaMalloc dCount"))
    {
        return 0;
    }

    if (!cudaOk(cudaMemset(dCount, 0, sizeof(ull)), "cudaMemset dCount"))
    {
        cudaFree(dCount);
        return 0;
    }

    countFactorableKernel<<<blockCount, threadsPerBlock>>>(limit, total, dCount);

    if (!cudaOk(cudaGetLastError(), "kernel launch") ||
        !cudaOk(cudaDeviceSynchronize(), "kernel synchronize") ||
        !cudaOk(cudaMemcpy(&hCount, dCount, sizeof(ull), cudaMemcpyDeviceToHost), "cudaMemcpy result"))
    {
        cudaFree(dCount);
        return 0;
    }

    cudaFree(dCount);

    auto end = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration<double, std::milli>(end - start).count();

    return hCount;
}

static bool fileExists(const std::string &fileName)
{
    std::ifstream file(fileName);
    return file.good();
}

static void appendCsv(const std::string &fileName,
                      int limit,
                      int threadsPerBlock,
                      int blockCount,
                      int cpuCheck,
                      int repeatNumber,
                      ull totalQuadratics,
                      ull gpuCount,
                      double gpuMs,
                      double gpuThroughput,
                      ll cpuCount,
                      double cpuMs,
                      double cpuThroughput,
                      int match)
{
    bool exists = fileExists(fileName);

    std::ofstream out(fileName, std::ios::app);

    if (!exists)
    {
        out << "limit,threadsPerBlock,blockCount,cpuCheck,repeat,totalQuadratics,"
            << "gpuCount,gpuMs,gpuThroughput,cpuCount,cpuMs,cpuThroughput,match\n";
    }

    out << limit << ","
        << threadsPerBlock << ","
        << blockCount << ","
        << cpuCheck << ","
        << repeatNumber << ","
        << totalQuadratics << ","
        << gpuCount << ","
        << gpuMs << ","
        << gpuThroughput << ","
        << cpuCount << ","
        << cpuMs << ","
        << cpuThroughput << ","
        << match << "\n";
}

int main(int argc, char **argv)
{
    if (argc < 2 || argc > 7)
    {
        std::cout << "Usage: factorable_quadratics <i> [threadsPerBlock] [blockCount] [cpuCheck] [repeats] [csvFile]\n";
        std::cout << "Example: factorable_quadratics 2000 256 4096 0 5 results.csv\n";
        std::cout << "cpuCheck: 1 runs a CPU comparison too, 0 skips it\n";
        return 1;
    }

    int limit = std::atoi(argv[1]);
    int threadsPerBlock = (argc >= 3) ? std::atoi(argv[2]) : 256;
    int blockCount = (argc >= 4) ? std::atoi(argv[3]) : 4096;
    int cpuCheck = (argc >= 5) ? std::atoi(argv[4]) : 0;
    int repeats = (argc >= 6) ? std::atoi(argv[5]) : 1;
    std::string csvFile = (argc >= 7) ? argv[6] : "";

    if (limit <= 0 || threadsPerBlock <= 0 || blockCount <= 0 || repeats <= 0)
    {
        std::cerr << "All numeric arguments must be positive.\n";
        return 1;
    }

    int device = 0;
    cudaDeviceProp prop{};

    if (!cudaOk(cudaGetDevice(&device), "cudaGetDevice") ||
        !cudaOk(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties"))
    {
        return 1;
    }

    ull totalQuadratics = totalQuadraticsForLimit(limit);

    std::cout << "Factorable quadratics in coefficient range [-" << limit << ", " << limit << "] excluding 0\n";
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Threads per block: " << threadsPerBlock << "\n";
    std::cout << "Blocks: " << blockCount << "\n";
    std::cout << "Repeats: " << repeats << "\n";
    std::cout << "Total quadratics checked: " << totalQuadratics << "\n";

    double gpuTotalMs = 0.0;
    ull lastGpuCount = 0;

    std::cout << std::fixed << std::setprecision(3);

    for (int repeat = 1; repeat <= repeats; repeat++)
    {
        double gpuMs = 0.0;
        ull gpuCount = countFactorableGpu(limit, threadsPerBlock, blockCount, gpuMs);
        double gpuSeconds = gpuMs / 1000.0;
        double gpuThroughput = (gpuSeconds > 0.0) ? static_cast<double>(totalQuadratics) / gpuSeconds : 0.0;

        ll cpuCount = -1;
        double cpuMs = 0.0;
        double cpuThroughput = 0.0;
        int match = -1;

        gpuTotalMs += gpuMs;
        lastGpuCount = gpuCount;

        std::cout << "Repeat " << repeat << "\n";
        std::cout << "GPU factorable count: " << gpuCount << "\n";
        std::cout << "GPU total time: " << gpuMs << " ms\n";
        std::cout << "GPU throughput: " << gpuThroughput << " quadratics/second\n";

        if (cpuCheck)
        {
            auto cpuStart = std::chrono::high_resolution_clock::now();
            cpuCount = countFactorableCpu(limit);
            auto cpuEnd = std::chrono::high_resolution_clock::now();

            cpuMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();
            double cpuSeconds = cpuMs / 1000.0;
            cpuThroughput = (cpuSeconds > 0.0) ? static_cast<double>(totalQuadratics) / cpuSeconds : 0.0;

            match = (static_cast<ull>(cpuCount) == gpuCount) ? 1 : 0;

            std::cout << "CPU factorable count: " << cpuCount << "\n";
            std::cout << "CPU total time: " << cpuMs << " ms\n";
            std::cout << "CPU throughput: " << cpuThroughput << " quadratics/second\n";

            if (match)
            {
                std::cout << "CPU and GPU counts match.\n";
            }
            else
            {
                std::cout << "WARNING: CPU and GPU counts do not match.\n";
            }
        }

        if (csvFile.length() > 0)
        {
            appendCsv(csvFile,
                      limit,
                      threadsPerBlock,
                      blockCount,
                      cpuCheck,
                      repeat,
                      totalQuadratics,
                      gpuCount,
                      gpuMs,
                      gpuThroughput,
                      cpuCount,
                      cpuMs,
                      cpuThroughput,
                      match);
        }
    }

    double avgGpuMs = gpuTotalMs / static_cast<double>(repeats);
    double avgGpuSeconds = avgGpuMs / 1000.0;
    double avgThroughput = (avgGpuSeconds > 0.0) ? static_cast<double>(totalQuadratics) / avgGpuSeconds : 0.0;

    std::cout << "\nAverage GPU time: " << avgGpuMs << " ms\n";
    std::cout << "Average GPU throughput: " << avgThroughput << " quadratics/second\n";
    std::cout << "Last GPU factorable count: " << lastGpuCount << "\n";

    return 0;
}
