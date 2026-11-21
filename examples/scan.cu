// Benchmark the mathematical writeback after every token. The scratch state is FP32.
// Each half warp owns one block of sixteen coordinates.
__device__ float half_value(float value) {
    unsigned short bits;
    float result;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(bits) : "f"(value));
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

__device__ float maximum(float x) {
    for (int distance = 8; distance; distance /= 2)
        x = fmaxf(x, __shfl_xor_sync(0xffffu << (threadIdx.x & 16), x, distance, 16));
    return x;
}

__device__ float writeback(float value, int mode, int lane) {
    if (mode == 3) return value;
    if (mode == 2) return half_value(value);
    float scale = fmaxf(maximum(fabsf(value)) * (1.0f / 127.0f), 1.17549435e-38f);
    int code = max(-127, min(127, __float2int_rn(value / scale)));
    float restored = code * scale;
    if (mode == 1) {
        float error = value - restored;
        float biggest = maximum(fabsf(error));
        // Reduce a tie by relative lane number inside the current 16-lane group.
        int best = fabsf(error) == biggest ? lane : 16;
        for (int step = 8; step; step /= 2)
            best = min(best, __shfl_xor_sync(0xffffu << (threadIdx.x & 16), best, step, 16));
        if (lane == best) restored += half_value(error);
    }
    return restored;
}

extern "C" __global__ void scan(const float* input, const float* delta, const float* bvec,
    const float* cvec, const float* a, const unsigned char* modes, float* state,
    float* output, int width, int length, int first, int end, int boundary, int policy) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid % 16, block = tid / 16, channel = block % width, row = block / width;
    float value = state[tid], decay = a[channel*16 + lane];
    int chosen = modes[channel];
    for (int t = first; t < end; ++t) {
        int at = (row*length + t)*width + channel;
        int coef = (row*length + t)*16 + lane;
        float dt = delta[at];
        value = expf(decay*dt)*value + (dt*input[at])*bvec[coef];
        float dot = value*cvec[coef];
        for (int step = 8; step; step /= 2)
            dot += __shfl_down_sync(0xffffu << (threadIdx.x & 16), dot, step, 16);
        if (lane == 0) output[at] = dot;
        int mode = policy == 0 ? 3 : 1;
        if (policy == 2 && t >= boundary) mode = chosen;
        value = writeback(value, mode, lane);
    }
    state[tid] = value;
}
