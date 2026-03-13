/*
 * complex_cartesian_attn.cu
 *
 * Complex Cartesian FlashAttention — 自定义 CUDA kernel
 *
 * 核心创新: 将笛卡尔分解注意力 A = softmax(Re(S)) · exp(i·Im(S))
 * 的全部计算在 SRAM (共享内存/寄存器) 中完成, 永不将 S×S score
 * 矩阵写入 HBM。
 *
 * 数学等价于:
 *   Re(S) = (Qr@KrT + Qi@KiT) / √Dh
 *   Im(S) = (Qi@KrT - Qr@KiT) / √Dh
 *   P = softmax_j(Re(S))
 *   Or = Σ_j P_ij (cos(Im_ij)·Vr_j - sin(Im_ij)·Vi_j)
 *   Oi = Σ_j P_ij (cos(Im_ij)·Vi_j + sin(Im_ij)·Vr_j)
 *
 * 使用 FlashAttention 的 online softmax 技巧, 分块处理 K/V:
 *   m_new = max(m_old, max(Re(S_block)))
 *   O *= exp(m_old - m_new)      // 重缩放旧累加器
 *   l *= exp(m_old - m_new)      // 重缩放旧分母
 *   O += exp(Re(S) - m_new) · rotated_V   // 累加新贡献
 *   l += Σ exp(Re(S) - m_new)    // 更新分母
 *   最终: O /= l
 *
 * 输入格式: Q, K, V 的 real/imag 分量为独立连续张量 (非 cfloat)
 *   形状: (N, S, Dh) 其中 N = B*H
 *
 * 线程模型:
 *   Grid:  (ceil(S/BR), N)
 *   Block: (BR,)  — 每个线程处理一个 query 行
 *   Dh 维度在寄存器中展开 (Dh=16 时仅 64 个寄存器)
 *
 * SRAM 预算 (BR=BC=64, Dh=16):
 *   K_block: 2 × BC × Dh × 4 = 8 KB (Kr + Ki)
 *   V_block: 2 × BC × Dh × 4 = 8 KB (Vr + Vi)
 *   合计: 16 KB — 远低于 SM 共享内存上限
 *
 * 编译: 由 PyTorch JIT (torch.utils.cpp_extension.load) 自动处理
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

// =====================================================================
//  编译期常量
// =====================================================================

// 每个线程块处理的 query 行数 (= blockDim.x)
#define BR 64
// 每次从 HBM 加载的 key 行数
#define BC 64
// 头维度 (hardcoded for d_model=64, num_heads=4 → Dh=16)
// 未来可改为模板参数支持多种 Dh
#define DH 16

// =====================================================================
//  Forward Kernel
// =====================================================================

__global__ void complex_cartesian_attn_fwd_kernel(
    const float* __restrict__ Qr,  // (N, S, Dh)
    const float* __restrict__ Qi,
    const float* __restrict__ Kr,
    const float* __restrict__ Ki,
    const float* __restrict__ Vr,
    const float* __restrict__ Vi,
    float* __restrict__ Or,         // (N, S, Dh)
    float* __restrict__ Oi,
    float* __restrict__ Lse,        // (N, S) — log-sum-exp, 保存供 backward 使用
    const int S,
    const float scale
) {
    // 当前 (batch*head) 索引
    const int n = blockIdx.y;
    // 当前 query block 的起始行
    const int q_start = blockIdx.x * BR;
    // 当前线程负责的 query 行
    const int q_row = q_start + threadIdx.x;
    // 是否越界
    const bool valid_q = (q_row < S);

    // 偏移到当前 (n) 的数据: base pointer + n * S * Dh
    const int nSD = n * S * DH;
    const float* qr_base = Qr + nSD;
    const float* qi_base = Qi + nSD;
    const float* kr_base = Kr + nSD;
    const float* ki_base = Ki + nSD;
    const float* vr_base = Vr + nSD;
    const float* vi_base = Vi + nSD;

    // ---- 寄存器: query 值 (从 HBM 加载一次, 全程复用) ----
    float qr[DH], qi_reg[DH];
    if (valid_q) {
        const int q_offset = q_row * DH;
        #pragma unroll
        for (int d = 0; d < DH; d++) {
            qr[d] = qr_base[q_offset + d];
            qi_reg[d] = qi_base[q_offset + d];
        }
    }

    // ---- 寄存器: output 累加器 + online softmax 状态 ----
    float o_r[DH], o_i[DH];
    #pragma unroll
    for (int d = 0; d < DH; d++) {
        o_r[d] = 0.0f;
        o_i[d] = 0.0f;
    }
    float m_i = -FLT_MAX;  // running max of Re(S)
    float l_i = 0.0f;       // running sum of exp(Re(S) - m)

    // ---- 共享内存: K/V 块 ----
    __shared__ float sKr[BC][DH];
    __shared__ float sKi[BC][DH];
    __shared__ float sVr[BC][DH];
    __shared__ float sVi[BC][DH];

    // ---- 外循环: 遍历所有 key 块 ----
    const int num_k_blocks = (S + BC - 1) / BC;

    for (int kb = 0; kb < num_k_blocks; kb++) {
        const int k_start = kb * BC;

        // 协作加载 K, V 块到共享内存
        // 每个线程加载一行 (threadIdx.x < BC 时)
        if (threadIdx.x < BC) {
            const int k_row = k_start + threadIdx.x;
            if (k_row < S) {
                const int k_offset = k_row * DH;
                #pragma unroll
                for (int d = 0; d < DH; d++) {
                    sKr[threadIdx.x][d] = kr_base[k_offset + d];
                    sKi[threadIdx.x][d] = ki_base[k_offset + d];
                    sVr[threadIdx.x][d] = vr_base[k_offset + d];
                    sVi[threadIdx.x][d] = vi_base[k_offset + d];
                }
            } else {
                // 越界的 key: 将 Kr/Ki 设为 0 (不影响结果, Re(S)=-inf 被 softmax 消除)
                #pragma unroll
                for (int d = 0; d < DH; d++) {
                    sKr[threadIdx.x][d] = 0.0f;
                    sKi[threadIdx.x][d] = 0.0f;
                    sVr[threadIdx.x][d] = 0.0f;
                    sVi[threadIdx.x][d] = 0.0f;
                }
            }
        }
        __syncthreads();

        if (valid_q) {
            // 计算当前块内有效 key 数
            const int bc_actual = min(BC, S - k_start);

            // 第一遍: 找当前块的 max(Re(S))
            float block_max = -FLT_MAX;
            for (int j = 0; j < bc_actual; j++) {
                float re_s = 0.0f;
                #pragma unroll
                for (int d = 0; d < DH; d++) {
                    re_s += qr[d] * sKr[j][d] + qi_reg[d] * sKi[j][d];
                }
                re_s *= scale;
                if (re_s > block_max) block_max = re_s;
            }

            // Online softmax 更新: 重缩放旧累加器
            float m_new = fmaxf(m_i, block_max);
            float rescale = expf(m_i - m_new);
            #pragma unroll
            for (int d = 0; d < DH; d++) {
                o_r[d] *= rescale;
                o_i[d] *= rescale;
            }
            l_i *= rescale;
            m_i = m_new;

            // 第二遍: 累加当前块的贡献
            for (int j = 0; j < bc_actual; j++) {
                // Hermitian 内积
                float re_s = 0.0f, im_s = 0.0f;
                #pragma unroll
                for (int d = 0; d < DH; d++) {
                    re_s += qr[d] * sKr[j][d] + qi_reg[d] * sKi[j][d];
                    im_s += qi_reg[d] * sKr[j][d] - qr[d] * sKi[j][d];
                }
                re_s *= scale;
                im_s *= scale;

                // exp(Re(S) - m) = unnormalized attention weight
                float p = expf(re_s - m_i);
                l_i += p;

                // cos/sin of imaginary part (相位旋转)
                float c = cosf(im_s);
                float s = sinf(im_s);

                // 累加 p * (cos·V_r - sin·V_i, cos·V_i + sin·V_r)
                #pragma unroll
                for (int d = 0; d < DH; d++) {
                    o_r[d] += p * (c * sVr[j][d] - s * sVi[j][d]);
                    o_i[d] += p * (c * sVi[j][d] + s * sVr[j][d]);
                }
            }
        }
        __syncthreads();
    }

    // ---- 写回: 归一化并存储 ----
    if (valid_q) {
        float inv_l = 1.0f / l_i;
        const int out_offset = nSD + q_row * DH;
        #pragma unroll
        for (int d = 0; d < DH; d++) {
            Or[out_offset + d] = o_r[d] * inv_l;
            Oi[out_offset + d] = o_i[d] * inv_l;
        }
        // 保存 log-sum-exp 供 backward 使用
        Lse[n * S + q_row] = m_i + logf(l_i);
    }
}


// =====================================================================
//  Backward Kernel
// =====================================================================

__global__ void complex_cartesian_attn_bwd_kernel(
    const float* __restrict__ Qr,   // (N, S, Dh)
    const float* __restrict__ Qi,
    const float* __restrict__ Kr,
    const float* __restrict__ Ki,
    const float* __restrict__ Vr,
    const float* __restrict__ Vi,
    const float* __restrict__ Or,   // 前向输出
    const float* __restrict__ Oi,
    const float* __restrict__ Lse,  // (N, S) log-sum-exp from forward
    const float* __restrict__ dOr,  // 上游梯度
    const float* __restrict__ dOi,
    float* __restrict__ dQr,        // 梯度输出
    float* __restrict__ dQi,
    float* __restrict__ dKr,
    float* __restrict__ dKi,
    float* __restrict__ dVr,
    float* __restrict__ dVi,
    const int S,
    const float scale
) {
    /*
     * 反向传播结构 (类 FlashAttention-2 backward):
     * 外循环: 遍历 K/V 块 (累加 dK, dV)
     * 内循环: 遍历 Q 块 (重计算 S, 累加 dQ)
     *
     * 每个线程块负责一个 K/V 块:
     *   Grid:  (ceil(S/BC), N)
     *   Block: (BC,)
     *   threadIdx.x → 当前线程负责的 key 行
     */
    const int n = blockIdx.y;
    const int k_start = blockIdx.x * BC;
    const int k_row = k_start + threadIdx.x;
    const bool valid_k = (k_row < S);

    const int nSD = n * S * DH;

    // 加载当前 key 行到寄存器
    float kr[DH], ki_reg[DH], vr[DH], vi_reg[DH];
    float dk_r[DH], dk_i[DH], dv_r[DH], dv_i[DH];

    if (valid_k) {
        const int k_off = k_row * DH;
        #pragma unroll
        for (int d = 0; d < DH; d++) {
            kr[d] = Kr[nSD + k_off + d];
            ki_reg[d] = Ki[nSD + k_off + d];
            vr[d] = Vr[nSD + k_off + d];
            vi_reg[d] = Vi[nSD + k_off + d];
            dk_r[d] = 0.0f;
            dk_i[d] = 0.0f;
            dv_r[d] = 0.0f;
            dv_i[d] = 0.0f;
        }
    }

    // 共享内存: Q 块 + dO 块 + O 块 (用于计算 D_i)
    __shared__ float sQr[BR][DH];
    __shared__ float sQi[BR][DH];
    __shared__ float sdOr[BR][DH];
    __shared__ float sdOi[BR][DH];
    __shared__ float sOr[BR][DH];
    __shared__ float sOi[BR][DH];
    __shared__ float sLse[BR];

    const int num_q_blocks = (S + BR - 1) / BR;

    for (int qb = 0; qb < num_q_blocks; qb++) {
        const int q_start = qb * BR;

        // 协作加载 Q, dO, O, Lse
        if (threadIdx.x < BR) {
            const int q_row_load = q_start + threadIdx.x;
            if (q_row_load < S) {
                const int q_off = q_row_load * DH;
                #pragma unroll
                for (int d = 0; d < DH; d++) {
                    sQr[threadIdx.x][d] = Qr[nSD + q_off + d];
                    sQi[threadIdx.x][d] = Qi[nSD + q_off + d];
                    sdOr[threadIdx.x][d] = dOr[nSD + q_off + d];
                    sdOi[threadIdx.x][d] = dOi[nSD + q_off + d];
                    sOr[threadIdx.x][d] = Or[nSD + q_off + d];
                    sOi[threadIdx.x][d] = Oi[nSD + q_off + d];
                }
                sLse[threadIdx.x] = Lse[n * S + q_row_load];
            } else {
                #pragma unroll
                for (int d = 0; d < DH; d++) {
                    sQr[threadIdx.x][d] = 0.0f;
                    sQi[threadIdx.x][d] = 0.0f;
                    sdOr[threadIdx.x][d] = 0.0f;
                    sdOi[threadIdx.x][d] = 0.0f;
                    sOr[threadIdx.x][d] = 0.0f;
                    sOi[threadIdx.x][d] = 0.0f;
                }
                sLse[threadIdx.x] = -FLT_MAX;
            }
        }
        __syncthreads();

        if (valid_k) {
            const int br_actual = min(BR, S - q_start);

            for (int i = 0; i < br_actual; i++) {
                // 重计算 score
                float re_s = 0.0f, im_s = 0.0f;
                #pragma unroll
                for (int d = 0; d < DH; d++) {
                    re_s += sQr[i][d] * kr[d] + sQi[i][d] * ki_reg[d];
                    im_s += sQi[i][d] * kr[d] - sQr[i][d] * ki_reg[d];
                }
                re_s *= scale;
                im_s *= scale;

                // 重计算 p_ij, cos, sin
                float p = expf(re_s - sLse[i]);
                float c = cosf(im_s);
                float s = sinf(im_s);

                // ---- dV ----
                // dVr_jd += p * (c * dOr_id + s * dOi_id)
                // dVi_jd += p * (-s * dOr_id + c * dOi_id)
                #pragma unroll
                for (int d = 0; d < DH; d++) {
                    dv_r[d] += p * (c * sdOr[i][d] + s * sdOi[i][d]);
                    dv_i[d] += p * (-s * sdOr[i][d] + c * sdOi[i][d]);
                }

                // ---- dp (标量) ----
                // dp_ij = Σ_d (dOr·Ar + dOi·Ai)
                //   其中 Ar = c*Vr - s*Vi, Ai = c*Vi + s*Vr
                float dp = 0.0f;
                #pragma unroll
                for (int d = 0; d < DH; d++) {
                    float ar = c * vr[d] - s * vi_reg[d];
                    float ai = c * vi_reg[d] + s * vr[d];
                    dp += sdOr[i][d] * ar + sdOi[i][d] * ai;
                }

                // ---- D_i (softmax 校正项) ----
                // D_i = Σ_d (dOr·Or + dOi·Oi)
                float D_val = 0.0f;
                #pragma unroll
                for (int d = 0; d < DH; d++) {
                    D_val += sdOr[i][d] * sOr[i][d] + sdOi[i][d] * sOi[i][d];
                }

                // dre_s = p * (dp - D_i)
                float dre_s = p * (dp - D_val);

                // ---- dim_s (虚部梯度) ----
                // g_c = p * Σ_d (dOr·Vr + dOi·Vi)
                // g_s = p * Σ_d (-dOr·Vi + dOi·Vr)
                // dim_s = -s*g_c + c*g_s
                float g_c = 0.0f, g_s = 0.0f;
                #pragma unroll
                for (int d = 0; d < DH; d++) {
                    g_c += sdOr[i][d] * vr[d] + sdOi[i][d] * vi_reg[d];
                    g_s += -sdOr[i][d] * vi_reg[d] + sdOi[i][d] * vr[d];
                }
                g_c *= p;
                g_s *= p;
                float dim_s = -s * g_c + c * g_s;

                // ---- dK ----
                // dKr_jd += (dre_s * Qr_id + dim_s * Qi_id) * scale
                // dKi_jd += (dre_s * Qi_id - dim_s * Qr_id) * scale
                #pragma unroll
                for (int d = 0; d < DH; d++) {
                    dk_r[d] += (dre_s * sQr[i][d] + dim_s * sQi[i][d]) * scale;
                    dk_i[d] += (dre_s * sQi[i][d] - dim_s * sQr[i][d]) * scale;
                }

                // ---- dQ (通过 atomicAdd, 因为多个 K block 会写同一个 Q 行) ----
                // dQr_id += (dre_s * Kr_jd - dim_s * Ki_jd) * scale
                // dQi_id += (dre_s * Ki_jd + dim_s * Kr_jd) * scale
                const int q_global = q_start + i;
                if (q_global < S) {
                    const int q_off = nSD + q_global * DH;
                    #pragma unroll
                    for (int d = 0; d < DH; d++) {
                        atomicAdd(&dQr[q_off + d],
                                  (dre_s * kr[d] - dim_s * ki_reg[d]) * scale);
                        atomicAdd(&dQi[q_off + d],
                                  (dre_s * ki_reg[d] + dim_s * kr[d]) * scale);
                    }
                }
            }
        }
        __syncthreads();
    }

    // 写回 dK, dV (每个线程负责一行, 无竞争)
    if (valid_k) {
        const int k_off = nSD + k_row * DH;
        #pragma unroll
        for (int d = 0; d < DH; d++) {
            dKr[k_off + d] = dk_r[d];
            dKi[k_off + d] = dk_i[d];
            dVr[k_off + d] = dv_r[d];
            dVi[k_off + d] = dv_i[d];
        }
    }
}


// =====================================================================
//  C++ Wrapper Functions (PyTorch 接口)
// =====================================================================

std::vector<torch::Tensor> complex_cartesian_attn_fwd_cuda(
    torch::Tensor Qr, torch::Tensor Qi,
    torch::Tensor Kr, torch::Tensor Ki,
    torch::Tensor Vr, torch::Tensor Vi,
    float scale
) {
    // 输入: (N, S, Dh), N = B*H
    const int N = Qr.size(0);
    const int S = Qr.size(1);
    const int Dh = Qr.size(2);

    TORCH_CHECK(Dh == DH, "Head dim must be ", DH, ", got ", Dh);

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(Qr.device());
    auto Or = torch::zeros({N, S, Dh}, opts);
    auto Oi = torch::zeros({N, S, Dh}, opts);
    auto Lse = torch::zeros({N, S}, opts);

    dim3 grid((S + BR - 1) / BR, N);
    dim3 block(BR);

    complex_cartesian_attn_fwd_kernel<<<grid, block>>>(
        Qr.data_ptr<float>(), Qi.data_ptr<float>(),
        Kr.data_ptr<float>(), Ki.data_ptr<float>(),
        Vr.data_ptr<float>(), Vi.data_ptr<float>(),
        Or.data_ptr<float>(), Oi.data_ptr<float>(),
        Lse.data_ptr<float>(),
        S, scale
    );

    return {Or, Oi, Lse};
}


std::vector<torch::Tensor> complex_cartesian_attn_bwd_cuda(
    torch::Tensor Qr, torch::Tensor Qi,
    torch::Tensor Kr, torch::Tensor Ki,
    torch::Tensor Vr, torch::Tensor Vi,
    torch::Tensor Or, torch::Tensor Oi,
    torch::Tensor Lse,
    torch::Tensor dOr, torch::Tensor dOi,
    float scale
) {
    const int N = Qr.size(0);
    const int S = Qr.size(1);
    const int Dh = Qr.size(2);

    TORCH_CHECK(Dh == DH, "Head dim must be ", DH, ", got ", Dh);

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(Qr.device());
    // dQ 需要 atomicAdd, 必须初始化为 0
    auto dQr = torch::zeros({N, S, Dh}, opts);
    auto dQi = torch::zeros({N, S, Dh}, opts);
    // dK, dV 由每个线程独立写入, 不需要初始化
    auto dKr = torch::empty({N, S, Dh}, opts);
    auto dKi = torch::empty({N, S, Dh}, opts);
    auto dVr = torch::empty({N, S, Dh}, opts);
    auto dVi = torch::empty({N, S, Dh}, opts);

    dim3 grid((S + BC - 1) / BC, N);
    dim3 block(BC);

    // 共享内存大小: Q(2×BR×DH) + dO(2×BR×DH) + O(2×BR×DH) + Lse(BR)
    // = 6×BR×DH×4 + BR×4 = 6×64×16×4 + 256 = 24,832 bytes
    // 无需动态分配, 静态声明足够

    complex_cartesian_attn_bwd_kernel<<<grid, block>>>(
        Qr.data_ptr<float>(), Qi.data_ptr<float>(),
        Kr.data_ptr<float>(), Ki.data_ptr<float>(),
        Vr.data_ptr<float>(), Vi.data_ptr<float>(),
        Or.data_ptr<float>(), Oi.data_ptr<float>(),
        Lse.data_ptr<float>(),
        dOr.data_ptr<float>(), dOi.data_ptr<float>(),
        dQr.data_ptr<float>(), dQi.data_ptr<float>(),
        dKr.data_ptr<float>(), dKi.data_ptr<float>(),
        dVr.data_ptr<float>(), dVi.data_ptr<float>(),
        S, scale
    );

    return {dQr, dQi, dKr, dKi, dVr, dVi};
}


// =====================================================================
//  PyBind11 绑定
// =====================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &complex_cartesian_attn_fwd_cuda,
          "Complex Cartesian Attention Forward (CUDA)");
    m.def("backward", &complex_cartesian_attn_bwd_cuda,
          "Complex Cartesian Attention Backward (CUDA)");
}
