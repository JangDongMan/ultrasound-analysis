#include "skin_cnn.h"
#include "model_weights.h"
#include <math.h>
#include <string.h>

/* 스크래치 버퍼 — 레이어 간 활성화값 저장 (channel-first: [ch, len]) */
/* 최대 크기: F4 * L3_OUT = 64 * 157 = 10048 */
#define SCRATCH_SIZE (F4 * L3_OUT)
static float buf_a[SCRATCH_SIZE];
static float buf_b[SCRATCH_SIZE];


/* ── 1D Convolution + ReLU ──────────────────────────────────────────────────
 * weight 배열 순서: [out_ch][in_ch][k]  (C-contiguous, row-major)
 * output 배열 순서: [out_ch][out_len]
 */
static void conv1d_relu(const float *in,  int in_ch,  int in_len,
                              float *out, int out_ch, int out_len,
                        const float *weight, const float *bias,
                        int kernel, int stride, int padding)
{
    for (int oc = 0; oc < out_ch; oc++) {
        const float *w_oc = weight + oc * in_ch * kernel;
        for (int t = 0; t < out_len; t++) {
            float sum = bias[oc];
            int t0    = t * stride - padding;
            for (int ic = 0; ic < in_ch; ic++) {
                const float *in_ic = in + ic * in_len;
                const float *w_ic  = w_oc + ic * kernel;
                for (int k = 0; k < kernel; k++) {
                    int ts = t0 + k;
                    if (ts >= 0 && ts < in_len)
                        sum += in_ic[ts] * w_ic[k];
                }
            }
            out[oc * out_len + t] = sum > 0.0f ? sum : 0.0f;  /* ReLU */
        }
    }
}


/* ── Adaptive Average Pooling (전체 길이 평균) ─────────────────────────────*/
static void adaptive_avg_pool(const float *in, int ch, int in_len, float *out)
{
    for (int c = 0; c < ch; c++) {
        float sum = 0.0f;
        const float *p = in + c * in_len;
        for (int t = 0; t < in_len; t++) sum += p[t];
        out[c] = sum / (float)in_len;
    }
}


/* ── Fully Connected + ReLU ─────────────────────────────────────────────── */
static void linear_relu(const float *in,  int in_sz,
                              float *out, int out_sz,
                        const float *weight, const float *bias)
{
    for (int o = 0; o < out_sz; o++) {
        float sum = bias[o];
        const float *w = weight + o * in_sz;
        for (int i = 0; i < in_sz; i++) sum += in[i] * w[i];
        out[o] = sum > 0.0f ? sum : 0.0f;
    }
}


/* ── Fully Connected (출력층, 비선형 없음) ──────────────────────────────── */
static void linear(const float *in,  int in_sz,
                         float *out, int out_sz,
                   const float *weight, const float *bias)
{
    for (int o = 0; o < out_sz; o++) {
        float sum = bias[o];
        const float *w = weight + o * in_sz;
        for (int i = 0; i < in_sz; i++) sum += in[i] * w[i];
        out[o] = sum;
    }
}


/* ── 공개 API ───────────────────────────────────────────────────────────── */
int skin_cnn_infer(const unsigned char *adc, int adc_len,
                   float *dermis_mm, float *fascia_mm)
{
    /* Python load_adc 로직과 동일:
     *   샘플 수 > TRIM_COUNT  → adc[TRIM_START : TRIM_START+TRIM_COUNT]
     *   샘플 수 == TRIM_COUNT → adc[0 : TRIM_COUNT]  (이미 트림된 파일)
     *   샘플 수 < TRIM_COUNT  → 오류 */
    int offset = 0;
    if (adc_len > TRIM_COUNT) {
        if (adc_len < TRIM_START + TRIM_COUNT) return -1;
        offset = TRIM_START;
    } else if (adc_len < TRIM_COUNT) {
        return -1;
    }

    /* 전처리: uint8 ADC → float [-1, +1]  ((x - 128) / 128) */
    float input[TRIM_COUNT];
    for (int i = 0; i < TRIM_COUNT; i++)
        input[i] = ((float)adc[offset + i] - 128.0f) / 128.0f;

    /* Encoder
     *   Conv1: (1, 1250) → (F1, L1_OUT=625)   k=15 s=2 p=7
     *   Conv2: (F1, 625) → (F2, L2_OUT=313)   k=7  s=2 p=3
     *   Conv3: (F2, 313) → (F4, L3_OUT=157)   k=5  s=2 p=2
     *   Conv4: (F4, 157) → (F4, L4_OUT=79)    k=5  s=2 p=2
     */
    conv1d_relu(input, 1,  TRIM_COUNT, buf_a, F1, L1_OUT, W_CONV1, B_CONV1, 15, 2, 7);
    conv1d_relu(buf_a, F1, L1_OUT,     buf_b, F2, L2_OUT, W_CONV2, B_CONV2,  7, 2, 3);
    conv1d_relu(buf_b, F2, L2_OUT,     buf_a, F4, L3_OUT, W_CONV3, B_CONV3,  5, 2, 2);
    conv1d_relu(buf_a, F4, L3_OUT,     buf_b, F4, L4_OUT, W_CONV4, B_CONV4,  5, 2, 2);

    /* AdaptiveAvgPool1d(1): (F4, L4_OUT) → (F4,) */
    float pool[F4];
    adaptive_avg_pool(buf_b, F4, L4_OUT, pool);

    /* Head
     *   Linear(64→32) + ReLU
     *   Linear(32→2)
     */
    float fc1_out[F2];   /* F2 = 32 */
    linear_relu(pool, F4, fc1_out, F2, W_FC1, B_FC1);

    float out[2];
    linear(fc1_out, F2, out, 2, W_FC2, B_FC2);

    *dermis_mm = out[0];
    *fascia_mm = out[1];
    return 0;
}
