/**
 * skin_npu.cc
 * TFLite Micro + NXP MCXNx4x N1-16 NPU 추론 엔진
 *
 * 빌드 환경: NXP MCUXpresso + eIQ Toolkit SDK
 * 타겟:      FRDM-MCXN947 (Dual Cortex-M33 @ 150MHz, N1-16 NPU)
 */

#include "skin_npu.h"
#include "model_int8.h"

/* ── TFLite Micro 헤더 (NXP eIQ SDK에 포함) ────────────────────────────── */
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

/* ── NXP Neutron NPU 드라이버 (eIQ SDK에 포함) ──────────────────────────── */
#ifdef ENABLE_NPU
#include "ethosu_driver.h"          /* ARM Ethos-U 호환 드라이버 */
#include "tensorflow/lite/micro/kernels/ethos_u/ethos_u_microkernel.h"
#endif

/* ── 상수 ───────────────────────────────────────────────────────────────── */
#define TRIM_START   1200
#define TRIM_COUNT   1250

/*
 * Tensor Arena (SRAM)
 *   INT8 모델 최대 활성화: 64ch × 157 = 10,048 bytes
 *   TFLite Micro 런타임 오버헤드: ~15 KB
 *   안전 마진 포함 → 28 KB
 */
#define TENSOR_ARENA_SIZE (28 * 1024)
static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

/* ── 전역 TFLite 객체 ───────────────────────────────────────────────────── */
static const tflite::Model        *s_model       = nullptr;
static tflite::MicroInterpreter   *s_interpreter = nullptr;
static TfLiteTensor               *s_input       = nullptr;
static TfLiteTensor               *s_output      = nullptr;
static bool                        s_initialized = false;

/* OpResolver — 모델에 필요한 연산만 등록 (Flash 절약) */
static tflite::MicroMutableOpResolver<8> s_resolver;


/* ── NPU 초기화 ─────────────────────────────────────────────────────────── */
extern "C" int skin_npu_init(void)
{
    if (s_initialized) return 0;

    /* 모델 로드 및 검증 */
    s_model = tflite::GetModel(model_int8_data);
    if (s_model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("TFLite 스키마 버전 불일치: %d != %d",
                    s_model->version(), TFLITE_SCHEMA_VERSION);
        return -1;
    }

    /* ── 연산자 등록 ──────────────────────────────────────────────────────
     * SkinCNN 구조에 필요한 ops:
     *   Conv1D → TFLite 내부에서 Conv2D (height=1) 로 표현
     *   BatchNorm → 변환 시 Conv에 융합됨
     *   AdaptiveAvgPool1d(1) → Mean
     *   Linear → FullyConnected
     */
#ifdef ENABLE_NPU
    /* NPU가 지원하는 ops는 Ethos-U 커널이 자동 위임 */
    s_resolver.AddEthosU();
#endif
    s_resolver.AddConv2D();
    s_resolver.AddReshape();
    s_resolver.AddRelu();
    s_resolver.AddMean();
    s_resolver.AddFullyConnected();
    s_resolver.AddQuantize();
    s_resolver.AddDequantize();
    s_resolver.AddPad();              /* onnx2tf 변환 시 추가될 수 있음 */

    /* 인터프리터 생성 */
    static tflite::MicroInterpreter interpreter(
        s_model, s_resolver, tensor_arena, TENSOR_ARENA_SIZE);
    s_interpreter = &interpreter;

    if (s_interpreter->AllocateTensors() != kTfLiteOk) {
        MicroPrintf("AllocateTensors 실패");
        return -1;
    }

    s_input  = s_interpreter->input(0);
    s_output = s_interpreter->output(0);

    /* 실제 사용된 Tensor Arena 크기 출력 (디버그) */
    MicroPrintf("Tensor Arena 사용: %d / %d bytes",
                (int)s_interpreter->arena_used_bytes(), TENSOR_ARENA_SIZE);

    s_initialized = true;
    return 0;
}


/* ── 추론 ───────────────────────────────────────────────────────────────── */
extern "C" int skin_npu_infer(const unsigned char *adc, int adc_len,
                               float *dermis_mm, float *fascia_mm)
{
    if (!s_initialized) return -1;

    /* 트림 오프셋 결정 (Python load_adc 로직과 동일) */
    int offset = 0;
    if (adc_len > TRIM_COUNT) {
        if (adc_len < TRIM_START + TRIM_COUNT) return -1;
        offset = TRIM_START;
    } else if (adc_len < TRIM_COUNT) {
        return -1;
    }

    /* ── 입력 텐서 채우기 ────────────────────────────────────────────────
     * 모델 입력: INT8 [-128, 127]
     * 정규화:    float = (adc - 128) / 128
     * INT8 양자화 파라미터: scale=1/128, zero_point=0  (변환기 자동 결정)
     *   → scale, zero_point는 변환 후 모델에 저장된 값 사용
     */
    const float in_scale     = s_input->params.scale;
    const int   in_zp        = s_input->params.zero_point;
    int8_t     *in_data      = s_input->data.int8;
    const int   in_elem      = s_input->bytes;  /* = TRIM_COUNT */

    for (int i = 0; i < TRIM_COUNT && i < in_elem; i++) {
        float norm = ((float)adc[offset + i] - 128.0f) / 128.0f;
        int   q    = (int)(norm / in_scale + in_zp + 0.5f);
        if (q >  127) q =  127;
        if (q < -128) q = -128;
        in_data[i] = (int8_t)q;
    }

    /* ── NPU 추론 실행 ────────────────────────────────────────────────── */
    if (s_interpreter->Invoke() != kTfLiteOk) return -2;

    /* ── 출력 역양자화 ────────────────────────────────────────────────── */
    const float out_scale = s_output->params.scale;
    const int   out_zp    = s_output->params.zero_point;
    const int8_t *out_data = s_output->data.int8;

    *dermis_mm = (out_data[0] - out_zp) * out_scale;
    *fascia_mm = (out_data[1] - out_zp) * out_scale;

    return 0;
}
