#ifndef SKIN_NPU_H
#define SKIN_NPU_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * skin_npu_init
 *   TFLite Micro 인터프리터 및 NXP NPU 초기화
 *   시스템 시작 시 1회 호출
 * @return 0: 성공, -1: 실패
 */
int skin_npu_init(void);

/**
 * skin_npu_infer
 *   ADC 원시 데이터 → 진피/근막 두께 예측 (NPU 가속)
 *
 * @param adc        uint8 ADC 샘플 배열
 *                   - 1250샘플: 그대로 사용 (이미 트림된 파일)
 *                   - 2500샘플 이상: [1200:2450] 자동 트림
 * @param adc_len    배열 길이
 * @param dermis_mm  [출력] 진피 두께 (mm)
 * @param fascia_mm  [출력] 근막 두께 (mm)
 * @return 0: 성공, -1: 입력 오류, -2: 추론 실패
 */
int skin_npu_infer(const unsigned char *adc, int adc_len,
                   float *dermis_mm, float *fascia_mm);

#ifdef __cplusplus
}
#endif

#endif /* SKIN_NPU_H */
