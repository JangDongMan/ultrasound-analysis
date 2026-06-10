#ifndef SKIN_CNN_H
#define SKIN_CNN_H

/**
 * skin_cnn_infer
 *
 * ADC 원시 데이터로부터 진피/근막 두께를 예측합니다.
 *
 * @param adc        uint8 ADC 샘플 배열 (최소 TRIM_START+TRIM_COUNT = 2450개)
 * @param adc_len    배열 길이
 * @param dermis_mm  [출력] 진피 두께 (mm)
 * @param fascia_mm  [출력] 근막 두께 (mm)
 * @return 0: 성공, -1: 입력 길이 부족
 */
int skin_cnn_infer(const unsigned char *adc, int adc_len,
                   float *dermis_mm, float *fascia_mm);

#endif /* SKIN_CNN_H */
