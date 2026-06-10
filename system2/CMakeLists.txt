# CMakeLists.txt — NXP MCUXpresso / eIQ SDK 빌드
# 타겟: FRDM-MCXN947 (Cortex-M33 + N1-16 NPU)
#
# 사용법 (MCUXpresso IDE):
#   File → Import → Existing Projects → 이 디렉토리 선택
#
# eIQ SDK 경로는 SDK_PATH 환경변수 또는 아래 경로를 수정하세요.

cmake_minimum_required(VERSION 3.20)
project(skin_npu C CXX)

set(CMAKE_C_STANDARD   99)
set(CMAKE_CXX_STANDARD 17)

# ── NXP eIQ SDK 경로 ──────────────────────────────────────────────────────
# MCUXpresso SDK 설치 후 실제 경로로 수정
set(SDK_PATH    "$ENV{SDK_PATH}"  CACHE PATH "NXP MCU SDK 루트")
set(EIQ_PATH    "${SDK_PATH}/middleware/eiq"   CACHE PATH "eIQ 미들웨어")
set(TFLM_PATH   "${EIQ_PATH}/tensorflow-lite-micro" CACHE PATH "TFLite Micro")
set(NPU_PATH    "${EIQ_PATH}/tensorflow-lite-micro/third_party/ethos_u" CACHE PATH "NPU 드라이버")

# ── NPU 활성화 옵션 ───────────────────────────────────────────────────────
option(ENABLE_NPU "NXP N1-16 NPU 가속 사용" ON)

# ── 소스 파일 ─────────────────────────────────────────────────────────────
set(SOURCES
    skin_npu.cc
    # 애플리케이션 진입점 (별도 MCU 프로젝트에서 제공)
    # main_mcu.c
)

add_library(skin_npu STATIC ${SOURCES})

target_include_directories(skin_npu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}          # skin_npu.h, model_int8.h
    ${TFLM_PATH}
    ${TFLM_PATH}/third_party/flatbuffers/include
    ${TFLM_PATH}/third_party/gemmlowp
    ${TFLM_PATH}/third_party/ruy
)

target_link_directories(skin_npu PUBLIC
    ${TFLM_PATH}/lib
)

target_link_libraries(skin_npu PUBLIC
    tensorflow-lite-micro
    m
)

# ── NPU 드라이버 연결 ─────────────────────────────────────────────────────
if(ENABLE_NPU)
    target_compile_definitions(skin_npu PUBLIC ENABLE_NPU)
    target_include_directories(skin_npu PUBLIC ${NPU_PATH}/core_driver/include)
    target_link_libraries(skin_npu PUBLIC ethosu_core_driver)
endif()

# ── MCU 컴파일 플래그 ─────────────────────────────────────────────────────
target_compile_options(skin_npu PRIVATE
    -mcpu=cortex-m33
    -mthumb
    -mfpu=fpv5-sp-d16
    -mfloat-abi=hard
    -Os                  # 크기 최적화
    -ffunction-sections
    -fdata-sections
)

# ── 메모리 사용량 출력 ────────────────────────────────────────────────────
add_custom_command(TARGET skin_npu POST_BUILD
    COMMAND ${CMAKE_SIZE} $<TARGET_FILE:skin_npu>
    COMMENT "메모리 사용량"
)
