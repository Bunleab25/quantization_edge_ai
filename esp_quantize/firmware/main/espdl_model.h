/**
 * @file espdl_model.h
 * @brief Pure C Inference Header for AdvancedMNISTCNN on ESP32-S3
 */

#pragma once

#include <stdint.h>
#include <stddef.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Microsecond-level layer execution timing profile captured via esp_timer_get_time()
 */
typedef struct {
    int64_t prep_us;
    int64_t stage1_conv1_us;
    int64_t stage1_conv2_us;
    int64_t stage1_pool_us;
    int64_t stage2_conv1_us;
    int64_t stage2_conv2_us;
    int64_t stage2_pool_us;
    int64_t stage3_conv1_us;
    int64_t stage3_conv2_us;
    int64_t gap_us;
    int64_t classifier_us;
    int64_t softmax_us;
    int64_t total_us;
} espdl_timing_profile_t;

/**
 * @brief Initializes the inference engine, LUT tables, and tensor buffers.
 * @return ESP_OK on success.
 */
esp_err_t espdl_model_init(void);

/**
 * @brief Runs INT8 forward inference with full microsecond-level profiling.
 * 
 * @param raw_pixels_784 Array of 784 bytes (28x28 grayscale, 0..255).
 * @param out_confidence Output pointer for predicted class probability (0.0 to 1.0).
 * @param out_profile Output pointer for fine-grained per-layer microsecond timing breakdown.
 * @return Predicted digit (0 to 9).
 */
int espdl_model_predict_profile(const uint8_t *raw_pixels_784, float *out_confidence, espdl_timing_profile_t *out_profile);

/**
 * @brief Standard prediction interface returning total latency in microseconds.
 */
int espdl_model_predict(const uint8_t *raw_pixels_784, float *out_confidence, int64_t *out_latency_us);

/**
 * @brief Prints an ASCII visual representation of the 28x28 input to UART.
 * @param raw_pixels_784 Array of 784 bytes.
 */
void espdl_print_ascii_art(const uint8_t *raw_pixels_784);

#ifdef __cplusplus
}
#endif
