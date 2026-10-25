/**
 * @file espdl_model.c
 * @brief High-Performance Pure C Inference Engine with Microsecond Profiling for ESP32-S3
 */

#include "espdl_model.h"
#include "app_config.h"
#include "model_data.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "esp_timer.h"
#include "esp_log.h"
#include "esp_task_wdt.h"

static const char *TAG = "ESPDL_MODEL";

// 256-byte Fast Normalization LUT: maps uint8 [0..255] -> quantized input tensor
static uint8_t g_pixel_lut[256];

// Static Ping-Pong RAM Buffers (Peak memory: 2 x 25 KB = ~50 KB SRAM)
// Statically allocated in internal DRAM to prevent heap fragmentation
#define MAX_FEATURE_MAP_SIZE (28 * 28 * 32) // 25,088 bytes
static uint8_t s_buffer_a[MAX_FEATURE_MAP_SIZE] __attribute__((aligned(16)));
static uint8_t s_buffer_b[MAX_FEATURE_MAP_SIZE] __attribute__((aligned(16)));
static uint8_t s_input_buf[MNIST_IMAGE_SIZE]    __attribute__((aligned(16)));

static inline uint8_t clamp_relu_u8(int32_t val, int32_t zp) {
    if (val < zp) val = zp;
    if (val > 255) val = 255;
    return (uint8_t)val;
}

/**
 * @brief Conv2D + ReLU with 3x3 kernel, padding=1, per-channel symmetric weights.
 * Weights layout: (out_ch, 3, 3, in_ch)
 */
static void conv3x3_relu(
    const uint8_t *input, int in_w, int in_h, int in_ch, int in_zp,
    const int8_t *weights, const int32_t *bias, const float *eff_scale,
    int out_ch, int out_zp,
    uint8_t *output
) {
    for (int oc = 0; oc < out_ch; ++oc) {
        const int32_t b_val = bias ? bias[oc] : 0;
        const float s_val = eff_scale[oc];
        const int8_t *w_oc = &weights[oc * 9 * in_ch];

        for (int y = 0; y < in_h; ++y) {
            for (int x = 0; x < in_w; ++x) {
                int32_t sum = b_val;

                for (int dy = -1; dy <= 1; ++dy) {
                    int py = y + dy;
                    if (py < 0 || py >= in_h) continue;

                    const uint8_t *in_row = &input[py * in_w * in_ch];
                    const int8_t *w_row = &w_oc[(dy + 1) * 3 * in_ch];

                    for (int dx = -1; dx <= 1; ++dx) {
                        int px = x + dx;
                        if (px < 0 || px >= in_w) continue;

                        const uint8_t *in_px = &in_row[px * in_ch];
                        const int8_t *w_px = &w_row[(dx + 1) * in_ch];

                        if (in_zp == 0) {
                            for (int ic = 0; ic < in_ch; ++ic) {
                                sum += ((int32_t)in_px[ic]) * ((int32_t)w_px[ic]);
                            }
                        } else {
                            for (int ic = 0; ic < in_ch; ++ic) {
                                sum += ((int32_t)in_px[ic] - in_zp) * ((int32_t)w_px[ic]);
                            }
                        }
                    }
                }

                int32_t rounded = (int32_t)roundf((float)sum * s_val) + out_zp;
                output[(y * in_w + x) * out_ch + oc] = clamp_relu_u8(rounded, out_zp);
            }
        }
    }
}

/**
 * @brief 2x2 Max Pooling with stride 2.
 */
static void maxpool2x2(const uint8_t *input, int in_w, int in_h, int ch, uint8_t *output) {
    int out_w = in_w / 2;
    int out_h = in_h / 2;

    for (int y = 0; y < out_h; ++y) {
        for (int x = 0; x < out_w; ++x) {
            for (int c = 0; c < ch; ++c) {
                uint8_t m0 = input[((2 * y + 0) * in_w + (2 * x + 0)) * ch + c];
                uint8_t m1 = input[((2 * y + 0) * in_w + (2 * x + 1)) * ch + c];
                uint8_t m2 = input[((2 * y + 1) * in_w + (2 * x + 0)) * ch + c];
                uint8_t m3 = input[((2 * y + 1) * in_w + (2 * x + 1)) * ch + c];

                uint8_t max_val = m0;
                if (m1 > max_val) max_val = m1;
                if (m2 > max_val) max_val = m2;
                if (m3 > max_val) max_val = m3;

                output[(y * out_w + x) * ch + c] = max_val;
            }
        }
    }
}

/**
 * @brief Global Average Pooling over 7x7 spatial dimensions for 128 channels.
 */
static void global_avgpool_7x7(const uint8_t *input, int ch, uint8_t *output) {
    for (int c = 0; c < ch; ++c) {
        int32_t sum = 0;
        for (int i = 0; i < 49; ++i) {
            sum += (int32_t)input[i * ch + c];
        }
        output[c] = (uint8_t)(sum / 49);
    }
}

/**
 * @brief Fully Connected Linear layer: 128 -> 10 classes with high precision float scale.
 */
static void classifier_head(const uint8_t *gap_in, int in_zp, float *out_logits) {
    for (int c = 0; c < MODEL_NUM_CLASSES; ++c) {
        int32_t sum = 0;
        const int8_t *w_c = &classifier_weights[c * 128];

        for (int i = 0; i < 128; ++i) {
            sum += ((int32_t)gap_in[i] - in_zp) * ((int32_t)w_c[i]);
        }
        out_logits[c] = ((float)sum * classifier_eff_scale[c]) + classifier_bias_float[c];
    }
}

/**
 * @brief Softmax over 10 output logits.
 */
static void softmax_10(const float *logits, float *probs) {
    float max_l = logits[0];
    for (int i = 1; i < MODEL_NUM_CLASSES; ++i) {
        if (logits[i] > max_l) max_l = logits[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < MODEL_NUM_CLASSES; ++i) {
        probs[i] = expf(logits[i] - max_l);
        sum += probs[i];
    }
    for (int i = 0; i < MODEL_NUM_CLASSES; ++i) {
        probs[i] /= sum;
    }
}

esp_err_t espdl_model_init(void) {
    ESP_LOGI(TAG, "Initializing AdvancedMNISTCNN Pure C Inference Engine...");

    // Build the 256-entry Fast Normalization LUT
    for (int p = 0; p < 256; ++p) {
        float norm = (((float)p / 255.0f) - MNIST_NORM_MEAN) / MNIST_NORM_STD;
        int32_t q = (int32_t)roundf(norm / MODEL_INPUT_SCALE) + MODEL_INPUT_ZP;
        if (q < 0)   q = 0;
        if (q > 255) q = 255;
        g_pixel_lut[p] = (uint8_t)q;
    }

    ESP_LOGI(TAG, "Normalization LUT built (256 bytes). Model weights in Flash: %d bytes (%.1f KB)",
             TOTAL_MODEL_WEIGHT_BYTES, (float)TOTAL_MODEL_WEIGHT_BYTES / 1024.0f);
    ESP_LOGI(TAG, "Static RAM allocation: Ping-Pong Buffers = %d bytes (~50 KB)", MAX_FEATURE_MAP_SIZE * 2);
    return ESP_OK;
}

int espdl_model_predict_profile(const uint8_t *raw_pixels_784, float *out_confidence, espdl_timing_profile_t *out_profile) {
    int64_t t0 = esp_timer_get_time();

    // 1. Preprocess: 256-byte LUT conversion [0..255] -> INT8 input buffer
    for (int i = 0; i < MNIST_IMAGE_SIZE; ++i) {
        s_input_buf[i] = g_pixel_lut[raw_pixels_784[i]];
    }
    int64_t t_prep = esp_timer_get_time();

    // 2. Stage 1: Dual 32-ch Conv + Pool (28x28 -> 14x14)
    conv3x3_relu(s_input_buf, 28, 28, 1, S1_C1_IN_ZP,
                 s1_c1_weights, s1_c1_bias, s1_c1_eff_scale,
                 32, S1_C1_OUT_ZP, s_buffer_a);
    int64_t t_s1_c1 = esp_timer_get_time();

    conv3x3_relu(s_buffer_a, 28, 28, 32, S1_C2_IN_ZP,
                 s1_c2_weights, s1_c2_bias, s1_c2_eff_scale,
                 32, S1_C2_OUT_ZP, s_buffer_b);
    int64_t t_s1_c2 = esp_timer_get_time();

    maxpool2x2(s_buffer_b, 28, 28, 32, s_buffer_a);
    int64_t t_s1_p = esp_timer_get_time();

    // 3. Stage 2: Dual 64-ch Conv + Pool (14x14 -> 7x7)
    conv3x3_relu(s_buffer_a, 14, 14, 32, S2_C1_IN_ZP,
                 s2_c1_weights, s2_c1_bias, s2_c1_eff_scale,
                 64, S2_C1_OUT_ZP, s_buffer_b);
    int64_t t_s2_c1 = esp_timer_get_time();

    conv3x3_relu(s_buffer_b, 14, 14, 64, S2_C2_IN_ZP,
                 s2_c2_weights, s2_c2_bias, s2_c2_eff_scale,
                 64, S2_C2_OUT_ZP, s_buffer_a);
    int64_t t_s2_c2 = esp_timer_get_time();

    maxpool2x2(s_buffer_a, 14, 14, 64, s_buffer_b);
    int64_t t_s2_p = esp_timer_get_time();

    // 4. Stage 3: Dual 128-ch Conv (Preserves 7x7)
    conv3x3_relu(s_buffer_b, 7, 7, 64, S3_C1_IN_ZP,
                 s3_c1_weights, s3_c1_bias, s3_c1_eff_scale,
                 128, S3_C1_OUT_ZP, s_buffer_a);
    int64_t t_s3_c1 = esp_timer_get_time();

    conv3x3_relu(s_buffer_a, 7, 7, 128, S3_C2_IN_ZP,
                 s3_c2_weights, s3_c2_bias, s3_c2_eff_scale,
                 128, S3_C2_OUT_ZP, s_buffer_b);
    int64_t t_s3_c2 = esp_timer_get_time();

    // 5. Global Average Pooling (7x7x128 -> 128)
    uint8_t gap_out[128];
    global_avgpool_7x7(s_buffer_b, 128, gap_out);
    int64_t t_gap = esp_timer_get_time();

    // 6. Classifier Head (128 -> 10 Logits)
    float logits[MODEL_NUM_CLASSES];
    classifier_head(gap_out, CLASSIFIER_IN_ZP, logits);
    int64_t t_cls = esp_timer_get_time();

    // 7. Softmax Probabilities
    float probs[MODEL_NUM_CLASSES];
    softmax_10(logits, probs);
    int64_t t_end = esp_timer_get_time();

    if (out_profile) {
        out_profile->prep_us         = t_prep - t0;
        out_profile->stage1_conv1_us = t_s1_c1 - t_prep;
        out_profile->stage1_conv2_us = t_s1_c2 - t_s1_c1;
        out_profile->stage1_pool_us  = t_s1_p - t_s1_c2;
        out_profile->stage2_conv1_us = t_s2_c1 - t_s1_p;
        out_profile->stage2_conv2_us = t_s2_c2 - t_s2_c1;
        out_profile->stage2_pool_us  = t_s2_p - t_s2_c2;
        out_profile->stage3_conv1_us = t_s3_c1 - t_s2_p;
        out_profile->stage3_conv2_us = t_s3_c2 - t_s3_c1;
        out_profile->gap_us          = t_gap - t_s3_c2;
        out_profile->classifier_us   = t_cls - t_gap;
        out_profile->softmax_us      = t_end - t_cls;
        out_profile->total_us        = t_end - t0;
    }

    // Find predicted class & max confidence
    int best_class = 0;
    float max_p = probs[0];
    for (int i = 1; i < MODEL_NUM_CLASSES; ++i) {
        if (probs[i] > max_p) {
            max_p = probs[i];
            best_class = i;
        }
    }

    if (out_confidence) {
        *out_confidence = max_p;
    }

    return best_class;
}

int espdl_model_predict(const uint8_t *raw_pixels_784, float *out_confidence, int64_t *out_latency_us) {
    espdl_timing_profile_t prof;
    int pred = espdl_model_predict_profile(raw_pixels_784, out_confidence, &prof);
    if (out_latency_us) {
        *out_latency_us = prof.total_us;
    }
    return pred;
}

void espdl_print_ascii_art(const uint8_t *raw_pixels_784) {
    static const char *ramp = " .:-=+*#%@";
    int ramp_len = strlen(ramp);

    printf("\n--- 28x28 Input Preview ---\n");
    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            uint8_t p = raw_pixels_784[y * 28 + x];
            int idx = (p * (ramp_len - 1)) / 255;
            putchar(ramp[idx]);
            putchar(' '); // 2:1 terminal aspect ratio correction
        }
        putchar('\n');
    }
    printf("---------------------------\n");
}
