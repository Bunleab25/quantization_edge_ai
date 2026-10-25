#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "firmware/main/model_data.h"
#include "firmware/main/test_samples.h"

#define MAX_FEAT (28 * 28 * 32)
static uint8_t s_buf_a[MAX_FEAT] __attribute__((aligned(16)));
static uint8_t s_buf_b[MAX_FEAT] __attribute__((aligned(16)));
static uint8_t s_input[28 * 28]  __attribute__((aligned(16)));
static uint8_t g_pixel_lut[256];

#define MNIST_NORM_MEAN 0.1307f
#define MNIST_NORM_STD  0.3081f

static void init_lut(void) {
    for (int p = 0; p < 256; ++p) {
        float norm = (((float)p / 255.0f) - MNIST_NORM_MEAN) / MNIST_NORM_STD;
        int32_t q = (int32_t)roundf(norm / MODEL_INPUT_SCALE) + MODEL_INPUT_ZP;
        if (q < 0)   q = 0;
        if (q > 255) q = 255;
        g_pixel_lut[p] = (uint8_t)q;
    }
}

static inline uint8_t clamp_relu_u8(int32_t val, int32_t zp) {
    if (val < zp) val = zp;
    if (val > 255) val = 255;
    return (uint8_t)val;
}

// Optimized Conv3x3 + ReLU
// Weights shaped (out_ch, 3, 3, in_ch)
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

static void global_avgpool_7x7(const uint8_t *input, int ch, uint8_t *output) {
    for (int c = 0; c < ch; ++c) {
        int32_t sum = 0;
        for (int i = 0; i < 49; ++i) {
            sum += (int32_t)input[i * ch + c];
        }
        output[c] = (uint8_t)(sum / 49);
    }
}

static void classifier_head(const uint8_t *gap_in, int in_zp, float *logits) {
    for (int c = 0; c < MODEL_NUM_CLASSES; ++c) {
        int32_t sum = 0;
        const int8_t *w = &classifier_weights[c * 128];
        for (int i = 0; i < 128; ++i) {
            sum += ((int32_t)gap_in[i] - in_zp) * ((int32_t)w[i]);
        }
        logits[c] = ((float)sum * classifier_eff_scale[c]) + classifier_bias_float[c];
    }
}

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

static int predict(const uint8_t *raw_pixels_784, float *confidence) {
    for (int i = 0; i < 784; ++i) {
        s_input[i] = g_pixel_lut[raw_pixels_784[i]];
    }

    // Stage 1
    conv3x3_relu(s_input, 28, 28, 1, S1_C1_IN_ZP,
                 s1_c1_weights, s1_c1_bias, s1_c1_eff_scale,
                 32, S1_C1_OUT_ZP, s_buf_a);

    conv3x3_relu(s_buf_a, 28, 28, 32, S1_C2_IN_ZP,
                 s1_c2_weights, s1_c2_bias, s1_c2_eff_scale,
                 32, S1_C2_OUT_ZP, s_buf_b);

    maxpool2x2(s_buf_b, 28, 28, 32, s_buf_a);

    // Stage 2
    conv3x3_relu(s_buf_a, 14, 14, 32, S2_C1_IN_ZP,
                 s2_c1_weights, s2_c1_bias, s2_c1_eff_scale,
                 64, S2_C1_OUT_ZP, s_buf_b);

    conv3x3_relu(s_buf_b, 14, 14, 64, S2_C2_IN_ZP,
                 s2_c2_weights, s2_c2_bias, s2_c2_eff_scale,
                 64, S2_C2_OUT_ZP, s_buf_a);

    maxpool2x2(s_buf_a, 14, 14, 64, s_buf_b);

    // Stage 3
    conv3x3_relu(s_buf_b, 7, 7, 64, S3_C1_IN_ZP,
                 s3_c1_weights, s3_c1_bias, s3_c1_eff_scale,
                 128, S3_C1_OUT_ZP, s_buf_a);

    conv3x3_relu(s_buf_a, 7, 7, 128, S3_C2_IN_ZP,
                 s3_c2_weights, s3_c2_bias, s3_c2_eff_scale,
                 128, S3_C2_OUT_ZP, s_buf_b);

    // GAP
    uint8_t gap_out[128];
    global_avgpool_7x7(s_buf_b, 128, gap_out);

    // Head
    float logits[10];
    classifier_head(gap_out, CLASSIFIER_IN_ZP, logits);

    float probs[10];
    softmax_10(logits, probs);

    int best = 0;
    float max_p = probs[0];
    for (int i = 1; i < 10; ++i) {
        if (probs[i] > max_p) {
            max_p = probs[i];
            best = i;
        }
    }
    if (confidence) *confidence = max_p;
    return best;
}

int main(void) {
    printf("============================================================\n");
    printf("   VERIFYING FAST PURE C INFERENCE ENGINE ON PC (HOST)\n");
    printf("============================================================\n");

    init_lut();

    int passed = 0;
    for (int i = 0; i < NUM_TEST_SAMPLES; ++i) {
        float conf = 0.0f;
        clock_t t0 = clock();
        int pred = predict(TEST_DIGIT_SAMPLES[i], &conf);
        clock_t t1 = clock();
        double ms = (double)(t1 - t0) * 1000.0 / CLOCKS_PER_SEC;

        int expected = TEST_EXPECTED_LABELS[i];
        int ok = (pred == expected);
        if (ok) passed++;

        printf("Digit %d | Expected: %d | Predicted: %d | Conf: %5.1f%% | Time: %5.2f ms | %s\n",
               i, expected, pred, conf * 100.0f, ms, ok ? "PASS" : "FAIL");
    }

    printf("============================================================\n");
    printf("Summary: %d / %d Correct!\n", passed, NUM_TEST_SAMPLES);
    printf("============================================================\n");
    return (passed == NUM_TEST_SAMPLES) ? 0 : 1;
}
