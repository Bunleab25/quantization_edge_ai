#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "firmware/main/model_data.h"

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

static int predict(const uint8_t *raw_pixels_784) {
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

    int best = 0;
    float max_l = logits[0];
    for (int i = 1; i < 10; ++i) {
        if (logits[i] > max_l) {
            max_l = logits[i];
            best = i;
        }
    }
    return best;
}

int main(void) {
    printf("======================================================================\n");
    printf("     🚀 10,000 MNIST TEST SET ACCURACY VERIFICATION (PURE C ENGINE)   \n");
    printf("======================================================================\n");

    const char *img_path = "data/MNIST/raw/t10k-images-idx3-ubyte";
    const char *lbl_path = "data/MNIST/raw/t10k-labels-idx1-ubyte";

    FILE *f_img = fopen(img_path, "rb");
    FILE *f_lbl = fopen(lbl_path, "rb");
    if (!f_img || !f_lbl) {
        fprintf(stderr, "Error: Could not open MNIST raw files!\n");
        return 1;
    }

    // Skip headers (16 bytes for images, 8 bytes for labels)
    fseek(f_img, 16, SEEK_SET);
    fseek(f_lbl, 8, SEEK_SET);

    init_lut();

    uint8_t img_buf[784];
    uint8_t true_lbl = 0;

    int total = 10000;
    int correct = 0;
    int per_digit_total[10] = {0};
    int per_digit_correct[10] = {0};
    int confusion[10][10] = {{0}};

    clock_t t0 = clock();

    for (int i = 0; i < total; ++i) {
        if (fread(img_buf, 1, 784, f_img) != 784) break;
        if (fread(&true_lbl, 1, 1, f_lbl) != 1) break;

        int pred = predict(img_buf);

        per_digit_total[true_lbl]++;
        confusion[true_lbl][pred]++;

        if (pred == (int)true_lbl) {
            correct++;
            per_digit_correct[true_lbl]++;
        }

        if ((i + 1) % 1000 == 0) {
            printf("  Processed %5d / %d samples | Running Accuracy: %6.2f%%\n",
                   i + 1, total, (double)correct * 100.0 / (i + 1));
        }
    }

    clock_t t1 = clock();
    double total_sec = (double)(t1 - t0) / CLOCKS_PER_SEC;

    fclose(f_img);
    fclose(f_lbl);

    double accuracy = (double)correct * 100.0 / total;

    printf("\n======================================================================\n");
    printf("                    🏁 FINAL 10,000 TEST EVALUATION RESULT            \n");
    printf("======================================================================\n");
    printf("Total Test Samples Evaluated : %d\n", total);
    printf("Total Correct Predictions    : %d / %d\n", correct, total);
    printf("Total Errors                 : %d / %d\n", total - correct, total);
    printf("Top-1 Test Accuracy          : %.2f%%\n", accuracy);
    printf("Total Evaluation Time        : %.2f seconds (%.2f ms / sample)\n", total_sec, (total_sec * 1000.0) / total);
    printf("Inference Throughput         : %.1f samples / second\n", total / total_sec);
    printf("----------------------------------------------------------------------\n");
    printf("Per-Digit Accuracy Breakdown:\n");
    for (int d = 0; d < 10; ++d) {
        printf("  Digit %d : %4d / %4d Correct (%6.2f%%)\n",
               d, per_digit_correct[d], per_digit_total[d],
               (double)per_digit_correct[d] * 100.0 / per_digit_total[d]);
    }
    printf("======================================================================\n");
    return 0;
}
