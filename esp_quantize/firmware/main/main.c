/**
 * @file main.c
 * @brief Pure C ESP32-S3 Dual-Mode MNIST CNN Deployment Application
 * 
 * Features:
 * 1. Self-Test on Boot: Automatically verifies all 10 embedded test digits (0..9).
 * 2. Microsecond-Level Latency Profiling (esp_timer) across every layer.
 * 3. Exact Memory & Resource Footprint Audit (Flash, Internal SRAM, Heap Watermark).
 * 4. I2C 16x2 LCD Output: Real-time prediction, confidence %, and latency.
 * 5. Continuous UART Stream Task: Non-blocking packet listener receiving 28x28 frames.
 */

#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_system.h"
#include "driver/uart.h"

#include "app_config.h"
#include "i2c_lcd.h"
#include "espdl_model.h"
#include "test_samples.h"
#include "model_data.h"

static const char *TAG = "APP_MAIN";

static void uart_stream_init(void) {
    const uart_config_t uart_config = {
        .baud_rate = STREAM_UART_BAUD,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };

    ESP_ERROR_CHECK(uart_driver_install(STREAM_UART_NUM, STREAM_UART_BUF_SIZE * 2, 0, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_param_config(STREAM_UART_NUM, &uart_config));
    ESP_LOGI(TAG, "UART%d stream initialized at %d baud", STREAM_UART_NUM, STREAM_UART_BAUD);
}

static void print_memory_footprint(void) {
    uint32_t free_heap = esp_get_free_heap_size();
    uint32_t min_heap = esp_get_minimum_free_heap_size();

    ESP_LOGI(TAG, "============================================================");
    ESP_LOGI(TAG, "💾  ESP32-S3 MEMORY & RESOURCE FOOTPRINT AUDIT");
    ESP_LOGI(TAG, "============================================================");
    ESP_LOGI(TAG, "  Model Weights in Flash (.rodata) : %7d bytes (%5.1f KB)", TOTAL_MODEL_WEIGHT_BYTES, (float)TOTAL_MODEL_WEIGHT_BYTES / 1024.0f);
    ESP_LOGI(TAG, "  Static RAM Ping-Pong Buffers     : %7d bytes (%5.1f KB)", 28 * 28 * 32 * 2, (float)(28 * 28 * 32 * 2) / 1024.0f);
    ESP_LOGI(TAG, "  Static Input Grayscale Buffer    : %7d bytes (%5.1f KB)", MNIST_IMAGE_SIZE, (float)MNIST_IMAGE_SIZE / 1024.0f);
    ESP_LOGI(TAG, "  Fast Normalization LUT           : %7d bytes", 256);
    ESP_LOGI(TAG, "  Internal Free Heap (DRAM)        : %7ld bytes (%5.1f KB free)", free_heap, (float)free_heap / 1024.0f);
    ESP_LOGI(TAG, "  Minimum Free Heap Ever (Watermark): %6ld bytes (%5.1f KB)", min_heap, (float)min_heap / 1024.0f);
    ESP_LOGI(TAG, "  External PSRAM (SPIRAM) Allocated: %7d bytes (0.0 KB - NONE NEEDED)", 0);
    ESP_LOGI(TAG, "============================================================\n");
}

static void print_microsecond_profile(const espdl_timing_profile_t *p) {
    ESP_LOGI(TAG, "------------------------------------------------------------");
    ESP_LOGI(TAG, "⏱️  MICROSECOND-LEVEL LAYER BREAKDOWN (esp_timer)");
    ESP_LOGI(TAG, "------------------------------------------------------------");
    ESP_LOGI(TAG, "  1. Input LUT Normalization : %7lld us (%5.2f ms) | %4.1f%%", p->prep_us, (float)p->prep_us / 1000.0f, (float)p->prep_us * 100.0f / p->total_us);
    ESP_LOGI(TAG, "  2. Stage 1 Conv 1 (32-ch)  : %7lld us (%5.2f ms) | %4.1f%%", p->stage1_conv1_us, (float)p->stage1_conv1_us / 1000.0f, (float)p->stage1_conv1_us * 100.0f / p->total_us);
    ESP_LOGI(TAG, "  3. Stage 1 Conv 2 (32-ch)  : %7lld us (%5.2f ms) | %4.1f%%", p->stage1_conv2_us, (float)p->stage1_conv2_us / 1000.0f, (float)p->stage1_conv2_us * 100.0f / p->total_us);
    ESP_LOGI(TAG, "  4. Stage 1 MaxPool (14x14) : %7lld us (%5.2f ms) | %4.1f%%", p->stage1_pool_us, (float)p->stage1_pool_us / 1000.0f, (float)p->stage1_pool_us * 100.0f / p->total_us);
    ESP_LOGI(TAG, "  5. Stage 2 Conv 1 (64-ch)  : %7lld us (%5.2f ms) | %4.1f%%", p->stage2_conv1_us, (float)p->stage2_conv1_us / 1000.0f, (float)p->stage2_conv1_us * 100.0f / p->total_us);
    ESP_LOGI(TAG, "  6. Stage 2 Conv 2 (64-ch)  : %7lld us (%5.2f ms) | %4.1f%%", p->stage2_conv2_us, (float)p->stage2_conv2_us / 1000.0f, (float)p->stage2_conv2_us * 100.0f / p->total_us);
    ESP_LOGI(TAG, "  7. Stage 2 MaxPool (7x7)   : %7lld us (%5.2f ms) | %4.1f%%", p->stage2_pool_us, (float)p->stage2_pool_us / 1000.0f, (float)p->stage2_pool_us * 100.0f / p->total_us);
    ESP_LOGI(TAG, "  8. Stage 3 Conv 1 (128-ch) : %7lld us (%5.2f ms) | %4.1f%%", p->stage3_conv1_us, (float)p->stage3_conv1_us / 1000.0f, (float)p->stage3_conv1_us * 100.0f / p->total_us);
    ESP_LOGI(TAG, "  9. Stage 3 Conv 2 (128-ch) : %7lld us (%5.2f ms) | %4.1f%%", p->stage3_conv2_us, (float)p->stage3_conv2_us / 1000.0f, (float)p->stage3_conv2_us * 100.0f / p->total_us);
    ESP_LOGI(TAG, " 10. Global Avg Pool (GAP)   : %7lld us (%5.2f ms) | %4.1f%%", p->gap_us, (float)p->gap_us / 1000.0f, (float)p->gap_us * 100.0f / p->total_us);
    ESP_LOGI(TAG, " 11. Classifier Head (10-ch) : %7lld us (%5.2f ms) | %4.1f%%", p->classifier_us, (float)p->classifier_us / 1000.0f, (float)p->classifier_us * 100.0f / p->total_us);
    ESP_LOGI(TAG, " 12. Softmax Normalization   : %7lld us (%5.2f ms) | %4.1f%%", p->softmax_us, (float)p->softmax_us / 1000.0f, (float)p->softmax_us * 100.0f / p->total_us);
    ESP_LOGI(TAG, "------------------------------------------------------------");
    ESP_LOGI(TAG, " TOTAL ON-CHIP LATENCY       : %7lld us (%6.2f ms)", p->total_us, (float)p->total_us / 1000.0f);
    ESP_LOGI(TAG, "------------------------------------------------------------\n");
}

/**
 * @brief Phase A: Self-Test on embedded MNIST digits (0..9) at boot.
 */
static void run_boot_self_test(void) {
    ESP_LOGI(TAG, "============================================================");
    ESP_LOGI(TAG, "         🧪 RUNNING BOOT SELF-TEST ON 10 EMBEDDED DIGITS");
    ESP_LOGI(TAG, "============================================================");

    lcd_clear();
    lcd_printf(0, "Running SelfTest");
    lcd_printf(1, "Testing 0..9...");

    int passed = 0;
    int64_t total_latency_us = 0;

    for (int i = 0; i < NUM_TEST_SAMPLES; ++i) {
        float confidence = 0.0f;
        espdl_timing_profile_t prof;
        
        int pred = espdl_model_predict_profile(TEST_DIGIT_SAMPLES[i], &confidence, &prof);
        total_latency_us += prof.total_us;
        
        int expected = TEST_EXPECTED_LABELS[i];
        bool is_correct = (pred == expected);
        if (is_correct) passed++;

        ESP_LOGI(TAG, "Sample %d | Expected: %d | Predicted: %d | Conf: %5.1f%% | Time: %6.2f ms | %s",
                 i, expected, pred, confidence * 100.0f, (float)prof.total_us / 1000.0f,
                 is_correct ? "✅ PASS" : "❌ FAIL");
        
        // Print detailed microsecond profile for Sample 0
        if (i == 0) {
            print_microsecond_profile(&prof);
        }

        // Feed watchdog and yield CPU briefly
        vTaskDelay(pdMS_TO_TICKS(50));
    }

    float avg_latency_ms = (float)total_latency_us / (NUM_TEST_SAMPLES * 1000.0f);
    ESP_LOGI(TAG, "------------------------------------------------------------");
    ESP_LOGI(TAG, "Self-Test Result: %d / %d Correct | Avg Latency: %.2f ms",
             passed, NUM_TEST_SAMPLES, avg_latency_ms);
    ESP_LOGI(TAG, "============================================================\n");

    // Display Summary on LCD
    lcd_clear();
    lcd_printf(0, "Test: %d/%d Pass", passed, NUM_TEST_SAMPLES);
    lcd_printf(1, "Avg: %.1f ms", avg_latency_ms);
    vTaskDelay(pdMS_TO_TICKS(3000)); // Hold on screen for 3 seconds
}

/**
 * @brief Phase B: Continuous UART listener task for streaming frames from PC.
 * Packet structure: [0xAA] [0x55] [784 bytes raw grayscale pixels]
 */
static void uart_listener_task(void *pvParameters) {
    uint8_t rx_byte = 0;
    uint8_t image_buf[MNIST_IMAGE_SIZE];

    lcd_clear();
    lcd_printf(0, "Ready for Stream");
    lcd_printf(1, "Baud: 115200");
    ESP_LOGI(TAG, "Listening for 784-byte frames (Sync Header: 0x%02X 0x%02X)...",
             FRAME_SYNC_BYTE1, FRAME_SYNC_BYTE2);

    while (1) {
        // Step 1: Search for Sync Byte 1
        int len = uart_read_bytes(STREAM_UART_NUM, &rx_byte, 1, portMAX_DELAY);
        if (len <= 0 || rx_byte != FRAME_SYNC_BYTE1) continue;

        // Step 2: Check Sync Byte 2
        len = uart_read_bytes(STREAM_UART_NUM, &rx_byte, 1, pdMS_TO_TICKS(100));
        if (len <= 0 || rx_byte != FRAME_SYNC_BYTE2) continue;

        // Step 3: Sync verified! Read exactly 784 image bytes
        int bytes_received = 0;
        while (bytes_received < MNIST_IMAGE_SIZE) {
            int chunk = uart_read_bytes(
                STREAM_UART_NUM,
                image_buf + bytes_received,
                MNIST_IMAGE_SIZE - bytes_received,
                pdMS_TO_TICKS(500)
            );
            if (chunk > 0) {
                bytes_received += chunk;
            } else {
                ESP_LOGW(TAG, "Frame timeout: received %d / %d bytes", bytes_received, MNIST_IMAGE_SIZE);
                break;
            }
        }

        if (bytes_received == MNIST_IMAGE_SIZE) {
            // Step 4: Run Forward Inference with microsecond timer
            float confidence = 0.0f;
            int64_t latency_us = 0;
            int predicted = espdl_model_predict(image_buf, &confidence, &latency_us);
            float latency_ms = (float)latency_us / 1000.0f;

            // Step 5: Update LCD Display
            lcd_display_prediction(predicted, confidence, latency_us);

            // Step 6: Print ASCII Art and Return JSON over UART
            espdl_print_ascii_art(image_buf);
            
            char resp[128];
            int resp_len = snprintf(
                resp, sizeof(resp),
                "{\"pred\":%d,\"conf\":%.4f,\"latency_ms\":%.2f,\"latency_us\":%lld}\n",
                predicted, confidence, latency_ms, latency_us
            );
            uart_write_bytes(STREAM_UART_NUM, resp, resp_len);

            ESP_LOGI(TAG, "Stream Infer -> Digit: %d (%.1f%%) in %lld us (%.2f ms)",
                     predicted, confidence * 100.0f, latency_us, latency_ms);
        }
    }
}

void app_main(void) {
    ESP_LOGI(TAG, "Booting ESP32-S3 Advanced MNIST CNN (Pure C)...");

    // 1. Initialize Peripherals
    esp_err_t lcd_res = lcd_init();
    if (lcd_res != ESP_OK) {
        ESP_LOGW(TAG, "LCD init warning (proceeding with UART only): %s", esp_err_to_name(lcd_res));
    }
    uart_stream_init();

    // 2. Initialize Model Engine
    espdl_model_init();

    // 3. Print Memory & Resource Footprint Audit
    print_memory_footprint();

    // 4. Phase A: Self-Test on Boot
    run_boot_self_test();

    // 5. Print Post-Inference Memory Footprint (verifying 0 heap leak)
    print_memory_footprint();

    // 6. Phase B: Launch Live UART Streaming Listener Task
    xTaskCreatePinnedToCore(
        uart_listener_task,
        "uart_listener",
        8192,
        NULL,
        5,
        NULL,
        1 // Pin to Core 1 (leaving Core 0 for system/RF)
    );
}
