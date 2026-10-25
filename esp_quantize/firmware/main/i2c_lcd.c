/**
 * @file i2c_lcd.c
 * @brief Implementation of PCF8574 I2C LCD Driver for ESP-IDF
 */

#include "i2c_lcd.h"
#include "app_config.h"
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_rom_sys.h"

static const char *TAG = "LCD_I2C";

static esp_err_t lcd_write_nibble(uint8_t nibble, uint8_t mode) {
    uint8_t data = (nibble & 0xF0) | LCD_BACKLIGHT | mode;
    uint8_t seq[3] = {
        data,
        (uint8_t)(data | LCD_ENABLE),
        (uint8_t)(data & ~LCD_ENABLE)
    };
    
    esp_err_t ret = i2c_master_write_to_device(
        LCD_I2C_NUM, LCD_I2C_ADDR, seq, sizeof(seq), pdMS_TO_TICKS(50)
    );
    esp_rom_delay_us(50);
    return ret;
}

static esp_err_t lcd_send_byte(uint8_t byte, uint8_t mode) {
    esp_err_t ret = lcd_write_nibble(byte & 0xF0, mode);
    if (ret != ESP_OK) return ret;
    return lcd_write_nibble((byte << 4) & 0xF0, mode);
}

static void lcd_send_cmd(uint8_t cmd) {
    lcd_send_byte(cmd, 0);
}

static void lcd_send_data(uint8_t data) {
    lcd_send_byte(data, LCD_RS);
}

esp_err_t lcd_init(void) {
    // 1. Initialize I2C Peripheral
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = LCD_I2C_SDA_PIN,
        .scl_io_num = LCD_I2C_SCL_PIN,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master = {
            .clk_speed = LCD_I2C_FREQ_HZ,
        },
        .clk_flags = 0,
    };
    
    esp_err_t ret = i2c_param_config(LCD_I2C_NUM, &conf);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "I2C param config failed: %s", esp_err_to_name(ret));
        return ret;
    }
    
    ret = i2c_driver_install(LCD_I2C_NUM, conf.mode, 0, 0, 0);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "I2C driver install failed: %s", esp_err_to_name(ret));
        return ret;
    }

    // 2. LCD Hardware Reset Sequence (HD44780 4-bit standard)
    vTaskDelay(pdMS_TO_TICKS(50));
    lcd_write_nibble(0x30, 0);
    vTaskDelay(pdMS_TO_TICKS(5));
    lcd_write_nibble(0x30, 0);
    esp_rom_delay_us(150);
    lcd_write_nibble(0x30, 0);
    esp_rom_delay_us(150);
    lcd_write_nibble(0x20, 0); // Switch to 4-bit mode
    esp_rom_delay_us(150);

    // 3. LCD Function Sets
    lcd_send_cmd(0x28); // 4-bit, 2 lines, 5x8 font
    lcd_send_cmd(0x0C); // Display ON, Cursor OFF, Blink OFF
    lcd_send_cmd(0x06); // Increment cursor, no shift
    lcd_send_cmd(0x01); // Clear display
    vTaskDelay(pdMS_TO_TICKS(5));

    ESP_LOGI(TAG, "I2C 16x2 LCD initialized at addr 0x%02X (SDA=%d, SCL=%d)",
             LCD_I2C_ADDR, LCD_I2C_SDA_PIN, LCD_I2C_SCL_PIN);
    return ESP_OK;
}

void lcd_clear(void) {
    lcd_send_cmd(0x01);
    vTaskDelay(pdMS_TO_TICKS(3));
}

void lcd_set_cursor(uint8_t col, uint8_t row) {
    static const uint8_t row_offsets[] = {0x00, 0x40};
    if (row > 1) row = 1;
    if (col > 15) col = 15;
    lcd_send_cmd(0x80 | (col + row_offsets[row]));
}

void lcd_print(const char *str) {
    while (*str) {
        lcd_send_data((uint8_t)(*str++));
    }
}

void lcd_printf(uint8_t row, const char *format, ...) {
    char buf[17];
    va_list args;
    va_start(args, format);
    vsnprintf(buf, sizeof(buf), format, args);
    va_end(args);

    // Pad with spaces to clear trailing characters on row
    int len = strlen(buf);
    while (len < 16) {
        buf[len++] = ' ';
    }
    buf[16] = '\0';

    lcd_set_cursor(0, row);
    lcd_print(buf);
}

void lcd_display_prediction(int digit, float confidence, int64_t latency_us) {
    float latency_ms = (float)latency_us / 1000.0f;
    int conf_pct = (int)(confidence * 100.0f);

    lcd_printf(0, "Pred: %d (%d%%)", digit, conf_pct);
    lcd_printf(1, "Time: %.2f ms", latency_ms);
}
