/**
 * @file i2c_lcd.h
 * @brief Non-blocking driver for HD44780 LCD via PCF8574 I2C backpack on ESP-IDF
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initializes the I2C bus and the 16x2 LCD in 4-bit mode.
 * @return ESP_OK on success, or error code.
 */
esp_err_t lcd_init(void);

/**
 * @brief Clears the LCD display and resets cursor to (0, 0).
 */
void lcd_clear(void);

/**
 * @brief Sets cursor position on the 16x2 screen.
 * @param col Column (0 to 15)
 * @param row Row (0 or 1)
 */
void lcd_set_cursor(uint8_t col, uint8_t row);

/**
 * @brief Prints a null-terminated string at the current cursor position.
 * @param str String to print.
 */
void lcd_print(const char *str);

/**
 * @brief Prints a formatted string on a specific row.
 * @param row Row (0 or 1)
 * @param format Printf format string
 */
void lcd_printf(uint8_t row, const char *format, ...);

/**
 * @brief Displays a clean prediction summary:
 *        Row 0: "Pred: <digit> (<conf>%)"
 *        Row 1: "Time: <time_ms> ms"
 */
void lcd_display_prediction(int digit, float confidence, int64_t latency_us);

#ifdef __cplusplus
}
#endif
