/**
 * @file app_config.h
 * @brief Central Hardware and Deployment Configuration for ESP32-S3 MNIST CNN
 */

#pragma once

#include "driver/gpio.h"
#include "driver/i2c.h"
#include "driver/uart.h"

// ==============================================================================
// I2C 16x2 LCD (PCF8574 Backpack) Configuration
// ==============================================================================
#define LCD_I2C_NUM             I2C_NUM_0
#define LCD_I2C_SDA_PIN         GPIO_NUM_1      /**< Configurable I2C SDA (match your wiring) */
#define LCD_I2C_SCL_PIN         GPIO_NUM_2      /**< Configurable I2C SCL (match your wiring) */
#define LCD_I2C_FREQ_HZ         100000          /**< 100 kHz Standard I2C Clock */
#define LCD_I2C_ADDR            0x27            /**< Default PCF8574 address (or 0x3F) */

// PCF8574 Pin Mapping on I2C Backpack
#define LCD_BACKLIGHT           0x08
#define LCD_ENABLE              0x04
#define LCD_RW                  0x00
#define LCD_RS                  0x01

// ==============================================================================
// UART Serial Streamer Configuration (for live host communication)
// ==============================================================================
#define STREAM_UART_NUM         UART_NUM_0      /**< Default USB-to-UART bridge */
#define STREAM_UART_BAUD        115200
#define STREAM_UART_BUF_SIZE    1024

// Frame Synchronization (Host -> ESP32-S3)
#define FRAME_SYNC_BYTE1        0xAA
#define FRAME_SYNC_BYTE2        0x55

// ==============================================================================
// Image and Model Dimensions
// ==============================================================================
#define MNIST_IMAGE_WIDTH       28
#define MNIST_IMAGE_HEIGHT      28
#define MNIST_IMAGE_CHANNELS    1
#define MNIST_IMAGE_SIZE        (MNIST_IMAGE_WIDTH * MNIST_IMAGE_HEIGHT) // 784 bytes
#define MNIST_NUM_CLASSES       10

// Normalization Constants (derived from MNIST training set EDA)
#define MNIST_NORM_MEAN         0.1307f
#define MNIST_NORM_STD          0.3081f
