"""
Host PC Serial Streamer & Benchmark Tool for ESP32-S3
=====================================================
Streams 28x28 MNIST test images over USB serial (UART) to the ESP32-S3,
measures round-trip latency, and validates prediction accuracy.

Usage:
  python esp32_streamer.py --port /dev/ttyUSB0 --baud 115200 --samples 20
  python esp32_streamer.py --auto_port --digit 7
"""

import argparse
import sys
import os
import time
import json
import serial
import serial.tools.list_ports
import numpy as np
import torchvision
import torchvision.transforms as transforms


def auto_detect_esp32_port() -> str:
    """Finds the most likely USB-serial port for ESP32-S3."""
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        desc = (p.description or "").lower()
        hwid = (p.hwid or "").lower()
        if "cp210" in desc or "ch340" in desc or "ch341" in desc or "usb" in desc or "esp32" in desc or "jtag" in desc:
            return p.device
    if ports:
        return ports[0].device
    return "/dev/ttyUSB0"


def send_frame_and_receive(ser: serial.Serial, raw_pixels_784: np.ndarray, timeout_sec: float = 2.0) -> dict:
    """
    Sends sync header [0xAA, 0x55] + 784 bytes, then reads JSON response line from ESP32-S3.
    """
    ser.reset_input_buffer()
    
    # 1. Packet = Header (2 bytes) + Payload (784 bytes)
    packet = bytearray([0xAA, 0x55]) + bytearray(raw_pixels_784.astype(np.uint8).tobytes())
    
    t_start = time.perf_counter()
    ser.write(packet)
    ser.flush()
    
    # 2. Wait for newline-terminated JSON response (parsing past any ASCII art)
    ser.timeout = 0.5
    deadline = t_start + timeout_sec
    while time.perf_counter() < deadline:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if "{" in line and "}" in line:
            try:
                json_str = line[line.find("{"):line.rfind("}") + 1]
                data = json.loads(json_str)
                data["roundtrip_ms"] = (time.perf_counter() - t_start) * 1000.0
                if "pred" in data:
                    data["status"] = "ok"
                    data["digit"] = data["pred"]
                    data["confidence"] = data.get("conf", 0.0)
                return data
            except Exception:
                pass
            
    return {"status": "error", "raw_output": "timeout", "roundtrip_ms": (time.perf_counter() - t_start) * 1000.0}


def main():
    parser = argparse.ArgumentParser(description="ESP32-S3 Serial Inference Streamer")
    parser.add_argument("--port", type=str, default="", help="Serial port (e.g. /dev/ttyUSB0 or COM3)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--samples", type=int, default=10, help="Number of random test digits to send")
    parser.add_argument("--digit", type=int, default=-1, help="Test only specific digit (0..9)")
    parser.add_argument("--data_dir", type=str, default="./data", help="MNIST dataset root")
    args = parser.parse_args()

    port = args.port if args.port else auto_detect_esp32_port()
    print("=" * 80)
    print("        🚀 ESP32-S3 SERIAL MNIST STREAMER & BENCHMARK TOOL")
    print("=" * 80)
    print(f"Target Port:      {port}")
    print(f"Baud Rate:        {args.baud}")
    print(f"Test Samples:     {args.samples}")
    print("=" * 80)

    # Load MNIST test set
    transform = transforms.ToTensor()
    test_set = torchvision.datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform)

    # Filter samples
    selected = []
    for img, label in test_set:
        if args.digit >= 0 and label != args.digit:
            continue
        raw_bytes = (img.squeeze().numpy() * 255.0).astype(np.uint8).flatten()
        selected.append((raw_bytes, label))
        if len(selected) >= args.samples:
            break

    try:
        ser = serial.Serial(port, args.baud, timeout=1.0)
        print(f"Connected to {port} successfully!")
        print("Synchronizing with ESP32-S3 (waiting for boot self-test to complete)...")
        wait_start = time.time()
        ready = False
        while time.time() - wait_start < 15:
            l = ser.readline().decode("utf-8", errors="ignore")
            if "Listening for 784-byte frames" in l or "Ready for Stream" in l:
                ready = True
                print("✅ ESP32-S3 is ready for live frame streaming!\n")
                break
        if not ready:
            print("⚠️ Timeout waiting for ready prompt, proceeding anyway...\n")
        time.sleep(0.2)
        ser.reset_input_buffer()
    except Exception as e:
        print(f"❌ Error opening serial port {port}: {e}")
        print("Please check wiring or permissions (e.g. sudo usermod -aG dialout $USER).")
        return

    correct = 0
    total = len(selected)
    latencies_onchip = []
    latencies_roundtrip = []

    print(f"{'#':<4} | {'Expected':<9} | {'Predicted':<10} | {'Conf (%)':<9} | {'On-Chip (ms)':<13} | {'Roundtrip':<10} | {'Result'}")
    print("-" * 80)

    for i, (pixels, true_label) in enumerate(selected):
        res = send_frame_and_receive(ser, pixels)
        
        if res.get("status") == "ok":
            pred = res.get("digit", -1)
            conf = res.get("confidence", 0.0) * 100.0
            chip_ms = res.get("latency_ms", 0.0)
            rt_ms = res.get("roundtrip_ms", 0.0)
            
            is_match = (pred == true_label)
            if is_match: correct += 1
            latencies_onchip.append(chip_ms)
            latencies_roundtrip.append(rt_ms)

            status_str = "✅ PASS" if is_match else "❌ FAIL"
            print(f"{i+1:<4} | {true_label:<9} | {pred:<10} | {conf:>6.1f}%  | {chip_ms:>9.2f} ms   | {rt_ms:>7.1f} ms | {status_str}")
        else:
            print(f"{i+1:<4} | {true_label:<9} | ERROR      | N/A       | N/A           | {res.get('roundtrip_ms', 0):>7.1f} ms | ⚠️ TIMEOUT")

        time.sleep(0.1)  # 100ms pause between inferences

    ser.close()

    print("=" * 80)
    print("                    🏁 STREAMING BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Total Sent:               {total}")
    print(f"Accuracy:                 {correct}/{total} ({(correct/total)*100:.1f}%)")
    if latencies_onchip:
        print(f"Avg On-Chip Inference:    {np.mean(latencies_onchip):.2f} ms")
        print(f"Avg Host Round-Trip:      {np.mean(latencies_roundtrip):.2f} ms")
    print("=" * 80)


if __name__ == "__main__":
    main()
