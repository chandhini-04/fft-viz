import serial
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import tkinter as tk
from tkinter import messagebox
from scipy.signal import butter, filtfilt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Serial Port Configuration
SERIAL_PORT = 'COM7'  # Change based on your device
BAUD_RATE = 115200
TIMEOUT = 0

# Sweep 
START_FREQ = 2430000000
STOP_FREQ = 2725000000


POINTS_PER_SWEEP = 500
SWEEP_SEGMENTS = 4

def initialize_nanovna(ser, start_freq, stop_freq, points):
    ser.write(f"scan {start_freq} {stop_freq} {points}\n".encode())
    time.sleep(0.1)
    ser.write(b"data 1\n")
    time.sleep(0.1)

def collect_sweep_data(ser):
    data = []
    while ser.in_waiting > 0:
        line = ser.readline().decode('utf-8').strip()
        try:
            real, imag = map(float, line.split())
            data.append((real, imag))
        except ValueError:
            continue
    return data

def calculate_frequencies(segment_start, segment_stop, points):
    return np.linspace(segment_start, segment_stop, points)

def collect_and_save_data(duration=60):
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
        print("Connected to NanoVNA.")
    except serial.SerialException:
        messagebox.showerror("Error", f"Unable to connect to {SERIAL_PORT}.")
        return None

    all_data = []
    start_time = time.time()

    try:
        while time.time() - start_time < duration:
            segment_width = (STOP_FREQ - START_FREQ) // SWEEP_SEGMENTS
            for i in range(SWEEP_SEGMENTS):
                segment_start = START_FREQ + i * segment_width
                segment_stop = segment_start + segment_width
                initialize_nanovna(ser, segment_start, segment_stop, POINTS_PER_SWEEP)
                
                sweep_data = collect_sweep_data(ser)
                if sweep_data:
                    frequencies = calculate_frequencies(segment_start, segment_stop, POINTS_PER_SWEEP)
                    timestamp = round(time.time() - start_time, 2)
                    sweep_data_with_meta = [(timestamp, freq, real, imag) for freq, (real, imag) in zip(frequencies, sweep_data)]
                    all_data.extend(sweep_data_with_meta)
    finally:
        ser.close()
        print("Serial connection closed.")
        end_time = time.time()
        print(f"Data collection duration: {round(end_time - start_time, 2)} seconds")


    if all_data:
        df = pd.DataFrame(all_data, columns=["Timestamp", "Frequency", "Real", "Imag"])
        df["Magnitude_dB"] = 20 * np.log10(np.sqrt(df["Real"]**2 + df["Imag"]**2))
        df["Phase_Degrees"] = np.arctan2(df["Imag"], df["Real"]) * (180 / np.pi)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"nanovna_{timestamp}.csv"
        df.to_csv(file_path, index=False)
        return file_path
    return None

def bandpass_filter(signal, lowcut, highcut, fs, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def process_data(file_path, canvas, result_label):
    df = pd.read_csv(file_path)
    magnitude = 10 ** (df["Magnitude_dB"] / 20)
    phase = np.radians(df["Phase_Degrees"])
    num_points = len(magnitude)
    time_duration = max(df["Timestamp"]) - min(df["Timestamp"])
    fs = num_points / time_duration
    time_vector = np.linspace(0, time_duration, num_points)
    s11_complex = magnitude * np.exp(1j * phase)
    s11_time_domain = np.fft.ifft(s11_complex).real

    heart_signal = bandpass_filter(s11_time_domain, 1.07, 2.2, fs)
    heart_fft = np.abs(np.fft.fft(heart_signal))[:num_points//2]
    heart_freqs = np.fft.fftfreq(num_points, d=1/fs)[:num_points//2]
    heart_rate_bpm = heart_freqs[np.argmax(heart_fft)] * 60

    resp_signal = bandpass_filter(s11_time_domain, 0.22, 0.4, fs)
    resp_fft = np.abs(np.fft.fft(resp_signal))[:num_points//2]
    resp_freqs = np.fft.fftfreq(num_points, d=1/fs)[:num_points//2]
    respiratory_rate_bpm = resp_freqs[np.argmax(resp_fft)] * 60

    result_label.config(text=f"Heart Rate: {heart_rate_bpm:.2f} BPM\nRespiratory Rate: {respiratory_rate_bpm:.2f} BPM")

    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    axs[0, 0].plot(heart_freqs, heart_fft, 'b')
    axs[0, 0].set_title("Heart Rate Spectrum")
    axs[0, 1].plot(resp_freqs, resp_fft, 'g')
    axs[0, 1].set_title("Respiratory Rate Spectrum")
    axs[1, 0].plot(time_vector, heart_signal, 'r')
    axs[1, 0].set_title("Filtered Heart Signal")
    axs[1, 1].plot(time_vector, resp_signal, 'm')
    axs[1, 1].set_title("Filtered Respiratory Signal")
    fig.tight_layout()

    for widget in canvas.winfo_children():
        widget.destroy()
    canvas_widget = FigureCanvasTkAgg(fig, master=canvas)
    canvas_widget.get_tk_widget().pack()
    canvas_widget.draw()

def start_process():
    result_label.config(text="Collecting Data... Please Wait")
    root.update()
    file_path = collect_and_save_data(60)
    if file_path:
        process_data(file_path, canvas, result_label)
    else:
        result_label.config(text="Data collection failed.")

root = tk.Tk()
root.title("Heart and Respiratory Rate Monitor")

tk.Button(root, text="Start Measurement", command=start_process, font=("Arial", 14)).pack(pady=10)
result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack()
canvas = tk.Frame(root)
canvas.pack()

root.mainloop()