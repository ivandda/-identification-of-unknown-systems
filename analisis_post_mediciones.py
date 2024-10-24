import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import freqz, butter, filtfilt
import os
from datetime import datetime

from generador_de_senales import setup_directories, generate_linear_chirp


def analyze_frequency_response(
        input_file,
        output_file,
        window_size=1024,
        overlap=512,
        sample_rate=48000
):
    """
    Analiza la respuesta en frecuencia del sistema usando el método de Welch
    y estima la ubicación aproximada de polos y ceros.
    """
    # Leer archivos
    _, input_signal = wavfile.read(input_file)
    _, output_signal = wavfile.read(output_file)

    # Convertir a float
    input_signal = input_signal / 32767.0
    output_signal = output_signal / 32767.0

    # Calcular densidad espectral de potencia
    f, Pxx_input = signal.welch(input_signal, sample_rate, nperseg=window_size, noverlap=overlap)
    f, Pxx_output = signal.welch(output_signal, sample_rate, nperseg=window_size, noverlap=overlap)

    # Calcular función de transferencia
    H = np.sqrt(Pxx_output / np.clip(Pxx_input, 1e-10, None))
    phase = np.angle(H)

    # Graficar respuesta en frecuencia
    plt.figure(figsize=(15, 10))

    # Magnitud
    plt.subplot(2, 1, 1)
    plt.semilogx(f, 20 * np.log10(np.abs(H)))
    plt.title('Respuesta en Frecuencia del Sistema')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Magnitud (dB)')
    plt.grid(True)

    # Fase
    plt.subplot(2, 1, 2)
    plt.semilogx(f, np.unwrap(phase) * 180 / np.pi)
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Fase (grados)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Estimar ubicación de polos y ceros
    # Nota: Esta es una estimación aproximada
    peaks, _ = signal.find_peaks(20 * np.log10(np.abs(H)), height=3)
    valleys, _ = signal.find_peaks(-(20 * np.log10(np.abs(H))), height=3)

    print("Posibles frecuencias de polos (resonancias):")
    for peak in peaks:
        print(f"{f[peak]:.1f} Hz")

    print("\nPosibles frecuencias de ceros (antiresonancias):")
    for valley in valleys:
        print(f"{f[valley]:.1f} Hz")

    return f, H


def analyze_snr(
        signal_file,
        noise_file,
        window_size=1024,
        sample_rate=48000
):
    """
    Analiza la relación señal-ruido (SNR) y estima la calidad de la medición.
    """
    # Leer archivos
    _, signal_data = wavfile.read(signal_file)
    _, noise_data = wavfile.read(noise_file)

    # Convertir a float
    signal_data = signal_data / 32767.0
    noise_data = noise_data / 32767.0

    # Calcular potencia de señal y ruido
    signal_power = np.mean(signal_data ** 2)
    noise_power = np.mean(noise_data ** 2)

    # Calcular SNR
    snr = 10 * np.log10(signal_power / noise_power)

    # Calcular espectros
    f, Pxx_signal = signal.welch(signal_data, sample_rate, nperseg=window_size)
    f, Pxx_noise = signal.welch(noise_data, sample_rate, nperseg=window_size)

    # Graficar
    plt.figure(figsize=(15, 8))
    plt.semilogy(f, Pxx_signal, label='Señal')
    plt.semilogy(f, Pxx_noise, label='Ruido')
    plt.title(f'Análisis de SNR (SNR = {snr:.1f} dB)')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Densidad Espectral de Potencia')
    plt.grid(True)
    plt.legend()
    plt.show()

    return snr


def estimate_measurement_uncertainty(
        input_signal,
        output_signal,
        n_bootstrap=100,
        confidence_level=0.95,
        window_size=1024,
        sample_rate=48000
):
    """
    Estima la incertidumbre en las mediciones usando bootstrap.

    Parameters:
    -----------
    input_signal : array
        Señal de entrada
    output_signal : array
        Señal de salida
    n_bootstrap : int
        Número de iteraciones bootstrap
    confidence_level : float
        Nivel de confianza para los intervalos
    window_size : int
        Tamaño de ventana para análisis espectral
    sample_rate : int
        Frecuencia de muestreo

    Returns:
    --------
    tuple
        (frecuencias, magnitud media, límite inferior, límite superior)
    """
    n_samples = len(input_signal)
    transfer_functions = []

    for _ in range(n_bootstrap):
        # Muestreo aleatorio con reemplazo
        indices = np.random.randint(0, n_samples, size=n_samples)
        input_bootstrap = input_signal[indices]
        output_bootstrap = output_signal[indices]

        # Calcular función de transferencia
        f, Pxx_input = signal.welch(input_bootstrap, sample_rate, nperseg=window_size)
        _, Pxx_output = signal.welch(output_bootstrap, sample_rate, nperseg=window_size)
        H = np.sqrt(Pxx_output / np.clip(Pxx_input, 1e-10, None))
        transfer_functions.append(20 * np.log10(np.abs(H)))

    # Calcular estadísticas
    tf_array = np.array(transfer_functions)
    mean_tf = np.mean(tf_array, axis=0)
    lower_bound = np.percentile(tf_array, (1 - confidence_level) * 100 / 2, axis=0)
    upper_bound = np.percentile(tf_array, (1 + confidence_level) * 100 / 2, axis=0)

    # Graficar
    plt.figure(figsize=(15, 6))
    plt.semilogx(f, mean_tf, 'b-', label='Media')
    plt.fill_between(f, lower_bound, upper_bound, color='b', alpha=0.2,
                     label=f'Intervalo de confianza {confidence_level * 100}%')
    plt.title('Función de Transferencia con Intervalos de Confianza')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Magnitud (dB)')
    plt.grid(True)
    plt.legend()
    plt.show()

    return f, mean_tf, lower_bound, upper_bound


def analyze_distortion(
        input_file,
        output_file,
        fundamental_freq=1000,  # Frecuencia fundamental en Hz
        sample_rate=48000,
        window_size=8192
):
    """
    Analiza la distorsión armónica del sistema.

    Parameters:
    -----------
    input_file : str
        Archivo de entrada
    output_file : str
        Archivo de salida
    fundamental_freq : float
        Frecuencia fundamental esperada
    sample_rate : int
        Frecuencia de muestreo
    window_size : int
        Tamaño de ventana para FFT
    """
    # Leer archivos
    _, input_signal = wavfile.read(input_file)
    _, output_signal = wavfile.read(output_file)

    # Convertir a float
    input_signal = input_signal / 32767.0
    output_signal = output_signal / 32767.0

    # Calcular espectro
    f_in = np.fft.rfftfreq(window_size, 1 / sample_rate)
    spectrum_in = np.abs(np.fft.rfft(input_signal[:window_size]))
    spectrum_out = np.abs(np.fft.rfft(output_signal[:window_size]))

    # Encontrar armónicos
    harmonics_freq = np.arange(1, 6) * fundamental_freq
    harmonic_amplitudes = []

    for freq in harmonics_freq:
        idx = np.argmin(np.abs(f_in - freq))
        harmonic_amplitudes.append(spectrum_out[idx])

    # Calcular THD (Total Harmonic Distortion)
    fundamental = harmonic_amplitudes[0]
    harmonics = harmonic_amplitudes[1:]
    thd = np.sqrt(np.sum(np.array(harmonics) ** 2)) / fundamental * 100

    # Graficar
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    plt.plot(f_in, 20 * np.log10(np.clip(spectrum_in, 1e-10, None)),
             label='Entrada')
    plt.title('Espectro de Entrada')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Magnitud (dB)')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(f_in, 20 * np.log10(np.clip(spectrum_out, 1e-10, None)),
             label=f'Salida (THD={thd:.1f}%)')
    for i, freq in enumerate(harmonics_freq):
        plt.axvline(freq, color='r', linestyle='--', alpha=0.5)
        plt.text(freq, -20, f'H{i + 1}')
    plt.title('Espectro de Salida con Armónicos')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Magnitud (dB)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    return thd, harmonics_freq, harmonic_amplitudes


def analyze_bandwidth_vs_sampling(
        input_signal,
        sample_rates=[48000, 44100, 22050],
        window_size=1024
):
    """
    Analiza cómo afecta la frecuencia de muestreo al ancho de banda medible.

    Parameters:
    -----------
    input_signal : array
        Señal de entrada original
    sample_rates : list
        Lista de frecuencias de muestreo a probar
    window_size : int
        Tamaño de ventana para análisis espectral
    """
    plt.figure(figsize=(15, 8))

    for fs in sample_rates:
        # Remuestrear señal
        samples = int(len(input_signal) * fs / 48000)
        resampled = signal.resample(input_signal, samples)

        # Calcular espectro
        f, Pxx = signal.welch(resampled, fs, nperseg=window_size)

        # Graficar
        plt.semilogx(f, 10 * np.log10(Pxx),
                     label=f'Fs = {fs / 1000:.1f} kHz')

    plt.title('Efecto de la Frecuencia de Muestreo en el Ancho de Banda')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Densidad Espectral de Potencia (dB)')
    plt.grid(True)
    plt.legend()
    plt.show()


# Ejemplo de uso
if __name__ == "__main__":
    # Configuración básica
    setup_directories()

    # Generar señales de prueba
    chirp_file = generate_linear_chirp()

    # Análisis completo del sistema
    # (Asumiendo que ya tenemos archivos de entrada y salida)
    f, H = analyze_frequency_response('input.wav', 'output.wav')
    snr = analyze_snr('signal.wav', 'noise.wav')

    # Cargar señales para análisis de incertidumbre
    _, input_signal = wavfile.read('input.wav')
    _, output_signal = wavfile.read('output.wav')

    f, mean_tf, lower, upper = estimate_measurement_uncertainty(
        input_signal / 32767.0,
        output_signal / 32767.0
    )

    thd, harmonics, amplitudes = analyze_distortion(
        'input.wav',
        'output.wav'
    )

    analyze_bandwidth_vs_sampling(input_signal / 32767.0)