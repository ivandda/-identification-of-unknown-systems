import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy import signal as sg


def setup_directories():
    """
    Crea la estructura de directorios para organizar los archivos de prueba.
    """
    directories = [
        'signals/chirp',
        'signals/noise',
        'signals/amplitude_test',
        'signals/impulse',
        'analysis/linearity',
        'analysis/frequency_response',
        'analysis/noise_floor',
        'analysis/temporal'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def generate_linear_chirp(
        duration=5.0,
        f_start=20,
        f_end=20000,
        sample_rate=48000,
        amplitude=0.8,
):
    """
    Genera un chirp lineal con visualización mejorada y nombre de archivo descriptivo.
    """
    # Generar vector de tiempo
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Calcular la tasa de cambio de frecuencia
    k = (f_end - f_start) / duration

    # Generar el chirp
    signal = amplitude * np.sin(2 * np.pi * (f_start * t + (k / 2) * t ** 2))

    # Crear nombre de archivo descriptivo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'signals/chirp/chirp_linear_{f_start}Hz_{f_end}Hz_{duration}s_amp{amplitude}_{timestamp}.wav'

    # Normalizar y guardar
    signal_normalized = np.int16(signal * 32767)
    wavfile.write(filename, sample_rate, signal_normalized)

    # Visualización mejorada
    plt.figure(figsize=(15, 8))

    # Plot principal
    plt.subplot(2, 1, 1)
    plt.plot(t, signal, linewidth=0.5)
    plt.title(f'Chirp Lineal ({f_start}Hz → {f_end}Hz)')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.grid(True)

    # Plot del zoom para ver el detalle
    plt.subplot(2, 1, 2)
    zoom_samples = int(0.1 * sample_rate)  # Mostrar 100ms
    plt.plot(t[:zoom_samples], signal[:zoom_samples], linewidth=0.5)
    plt.title('Detalle (primeros 100ms)')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Generar y mostrar espectrograma
    plt.figure(figsize=(15, 5))
    plt.specgram(signal, Fs=sample_rate, NFFT=1024, noverlap=512)
    plt.title('Espectrograma del Chirp')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Frecuencia (Hz)')
    plt.colorbar(label='Intensidad (dB)')
    plt.show()

    return filename


def generate_exponential_chirp(
        duration=5.0,
        f_start=20,
        f_end=20000,
        sample_rate=48000,
        amplitude=0.8,
):
    """
    Genera un chirp exponencial que dedica más tiempo a frecuencias bajas.
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    beta = np.log(f_end / f_start)
    signal = amplitude * np.sin(2 * np.pi * f_start * duration / beta *
                                (np.exp(beta * t / duration) - 1))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'signals/chirp/chirp_exp_{f_start}Hz_{f_end}Hz_{duration}s_amp{amplitude}_{timestamp}.wav'

    signal_normalized = np.int16(signal * 32767)
    wavfile.write(filename, sample_rate, signal_normalized)

    # Visualización similar a la del chirp lineal
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    plt.plot(t, signal, linewidth=0.5)
    plt.title(f'Chirp Exponencial ({f_start}Hz → {f_end}Hz)')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.specgram(signal, Fs=sample_rate, NFFT=1024, noverlap=512)
    plt.title('Espectrograma del Chirp Exponencial')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Frecuencia (Hz)')
    plt.colorbar(label='Intensidad (dB)')

    plt.tight_layout()
    plt.show()

    return filename


def generate_impulse_train(
        duration=5.0,
        interval=1.0,
        sample_rate=48000,
        amplitude=0.8,
):
    """
    Genera una secuencia de impulsos para análisis de respuesta impulsional.
    """
    total_samples = int(duration * sample_rate)
    signal = np.zeros(total_samples)

    # Generar impulsos
    impulse_samples = np.arange(0, total_samples, int(interval * sample_rate))
    signal[impulse_samples] = amplitude

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'signals/impulse/impulse_train_{interval}s_interval_{duration}s_amp{amplitude}_{timestamp}.wav'

    signal_normalized = np.int16(signal * 32767)
    wavfile.write(filename, sample_rate, signal_normalized)

    plt.figure(figsize=(15, 5))
    plt.plot(np.arange(total_samples) / sample_rate, signal)
    plt.title('Tren de Impulsos')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.grid(True)
    plt.show()

    return filename


def generate_white_noise(
        duration=5.0,
        sample_rate=48000,
        amplitude=0.8,
        n_segments=10  # Para promediar el espectro
):
    """
    Genera ruido blanco gaussiano con visualización mejorada.

    Parameters:
    -----------
    duration : float
        Duración en segundos
    sample_rate : int
        Frecuencia de muestreo en Hz
    amplitude : float
        Amplitud (0 a 1)
    n_segments : int
        Número de segmentos para el análisis espectral
    """
    # Generar ruido blanco
    n_samples = int(duration * sample_rate)
    signal = amplitude * np.random.normal(0, 1, n_samples)

    # Crear nombre de archivo descriptivo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'signals/noise/white_noise_{duration}s_amp{amplitude}_{timestamp}.wav'

    # Normalizar y guardar
    signal_normalized = np.int16(signal * 32767)
    wavfile.write(filename, sample_rate, signal_normalized)

    # Visualización
    plt.figure(figsize=(15, 12))

    # Serie temporal
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(n_samples) / sample_rate, signal, linewidth=0.5)
    plt.title('Ruido Blanco - Serie Temporal')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.grid(True)

    # Zoom de la serie temporal
    plt.subplot(3, 1, 2)
    zoom_samples = int(0.01 * sample_rate)  # 10ms
    plt.plot(np.arange(zoom_samples) / sample_rate, signal[:zoom_samples], linewidth=0.5)
    plt.title('Detalle (10ms)')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.grid(True)

    # Densidad espectral de potencia
    plt.subplot(3, 1, 3)
    f, Pxx = sg.welch(signal, sample_rate, nperseg=int(sample_rate / 10))
    plt.semilogx(f, 10 * np.log10(Pxx))
    plt.title('Densidad Espectral de Potencia')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Potencia/Frecuencia (dB/Hz)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Calcular y mostrar estadísticas
    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))
    crest_factor = peak / rms

    print(f"Estadísticas del Ruido Blanco:")
    print(f"RMS: {rms:.3f}")
    print(f"Valor Pico: {peak:.3f}")
    print(f"Factor de Cresta: {crest_factor:.3f}")

    return filename


def generate_amplitude_test(
        frequency=1000,
        durations=[1.0],
        amplitudes=[0.1, 0.3, 0.5, 0.7, 0.9],
        sample_rate=48000,
        fade_duration=0.01  # Duración del fade in/out en segundos
):
    """
    Genera una señal de prueba de amplitud variable con transiciones suaves.

    Parameters:
    -----------
    frequency : float
        Frecuencia de la señal en Hz
    durations : list
        Lista de duraciones para cada amplitud en segundos
    amplitudes : list
        Lista de amplitudes a probar (0 a 1)
    sample_rate : int
        Frecuencia de muestreo en Hz
    fade_duration : float
        Duración de las transiciones suaves en segundos
    """
    # Asegurar que hay una duración para cada amplitud
    if len(durations) == 1:
        durations = durations * len(amplitudes)

    # Generar señal por segmentos
    signals = []
    times = []
    t_offset = 0
    fade_samples = int(fade_duration * sample_rate)

    for amp, dur in zip(amplitudes, durations):
        # Generar segmento de señal
        t = np.linspace(t_offset, t_offset + dur, int(sample_rate * dur))
        segment = amp * np.sin(2 * np.pi * frequency * t)

        # Aplicar fade in/out
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)

        segment[:fade_samples] *= fade_in
        segment[-fade_samples:] *= fade_out

        signals.append(segment)
        times.extend(t)
        t_offset += dur

    # Concatenar todos los segmentos
    final_signal = np.concatenate(signals)

    # Crear nombre de archivo descriptivo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'signals/amplitude_test/amp_test_{frequency}Hz_{len(amplitudes)}levels_{timestamp}.wav'

    # Normalizar y guardar
    signal_normalized = np.int16(final_signal * 32767)
    wavfile.write(filename, sample_rate, signal_normalized)

    # Visualización
    plt.figure(figsize=(15, 12))

    # Serie temporal completa
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(len(final_signal)) / sample_rate, final_signal)
    plt.title(f'Test de Amplitudes - {frequency}Hz')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.grid(True)

    # Zoom en una transición
    transition_start = int(durations[0] * sample_rate) - fade_samples
    transition_end = int(durations[0] * sample_rate) + fade_samples
    t_zoom = np.arange(transition_start, transition_end) / sample_rate

    plt.subplot(3, 1, 2)
    plt.plot(t_zoom, final_signal[transition_start:transition_end])
    plt.title('Detalle de Transición')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.grid(True)

    # Espectrograma
    plt.subplot(3, 1, 3)
    plt.specgram(final_signal, Fs=sample_rate, NFFT=1024, noverlap=512)
    plt.title('Espectrograma')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Frecuencia (Hz)')
    plt.colorbar(label='Intensidad (dB)')

    plt.tight_layout()
    plt.show()

    # Mostrar información de los niveles
    print("\nNiveles de amplitud:")
    for i, (amp, dur) in enumerate(zip(amplitudes, durations), 1):
        print(f"Nivel {i}: {amp:.3f} (duración: {dur:.1f}s)")

    return filename


def analyze_amplitude_linearity(
        input_file,
        output_file,
        n_levels=5,
        sample_rate=48000
):
    """
    Analiza la linealidad de la respuesta en amplitud del sistema.

    Parameters:
    -----------
    input_file : str
        Archivo de entrada con test de amplitudes
    output_file : str
        Archivo de salida correspondiente
    n_levels : int
        Número de niveles de amplitud en el test
    sample_rate : int
        Frecuencia de muestreo
    """
    # Leer archivos
    _, input_signal = wavfile.read(input_file)
    _, output_signal = wavfile.read(output_file)

    # Convertir a float
    input_signal = input_signal / 32767.0
    output_signal = output_signal / 32767.0

    # Calcular RMS por segmento
    samples_per_segment = len(input_signal) // n_levels
    input_rms = []
    output_rms = []

    for i in range(n_levels):
        start = i * samples_per_segment
        end = (i + 1) * samples_per_segment

        input_rms.append(np.sqrt(np.mean(input_signal[start:end] ** 2)))
        output_rms.append(np.sqrt(np.mean(output_signal[start:end] ** 2)))

    # Visualización
    plt.figure(figsize=(10, 8))

    # Gráfico de entrada vs salida
    plt.subplot(2, 1, 1)
    plt.plot(input_rms, output_rms, 'bo-', label='Medido')
    plt.plot([0, max(input_rms)], [0, max(output_rms)], 'r--', label='Lineal Ideal')
    plt.title('Análisis de Linealidad')
    plt.xlabel('Amplitud de Entrada (RMS)')
    plt.ylabel('Amplitud de Salida (RMS)')
    plt.grid(True)
    plt.legend()

    # Gráfico de desviación de la linealidad
    ideal_gain = output_rms[-1] / input_rms[-1]
    deviation = np.array(output_rms) - np.array(input_rms) * ideal_gain

    plt.subplot(2, 1, 2)
    plt.plot(input_rms, deviation, 'go-')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Desviación de la Linealidad')
    plt.xlabel('Amplitud de Entrada (RMS)')
    plt.ylabel('Desviación')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return input_rms, output_rms, deviation


# Ejemplo de uso
if __name__ == "__main__":
    setup_directories()

    generate_white_noise(
        duration=10.0,
        amplitude=0.0,  # Silencio
    )

    # # Generar diferentes tipos de señales de prueba
    # chirp_file = generate_linear_chirp(
    #     duration=20.0,
    #     f_start=20,
    #     f_end=20000,
    #     amplitude=0.8
    # )
    #
    # exp_chirp_file = generate_exponential_chirp(
    #     duration=20.0,
    #     f_start=20,
    #     f_end=20000,
    #     amplitude=0.8
    # )
    #
    #
    # impulse_file = generate_impulse_train(
    #     duration=10.0,
    #     interval=1.0,
    #     amplitude=0.8
    # )
    #
    # chirp_file = generate_linear_chirp(
    #     duration=20.0,
    #     f_start=20,
    #     f_end=20000,
    #     amplitude=0.8
    # )
    #
    # noise_file = generate_white_noise(
    #     duration=5.0,
    #     amplitude=0.5
    # )
    #
    # amp_test_file = generate_amplitude_test(
    #     frequency=1000,
    #     amplitudes=[0.1, 0.3, 0.5, 0.7, 0.9],
    #     durations=[1.0]
    # )
