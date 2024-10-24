import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt


def generate_linear_chirp(
        duration=5.0,  # Duración en segundos
        f_start=20,  # Frecuencia inicial en Hz
        f_end=20000,  # Frecuencia final en Hz
        sample_rate=48000,  # Frecuencia de muestreo en Hz
        amplitude=0.8,  # Amplitud (0 a 1)
        filename='chirp.wav'  # Nombre del archivo de salida
):
    """
    Genera un chirp lineal y lo guarda como archivo WAV.

    Parameters:
    -----------
    duration : float
        Duración de la señal en segundos
    f_start : float
        Frecuencia inicial en Hz
    f_end : float
        Frecuencia final en Hz
    sample_rate : int
        Frecuencia de muestreo en Hz
    amplitude : float
        Amplitud de la señal (0 a 1)
    filename : str
        Nombre del archivo de salida

    Returns:
    --------
    tuple
        (tiempos, señal) para visualización
    """
    # Generar vector de tiempo
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Calcular la tasa de cambio de frecuencia
    k = (f_end - f_start) / duration

    # Generar el chirp
    signal = amplitude * np.sin(2 * np.pi * (f_start * t + (k / 2) * t ** 2))

    # Normalizar y convertir a int16
    signal_normalized = np.int16(signal * 32767)

    # Guardar como WAV
    wavfile.write(filename, sample_rate, signal_normalized)

    # Crear gráfico
    plt.figure(figsize=(12, 4))
    plt.plot(t, signal)
    plt.title('Chirp Lineal')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.grid(True)
    plt.show()

    return t, signal


def generate_white_noise(
        duration=5.0,  # Duración en segundos
        sample_rate=48000,  # Frecuencia de muestreo en Hz
        amplitude=0.8,  # Amplitud (0 a 1)
        filename='noise.wav'  # Nombre del archivo de salida
):
    """
    Genera ruido blanco gaussiano y lo guarda como archivo WAV.

    Parameters:
    -----------
    duration : float
        Duración de la señal en segundos
    sample_rate : int
        Frecuencia de muestreo en Hz
    amplitude : float
        Amplitud de la señal (0 a 1)
    filename : str
        Nombre del archivo de salida

    Returns:
    --------
    tuple
        (tiempos, señal) para visualización
    """
    # Generar vector de tiempo
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Generar ruido blanco gaussiano
    signal = amplitude * np.random.normal(0, 1, len(t))

    # Normalizar y convertir a int16
    signal_normalized = np.int16(signal * 32767)

    # Guardar como WAV
    wavfile.write(filename, sample_rate, signal_normalized)

    # Crear gráfico
    plt.figure(figsize=(12, 4))
    plt.plot(t, signal)
    plt.title('Ruido Blanco')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.grid(True)
    plt.show()

    return t, signal


def generate_amplitude_test(
        frequency=1000,  # Frecuencia de la señal en Hz
        durations=[1.0],  # Lista de duraciones para cada amplitud
        amplitudes=[0.1, 0.3, 0.5, 0.7, 0.9],  # Lista de amplitudes a probar
        sample_rate=48000,  # Frecuencia de muestreo en Hz
        filename='amp_test.wav'  # Nombre del archivo de salida
):
    """
    Genera una señal senoidal que varía su amplitud por intervalos.
    Útil para probar la linealidad del sistema y distorsión.

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
    filename : str
        Nombre del archivo de salida

    Returns:
    --------
    tuple
        (tiempos, señal) para visualización
    """
    # Asegurarse que hay una duración para cada amplitud
    if len(durations) == 1:
        durations = durations * len(amplitudes)

    # Generar señal por segmentos
    signals = []
    times = []
    t_offset = 0

    for amp, dur in zip(amplitudes, durations):
        t = np.linspace(t_offset, t_offset + dur, int(sample_rate * dur))
        signal = amp * np.sin(2 * np.pi * frequency * t)
        signals.append(signal)
        times.extend(t)
        t_offset += dur

    # Concatenar todos los segmentos
    final_signal = np.concatenate(signals)

    # Normalizar y convertir a int16
    signal_normalized = np.int16(final_signal * 32767)

    # Guardar como WAV
    wavfile.write(filename, sample_rate, signal_normalized)

    # Crear gráfico
    plt.figure(figsize=(12, 4))
    plt.plot(times, final_signal)
    plt.title('Test de Amplitudes Variables')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.grid(True)
    plt.show()

    return times, final_signal


# Ejemplo de uso:
if __name__ == "__main__":
    # Generar chirp
    t_chirp, signal_chirp = generate_linear_chirp(
        duration=20.0,
        f_start=40,
        f_end=2000,
        amplitude=0.8,
        filename='chirp_test.wav'
    )

    # Generar ruido blanco
    # t_noise, signal_noise = generate_white_noise(
    #     duration=5.0,
    #     amplitude=0.1,
    #     filename='noise_test.wav'
    # )
    #
    # # Generar test de amplitudes
    # t_amp, signal_amp = generate_amplitude_test(
    #     frequency=1000,
    #     durations=[1.0],
    #     amplitudes=[0.1, 0.3, 0.5, 0.7, 0.9],
    #     filename='amplitude_test.wav'
    # )