# Guía Completa de Medición y Análisis del Sistema

## 1. Generación de Señales de Prueba

### 1.1 Caracterización del Sistema de Medición

1. **Test de Amplitudes**
```python
# Generar señales de 1kHz con diferentes amplitudes
amplitudes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for amp in amplitudes:
    generate_amplitude_test(
        frequency=1000,  # 1kHz
        amplitudes=[amp],
        durations=[5.0],  # 5 segundos por nivel
        filename=f'amp_test_{int(amp*100)}percent.wav'
    )
```

2. **Test de Ruido del Sistema**
```python
# Generar silencio para medir ruido del sistema
generate_white_noise(
    duration=10.0,
    amplitude=0.0,  # Silencio
    filename='system_noise_test.wav'
)
```

### 1.2 Señales para Caracterización en Frecuencia

1. **Chirps con diferentes duraciones**
```python
durations = [1, 2, 5, 10, 20]
for dur in durations:
    generate_linear_chirp(
        duration=dur,
        f_start=20,
        f_end=20000,
        amplitude=0.5,  # Amplitud moderada
        filename=f'chirp_{dur}s.wav'
    )
```

2. **Ruido Blanco para diferentes ventanas temporales**
```python
durations = [5, 10, 20]
for dur in durations:
    generate_white_noise(
        duration=dur,
        amplitude=0.5,
        filename=f'white_noise_{dur}s.wav'
    )
```

## 2. Protocolo de Medición

### 2.1 Caracterización del Sistema de Medición

1. **Medición del Ruido del Sistema**
- **¿Qué es?** El piso de ruido y las interferencias inherentes del sistema
- **¿Por qué?** Define el límite inferior de mediciones confiables
- **Procedimiento:**
  * Grabar la salida sin ninguna señal de entrada
  * Grabar con entrada en silencio
  * Registrar estas mediciones como referencia del piso de ruido

2. **Test de Linealidad**
- **¿Qué es?** Verificación de la proporcionalidad entrada/salida
- **¿Por qué?** Determina el rango útil de trabajo
- **Procedimiento:**
  * Reproducir las señales de test de amplitud en orden ascendente
  * Grabar la salida para cada nivel
  * IMPORTANTE: Anotar el nivel donde se observa distorsión o saturación

3. **Test de Estabilidad Temporal**
- **¿Qué es?** Verificación de la consistencia temporal del sistema
- **¿Por qué?** Asegura mediciones repetibles y confiables
- **Procedimiento:**
  * Grabar la respuesta a un chirp al inicio de la sesión
  * Repetir la misma medición al final
  * IMPORTANTE: Registrar cualquier cambio en las condiciones

### 2.2 Mediciones Principales

1. **Mediciones con Chirp**
- **¿Qué es?** Barrido en frecuencia para caracterización espectral
- **¿Por qué?** Ofrece mejor SNR y control frecuencial
- **Procedimiento:**
  * Comenzar con chirp de duración media (5s)
  * Si hay problemas de SNR → usar duraciones más largas
  * Si hay problemas de estabilidad → usar duraciones más cortas
  * IMPORTANTE: Registrar comportamientos anómalos

2. **Mediciones con Ruido Blanco**
- **¿Qué es?** Excitación simultánea de todas las frecuencias
- **¿Por qué?** Menos sensible a no linealidades, mejor para promediar
- **Procedimiento:**
  * Realizar múltiples mediciones (mínimo 3) para cada duración
  * Mantener condiciones constantes entre mediciones
  * Esperar unos segundos entre mediciones
  * IMPORTANTE: Notar cualquier variación significativa

## 3. Consideraciones Importantes

### 3.1 Frecuencia de Muestreo
- Registrar la frecuencia de muestreo utilizada
- Verificar que es al menos el doble de la máxima frecuencia de interés
- Considerar limitaciones del equipo

### 3.2 Control de Amplitudes
- Comenzar con amplitudes moderadas (0.5)
- Ajustar según la respuesta observada:
  * Si hay ruido excesivo → aumentar amplitud
  * Si hay distorsión → reducir amplitud
- Documentar niveles óptimos encontrados

### 3.3 Control de Tiempo
- Registrar:
  * Duración total de la sesión
  * Tiempos entre mediciones
  * Cambios en condiciones ambientales

### 3.4 Señales a Grabar
1. Ruido del sistema (sin entrada)
2. Respuesta a diferentes amplitudes
3. Chirps de diferentes duraciones
4. Múltiples mediciones de ruido blanco
5. Mediciones de verificación al final

### 3.5 Observaciones Importantes a Registrar
- Niveles de saturación
- Comportamientos no lineales
- Inestabilidades o variaciones temporales
- Problemas o anomalías durante la medición
- Condiciones que puedan afectar las mediciones