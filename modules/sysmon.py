"""Lectura de temperatura, frecuencia y throttling en Raspberry Pi 5.

Todas las funciones devuelven None si el dato no esta disponible (por ejemplo,
al probar el codigo en una PC), para que el benchmark no se rompa fuera de la Pi.
"""
import shutil
import subprocess
import time

_THERMAL_PATH = "/sys/class/thermal/thermal_zone0/temp"
_FREQ_PATH = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
_VCGENCMD = shutil.which("vcgencmd")

# Bits "actuales" de get_throttled (los bits 16-19 son historicos desde el arranque,
# por eso NO se usan para decidir si una corrida concreta tuvo throttling).
THROTTLE_NOW_MASK = 0xF
_THROTTLE_BITS = {
    0x1: "subvoltaje",
    0x2: "frecuencia limitada",
    0x4: "throttling activo",
    0x8: "limite de temperatura",
}


def _read_int(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _vcgencmd(*args):
    if not _VCGENCMD:
        return None
    try:
        out = subprocess.run([_VCGENCMD, *args], capture_output=True,
                             text=True, timeout=2)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def read_temp_c():
    """Temperatura del SoC en grados Celsius."""
    raw = _read_int(_THERMAL_PATH)
    return raw / 1000.0 if raw is not None else None


def read_freq_mhz():
    """Frecuencia actual de la CPU en MHz."""
    khz = _read_int(_FREQ_PATH)
    if khz is not None:
        return khz / 1000.0
    out = _vcgencmd("measure_clock", "arm")  # frequency(48)=2400000000
    if out and "=" in out:
        try:
            return float(out.split("=")[1]) / 1e6
        except ValueError:
            return None
    return None


def read_throttled():
    """Mascara de get_throttled (entero) o None."""
    out = _vcgencmd("get_throttled")  # throttled=0x0
    if out and "=" in out:
        try:
            return int(out.split("=")[1], 16)
        except ValueError:
            return None
    return None


def snapshot():
    """Lectura puntual de temperatura, frecuencia y estado de throttling."""
    mask = read_throttled()
    return {
        "temp_c": read_temp_c(),
        "freq_mhz": read_freq_mhz(),
        "throttled_now": (mask & THROTTLE_NOW_MASK) if mask is not None else None,
    }


def describe_throttle(mask):
    """Texto legible para una mascara 'actual' de throttling."""
    if mask is None:
        return "N/D"
    if mask == 0:
        return "ninguno"
    return ", ".join(name for bit, name in _THROTTLE_BITS.items() if mask & bit)


def wait_for_cooldown(target_c, timeout_s=300, poll_s=5):
    """Espera a que el SoC baje de target_c para que cada corrida parta de un
    estado termico comparable. Devuelve (segundos_esperados, temp_final)."""
    temp = read_temp_c()
    if temp is None or target_c is None:
        return 0.0, temp
    t0 = time.time()
    while temp is not None and temp > target_c and (time.time() - t0) < timeout_s:
        time.sleep(poll_s)
        temp = read_temp_c()
    return time.time() - t0, temp
