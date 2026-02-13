# solver_simple_fast_wave_materials_shadow_diffraction_wavesmooth.py
# Solver: fast analytic wave field with soft shadows + diffraction + spectral smoothing
# Solver: быстрый аналитический расчёт волнового поля с мягкими тенями + дифракцией + спектральным сглаживанием

import argparse
import os
from datetime import datetime
import math
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

try:
    import yaml
except Exception as e:
    raise ImportError("PyYAML is required. Install with: pip install pyyaml") from e


# World boundaries used for mesh generation and coordinate mapping
# Границы мира для генерации сетки и преобразования координат
WORLD = dict(xmin=0.0, xmax=3.0, ymin=0.0, ymax=2.0)

# Default radius for point-objects (treated as disks in the solver)
# Радиус по умолчанию для точек-объектов (в солвере считаются дисками)
DEFAULT_POINT_RADIUS = 0.08

# Default wave speed (c) per material (if not provided in scene.yaml)
# Скорость волны (c) по умолчанию для материалов (если не задано в scene.yaml)
DEFAULT_C_MAP = {
    "air": 3.0e8,
    "asphalt": 2.2e8,
    "brick": 2.0e8,
    "concrete": 2.0e8,
    "foam": 1.3e8,
    "glass": 2.2e8,
    "ice": 3.1e8,
    "metal": 2.5e8,
    "plastic": 2.4e8,
    "rubber": 1.6e8,
    "sand": 1.7e8,
    "water": 1.5e8,
    "wood": 2.0e8,
}

# Symbols used for drawing materials on plots (optional)
# Символы для отображения материалов на графиках (опционально)
MAT_SYMBOLS = {
    "metal": "■",
    "concrete": "▲",
    "brick": "◆",
    "glass": "◇",
    "wood": "●",
    "water": "≈",
    "plastic": "□",
    "rubber": "◉",
    "sand": "⋯",
    "foam": "✚",
    "ice": "✳",
    "asphalt": "▬",
    "air": "·",
}

# Visualization style defaults
# Настройки визуализации по умолчанию
PASTEL_CMAP = "cividis"
PASTEL_ALPHA = 0.93
PASTEL_VMAX_SCALE = 0.55


# ----------------------------
# Small helpers
# Вспомогательные функции
# ----------------------------
def _as_float(v, default: float = 0.0) -> float:
    """
    Convert value to float with fallback.
    Преобразовать значение в float с запасным значением.
    """
     # if v is None or v == "":
    #     return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def load_scene(scene_path: str) -> dict:
    """
    Load scene.yaml and validate structure.
    Загрузить scene.yaml и проверить структуру.
    """
    with open(scene_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "scene" not in data:
        raise ValueError("scene.yaml must contain top-level key: scene")
    return data


def get_material_props(materials: dict, name: str, air_abs: float):
    """
    Get material wave speed c and absorption alpha.
    Получить скорость волны c и поглощение alpha для материала.

    If material is not found, uses defaults (air as fallback).
    Если материал не найден — берутся значения по умолчанию (air как запасной).
    """
       # if not isinstance(name, str) or name == "":
    #     name = "air"
    n = (name or "air").strip().lower()
      # if not isinstance(materials, dict):
    #     materials = {}
    d = materials.get(n, {}) if isinstance(materials, dict) else {}
    c = _as_float(d.get("c", DEFAULT_C_MAP.get(n, DEFAULT_C_MAP["air"])), DEFAULT_C_MAP.get(n, DEFAULT_C_MAP["air"]))
    alpha = _as_float(d.get("absorption", air_abs), air_abs)
    return float(c), float(alpha)


def build_disks(objects, barrier_set):
    """
    Convert scene objects into internal disk representation.
    Преобразовать объекты сцены во внутренний формат дисков.

    Each disk: (cx, cy, r, material, is_barrier).
    Каждый диск: (cx, cy, r, material, is_barrier).
    """
    disks = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        cx = _as_float(obj.get("x", 0.0), 0.0)
        cy = _as_float(obj.get("y", 0.0), 0.0)
        r = abs(_as_float(obj.get("r", DEFAULT_POINT_RADIUS), DEFAULT_POINT_RADIUS))
        mat = str(obj.get("material", "air")).strip().lower()
         # if not mat:
        #     mat = "air"
        is_bar = mat in barrier_set
        disks.append((float(cx), float(cy), float(r), mat, is_bar))
    return disks


# ----------------------------
# Gaussian blur utilities
# Гауссово размытие
# ----------------------------
def gaussian_kernel_1d(kernel_size: int, sigma: float) -> np.ndarray:
    """
    Create normalized 1D Gaussian kernel.
    Создать нормированное 1D ядро Гаусса.
    """
    kernel_size = int(kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    half = kernel_size // 2
    x = np.arange(-half, half + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma)).astype(np.float32)
    k /= (k.sum() + 1e-12)
    return k


def gaussian_blur_2d(img: np.ndarray, sigma: float, kernel_size: int) -> np.ndarray:
    """
    Simple separable Gaussian blur (horizontal + vertical).
    Простое разделимое гауссово размытие (горизонталь + вертикаль).
    """
    if sigma <= 0:
        return img.astype(np.float32)
    k = gaussian_kernel_1d(kernel_size, sigma)
    pad = len(k) // 2

    # Blur X
    # Размытие по X
    tmp = np.pad(img, ((0, 0), (pad, pad)), mode="edge")
    out_x = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[1]):
        window = tmp[:, i:i + len(k)]
        out_x[:, i] = (window * k[None, :]).sum(axis=1)

    # Blur Y
    # Размытие по Y
    tmp2 = np.pad(out_x, ((pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(img, dtype=np.float32)
    for j in range(img.shape[0]):
        window = tmp2[j:j + len(k), :]
        out[j, :] = (window * k[:, None]).sum(axis=0)

    return out


def sigmoid(x):
    """
    Logistic sigmoid function.
    Сигмоида (логистическая функция).
    """
    return 1.0 / (1.0 + np.exp(-x))


# ----------------------------
# Soft shadow (occlusion)
# Мягкая тень (окклюзия)
# ----------------------------
def occlusion_map_geometric(X, Y, x0, y0, disks, occlusion_metal, occlusion_concrete, steps=40):
    """
    Compute geometric occlusion map along rays from source to each grid cell.
    Посчитать карту окклюзии по лучам от источника к каждой точке сетки.

    If a ray intersects a barrier disk:
      - metal -> occlusion_metal
      - other barrier -> occlusion_concrete
    Если луч пересекает барьерный диск:
      - metal -> occlusion_metal
      - другой барьер -> occlusion_concrete
    """
    H, W = X.shape
    occ = np.ones((H, W), dtype=np.float32)

    barrier = [(cx, cy, r, mat) for (cx, cy, r, mat, is_bar) in disks if is_bar]
    if len(barrier) == 0:
        return occ

    steps = max(6, int(steps))
    s = np.linspace(0.0, 1.0, steps, dtype=np.float32)[None, None, :, None]
    src = np.array([x0, y0], dtype=np.float32)[None, None, None, :]
    tgt = np.stack([X, Y], axis=-1)[:, :, None, :]
    samples = src + s * (tgt - src)

    sx = samples[..., 0]
    sy = samples[..., 1]

    occluded = np.zeros((H, W), dtype=bool)
    metal_hit = np.zeros((H, W), dtype=bool)

    for (cx, cy, r, mat) in barrier:
        inside = (sx - cx) ** 2 + (sy - cy) ** 2 <= (r ** 2)
        hit = inside.any(axis=2)
        occluded |= hit
        if mat == "metal":
            metal_hit |= hit

    occ[occluded & metal_hit] = float(occlusion_metal)
    occ[occluded & (~metal_hit)] = float(occlusion_concrete)
    return occ


# ----------------------------
# Diffraction helpers (edge detection)
# Дифракция (вспомогательные функции)
# ----------------------------
def barrier_mask_grid(X, Y, disks):
    """
    Build boolean mask of barrier areas on the grid.
    Построить булеву маску барьеров на сетке.
    """
    mask = np.zeros_like(X, dtype=bool)
    for (cx, cy, r, mat, is_bar) in disks:
        if not is_bar:
            continue
        mask |= (X - cx) ** 2 + (Y - cy) ** 2 <= (r ** 2)
    return mask


def erode_binary(mask: np.ndarray) -> np.ndarray:
    """
    Naive binary erosion (3x3 all-true condition).
    Примитивная эрозия бинарной маски (условие all-true в окрестности 3x3).
    """
    # if X is None or Y is None:
    #     raise ValueError("X and Y grids must not be None")

    # if X.shape != Y.shape:
    #     raise ValueError("X and Y must have the same shape")
    H, W = mask.shape
    m = mask
    out = m.copy()
    out[0, :] = False
    out[-1, :] = False
    out[:, 0] = False
    out[:, -1] = False
    for y in range(1, H - 1):
        row = m[y - 1:y + 2, :]
        for x in range(1, W - 1):
            # if r <= 0:
        #     continue
            if not row[:, x - 1:x + 2].all():
                out[y, x] = False
    return out


def edge_pixels(mask: np.ndarray) -> np.ndarray:
    """
    Compute edge pixels as mask minus eroded(mask).
    Вычислить пиксели границы: mask минус eroded(mask).
    """
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    er = erode_binary(mask)
    return mask & (~er)


def distance_to_edge(edge: np.ndarray, max_dist: int = 2000) -> np.ndarray:
    """
    BFS distance transform (Manhattan distance) from edge pixels.
    BFS-преобразование расстояний (манхэттенское) до пикселей границы.
    """
    H, W = edge.shape
    dist = np.full((H, W), fill_value=max_dist, dtype=np.int32)
    q = deque()

    ys, xs = np.where(edge)
    for y, x in zip(ys, xs):
        dist[y, x] = 0
        q.append((y, x))

    if len(q) == 0:
        return dist.astype(np.float32)

    while q:
        y, x = q.popleft()
        d = dist[y, x] + 1
        if d >= max_dist:
            continue
        if y > 0 and dist[y - 1, x] > d:
            dist[y - 1, x] = d
            q.append((y - 1, x))
        if y < H - 1 and dist[y + 1, x] > d:
            dist[y + 1, x] = d
            q.append((y + 1, x))
        if x > 0 and dist[y, x - 1] > d:
            dist[y, x - 1] = d
            q.append((y, x - 1))
        if x < W - 1 and dist[y, x + 1] > d:
            dist[y, x + 1] = d
            q.append((y, x + 1))

    return dist.astype(np.float32)


# ----------------------------
# Ray-march with smooth boundaries
# Трассировка луча со сглаженными границами
# ----------------------------
def raytrace_field_pretty(
    pts, source_xy, freq_hz, amplitude, materials, disks, air_abs, steps,
    trans_map, edge_smooth_m, batch=16000
):
    """
    Ray-march field computation with smooth disk boundaries.
    Расчёт поля по лучам с "мягкими" границами дисков.

    Uses sigmoid blending to smoothly mix material properties along the path.
    Использует сигмоиду для плавного смешивания свойств материалов вдоль пути.
    """
    P = pts.shape[0]
    U = np.zeros((P,), dtype=np.float32)

    # Cache material props for speed
    # Кэш свойств материалов для ускорения
    mat_cache = {}

    def props(mat):
        if mat not in mat_cache:
            mat_cache[mat] = get_material_props(materials, mat, air_abs)
        return mat_cache[mat]

    steps = max(8, int(steps))
    t = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    S = t.shape[0]
    c_air, a_air = props("air")

    # Edge smoothing parameter (meters)
    # Параметр сглаживания границы (в метрах)
    eps = max(1e-4, float(edge_smooth_m))

    for start in range(0, P, batch):
        end = min(P, start + batch)
        p = pts[start:end]
        B = p.shape[0]

        # Segment vector from source to point
        # Вектор сегмента от источника к точке
        v = p - source_xy[None, :]
        seg_len = np.sqrt((v * v).sum(axis=1) + 1e-12).astype(np.float32)

        # Step length along the ray
        # Длина шага вдоль луча
        ds = (seg_len / max(1, (S - 1))).astype(np.float32)

        # Sample points along ray (sx, sy)
        # Выборка точек вдоль луча (sx, sy)
        sx = source_xy[0] + v[:, 0:1] * t[None, :]
        sy = source_xy[1] + v[:, 1:2] * t[None, :]

        # Start with air properties everywhere
        # Изначально везде свойства воздуха
        c_s = np.full((B, S), np.float32(c_air), dtype=np.float32)
        a_s = np.full((B, S), np.float32(a_air), dtype=np.float32)
        trans_s = np.ones((B, S), dtype=np.float32)

        # Blend each disk material along the ray samples
        # Смешиваем материалы дисков вдоль луча
        for (cx, cy, r, mat, is_bar) in disks:
            dx = sx - cx
            dy = sy - cy
            dist = np.sqrt(dx * dx + dy * dy + 1e-12).astype(np.float32)

            # Soft membership inside disk
            # "Мягкая" принадлежность точек диску
            m = sigmoid((np.float32(r) - dist) / np.float32(eps)).astype(np.float32)

            if float(m.max()) < 1e-4:
                continue

            c_m, a_m = props(mat)
            c_s = c_s * (1.0 - m) + np.float32(c_m) * m
            a_s = a_s * (1.0 - m) + np.float32(a_m) * m

            # Transmission blending: take minimum inside material
            # Смешивание прохождения: берём минимум внутри материала
            tr = np.float32(trans_map.get(mat, 1.0))
            trans_s = trans_s * (1.0 - m) + np.minimum(trans_s, tr) * m

        # Compute accumulated phase and attenuation
        # Считаем суммарную фазу и затухание
        k_s = (2.0 * np.pi * float(freq_hz)) / c_s
        phase = (k_s * ds[:, None]).sum(axis=1)
        decay = (a_s * ds[:, None]).sum(axis=1)

        # Use the strongest transmission constraint along the ray
        # Берём наиболее "жёсткое" прохождение вдоль луча
        trans = trans_s.min(axis=1)

        U[start:end] = (float(amplitude) * np.cos(phase) * np.exp(-decay) * trans).astype(np.float32)

    return U


# ----------------------------
# Wave-like spectral smoothing
# Волнообразное спектральное сглаживание
# ----------------------------
def smoothstep01(x):
    """
    Smoothstep in [0..1].
    Smoothstep в диапазоне [0..1].
    """
    # if x is None:
    #     raise ValueError("x must not be None")
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def spectral_wave_smooth(U: np.ndarray, dx: float, dy: float, target_cyc: float,
                         bw: float, soft: float, keep_low_ratio: float, mix: float) -> np.ndarray:
    """
    Frequency-domain bandpass smoothing with soft edges and optional low-frequency keep.
    Сглаживание в частотной области: полосовой фильтр с мягкими краями и удержанием низких частот.

    mix controls blending between original and filtered signal.
    mix управляет смешиванием исходного и отфильтрованного сигнала.
    """
    mix = float(np.clip(mix, 0.0, 1.0))
    if mix <= 0.0:
        return U.astype(np.float32)

    H, W = U.shape
    fx = np.fft.fftfreq(W, d=dx).astype(np.float32)
    fy = np.fft.fftfreq(H, d=dy).astype(np.float32)
    FX, FY = np.meshgrid(fx, fy)
    R = np.sqrt(FX * FX + FY * FY + 1e-12).astype(np.float32)

    # target spatial cycles for main band
    # целевая пространственная частота для основной полосы
    t = float(max(1e-6, target_cyc))
    low = float(max(0.0, keep_low_ratio) * t)
    b0 = float((1.0 - bw) * t)
    b1 = float((1.0 + bw) * t)

    # Low-frequency keep mask
    # Маска удержания низких частот
    if low > 0:
        low_edge0 = low * (1.0 - soft)
        low_edge1 = low * (1.0 + soft)
        low_mask = 1.0 - smoothstep01((R - low_edge0) / max(1e-6, (low_edge1 - low_edge0)))
    else:
        low_mask = np.zeros_like(R, dtype=np.float32)

    # Band-pass mask with smooth edges
    # Полосовая маска с мягкими краями
    inner0 = b0 * (1.0 - soft)
    inner1 = b0 * (1.0 + soft)
    outer0 = b1 * (1.0 - soft)
    outer1 = b1 * (1.0 + soft)
  # if inner1 <= inner0:
    #     inner1 = inner0 + 1e-6
    # if outer1 <= outer0:
    #     outer1 = outer0 + 1e-6
    inner = smoothstep01((R - inner0) / max(1e-6, (inner1 - inner0)))
    outer = 1.0 - smoothstep01((R - outer0) / max(1e-6, (outer1 - outer0)))
    band_mask = (inner * outer).astype(np.float32)

    mask = np.clip(low_mask + band_mask, 0.0, 1.0).astype(np.float32)

    # Apply mask in frequency domain
    # Применяем маску в частотной области
    F = np.fft.fft2(U.astype(np.float32))
    Uf = np.fft.ifft2(F * mask).real.astype(np.float32)

    # Blend original and filtered
    # Смешиваем исходный и отфильтрованный
    return ((1.0 - mix) * U.astype(np.float32) + mix * Uf).astype(np.float32)


# ----------------------------
# Main field computation (pred/true variants)
# Основной расчёт поля (варианты pred/true)
# ----------------------------
def compute_field_variant(
    X, Y, pts, x0, y0, freq_hz, A,
    materials, disks, trans_map,
    air_abs,
    # base settings
    steps_base, edge_smooth_base,
    occl_metal, occl_conc, shadow_steps_base, blur_sigma_base, blur_kernel_base,
    edge_amp_base, edge_phase, edge_decay, edge_softness,
    spectral_mix_base, spectral_bw, spectral_soft, keep_low_ratio,
    zero_inside_barrier,
    barrier_wall_mask_cache=None,
    # quality ratio 0..1 (pred) or 1.0 (true)
    ratio=1.0,
):
    """
    Compute field with quality scaling (ratio).
    Посчитать поле с масштабированием качества (ratio).

    ratio=1.0 -> "true" (highest quality)
    ratio<1.0 -> "pred" (coarser / more artifacts)
    ratio=1.0 -> "true" (макс. качество)
    ratio<1.0 -> "pred" (грубее / больше артефактов)
    """
    ratio = float(np.clip(ratio, 0.0, 1.0))

    # Make "pred" coarser when ratio is small
    # Делаем "pred" грубее при малом ratio
    steps = max(10, int(round(steps_base * (0.35 + 0.65 * ratio))))
    shadow_steps = max(8, int(round(shadow_steps_base * (0.35 + 0.65 * ratio))))
    blur_sigma = float(blur_sigma_base * (0.30 + 0.70 * ratio))
    blur_kernel = int(max(11, int(round(blur_kernel_base * (0.35 + 0.65 * ratio)))))
    if blur_kernel % 2 == 0:
        blur_kernel += 1

    # Spectral mix is the strongest "quality" knob
    # Спектральное смешивание — самый сильный рычаг "качества"
    spectral_mix = (np.clip(spectral_mix_base * (0.40 + 0.60 * ratio), 0.0, 1.0))

    # Diffraction also grows with ratio
    # Дифракция тоже усиливается с ростом ratio
    edge_amp = float(edge_amp_base * (0.60 + 0.40 * ratio))

    # Distorted field (raytrace)
    # "Искажённое" поле (трассировка лучей)
    U_rt = raytrace_field_pretty(
        pts=pts,
        source_xy=np.array([x0, y0], dtype=np.float32),
        freq_hz=float(freq_hz),
        amplitude=float(A),
        materials=materials,
        disks=disks,
        air_abs=float(air_abs),
        steps=int(steps),
        trans_map=trans_map,
        edge_smooth_m=float(edge_smooth_base),
        batch=16000,
    ).reshape(Y.shape[0], X.shape[1])

    # Soft shadow (geometric occlusion + blur)
    # Мягкая тень (геом. окклюзия + размытие)
    occ_hard = occlusion_map_geometric(
        X, Y, x0, y0, disks,
        occlusion_metal=float(occl_metal),
        occlusion_concrete=float(occl_conc),
        steps=int(shadow_steps),
    )
    occ_soft = gaussian_blur_2d(occ_hard, sigma=float(blur_sigma), kernel_size=int(blur_kernel))
    min_occ = min(float(occl_metal), float(occl_conc), 0.08)
    occ_soft = np.clip(occ_soft, min_occ, 1.0).astype(np.float32)
    U_shadowed = (U_rt * occ_soft).astype(np.float32)

    # Diffraction: compute wall mask and edge distance
    # Дифракция: маска барьера и расстояние до края
    if barrier_wall_mask_cache is None:
        wall = barrier_mask_grid(X, Y, disks)
    else:
        wall = barrier_wall_mask_cache

    edge = edge_pixels(wall)
    dist_px = distance_to_edge(edge, max_dist=5000)
    if float(edge_softness) > 0:
        dist_px = gaussian_blur_2d(dist_px.astype(np.float32), sigma=float(edge_softness), kernel_size=21)

    # Convert pixel distance to meters (approx) using grid cell size
    # Переводим пиксельное расстояние в метры через размер клетки сетки
    xmin, xmax = WORLD["xmin"], WORLD["xmax"]
    ymin, ymax = WORLD["ymin"], WORLD["ymax"]
    nx = X.shape[1]
    ny = X.shape[0]
    dx = (xmax - xmin) / max(1, (nx - 1))
    dy = (ymax - ymin) / max(1, (ny - 1))
    cell = 0.5 * (dx + dy)
    dist_m = dist_px.astype(np.float32) * float(cell)

    # Edge wave term depends on shadow strength
    # Волновая добавка на краю зависит от силы тени
    shadow_strength = (1.0 - occ_soft).astype(np.float32)
    c_air = DEFAULT_C_MAP["air"]
    k_air = 2.0 * math.pi * float(freq_hz) / float(c_air)

    U_edge = (float(edge_amp) * float(A)) * np.cos(k_air * dist_m + float(edge_phase)) * np.exp(
        -float(edge_decay) * dist_m
    )
    U_edge = (U_edge.astype(np.float32) * shadow_strength)

    # Combine raytrace field + diffraction term
    # Итог: поле по лучам + дифракционная добавка
    U = (U_shadowed + U_edge).astype(np.float32)

    # Optionally zero out field inside barriers
    # При необходимости обнуляем поле внутри барьеров
    if zero_inside_barrier:
        U[wall] = 0.0

    # Wave-like spectral smoothing (FFT-based)
    # Спектральное сглаживание (через FFT)
    target_cyc = (k_air / (2.0 * math.pi))
    U = spectral_wave_smooth(
        U=U,
        dx=float(dx),
        dy=float(dy),
        target_cyc=float(target_cyc),
        bw=float(spectral_bw),
        soft=float(spectral_soft),
        keep_low_ratio=float(keep_low_ratio),
        mix=float(spectral_mix),
    )
    return U.astype(np.float32), wall


# ----------------------------
# CLI entry
# Запуск из командной строки
# ----------------------------
def main():
    """
    CLI main: loads scene, computes pred/true/err fields, saves PNG outputs.
    Основной CLI: загружает сцену, считает pred/true/err и сохраняет PNG.
    """
    p = argparse.ArgumentParser(description="Wave-like smoothing solver (GUI compatible): pred vs true vs err")
    p.add_argument("scene_path", type=str)

    # epochs control quality ratio between pred and true
    # epochs управляет отношением качества между pred и true
    p.add_argument("--epochs", type=int, default=2000, help="epochs for field_pred (quality level)")
    p.add_argument("--true_epochs", type=int, default=20000, help="epochs for field_true (ideal)")

    # keep old GUI args (ignored)
    # старые аргументы GUI (игнорируются)
    p.add_argument("--N", type=int, default=0, help="(ignored)")
    p.add_argument("--lr", type=float, default=0.0, help="(ignored)")

    # resolution / ray steps
    # разрешение / кол-во шагов по лучу
    p.add_argument("--nx", type=int, default=820)
    p.add_argument("--ny", type=int, default=540)
    p.add_argument("--steps", type=int, default=52)

    # wave parameters
    # параметры волны
    p.add_argument("--freq_hz", type=float, default=None)
    p.add_argument("--amplitude", type=float, default=None)
    p.add_argument("--air_abs", type=float, default=0.03)

    # smooth disk boundary thickness (meters)
    # толщина сглаживания границы дисков (в метрах)
    p.add_argument("--edge_smooth_m", type=float, default=0.04)

    # which materials act as barriers (occluders)
    # какие материалы считаются барьерами
    p.add_argument("--barrier_materials", nargs="+", default=["metal", "concrete"])
    p.add_argument("--zero_inside_barrier", action="store_true")

    # transmissions per material
    # коэффициенты прохождения для материалов
    p.add_argument("--trans_metal", type=float, default=0.03)
    p.add_argument("--trans_concrete", type=float, default=0.18)
    p.add_argument("--trans_brick", type=float, default=0.45)
    p.add_argument("--trans_glass", type=float, default=0.65)
    p.add_argument("--trans_wood", type=float, default=0.70)
    p.add_argument("--trans_water", type=float, default=0.85)
    p.add_argument("--trans_plastic", type=float, default=0.82)
    p.add_argument("--trans_rubber", type=float, default=0.55)
    p.add_argument("--trans_sand", type=float, default=0.62)
    p.add_argument("--trans_asphalt", type=float, default=0.58)
    p.add_argument("--trans_foam", type=float, default=0.78)
    p.add_argument("--trans_ice", type=float, default=0.75)
    p.add_argument("--trans_air", type=float, default=1.0)

    # soft shadow parameters
    # параметры мягкой тени
    p.add_argument("--occlusion_metal", type=float, default=0.05)
    p.add_argument("--occlusion_concrete", type=float, default=0.25)
    p.add_argument("--shadow_steps", type=int, default=40)
    p.add_argument("--blur_sigma", type=float, default=8.0)
    p.add_argument("--blur_kernel", type=int, default=51)

    # diffraction parameters
    # параметры дифракции
    p.add_argument("--edge_amp", type=float, default=0.10)
    p.add_argument("--edge_phase", type=float, default=1.1)
    p.add_argument("--edge_decay", type=float, default=0.22)
    p.add_argument("--edge_softness", type=float, default=2.0)

    # wave-like spectral smoothing parameters
    # параметры спектрального сглаживания
    p.add_argument("--spectral_mix", type=float, default=0.65)
    p.add_argument("--spectral_bw", type=float, default=0.30)
    p.add_argument("--spectral_soft", type=float, default=0.15)
    p.add_argument("--keep_low_ratio", type=float, default=0.10)

    # visualization parameters
    # параметры визуализации
    # if args.vmax_scale <= 0:
#     raise ValueError(f"vmax_scale must be > 0, got {args.vmax_scale}")
# if not (0.0 <= args.img_alpha <= 1.0):
#     raise ValueError(f"img_alpha must be in [0..1], got {args.img_alpha}")
    p.add_argument("--cmap", type=str, default=PASTEL_CMAP)
    p.add_argument("--vmax_scale", type=float, default=PASTEL_VMAX_SCALE)
    p.add_argument("--img_alpha", type=float, default=PASTEL_ALPHA)
    p.add_argument("--symbols", action="store_true", help="draw material symbols", default=True)

    args = p.parse_args()

    # Clamp epochs and compute quality ratio
    # Ограничиваем epochs и считаем коэффициент качества
    pred_epochs = int(args.epochs)
    true_epochs = int(args.true_epochs)
    pred_epochs = max(100, pred_epochs)
    true_epochs = max(pred_epochs, true_epochs)

    # ratio = pred/true in [0..1]
    # ratio = pred/true в диапазоне [0..1]
    ratio = float(pred_epochs) / float(true_epochs)
    ratio = float(np.clip(ratio, 0.0, 1.0))

    # Load scene data
    # Загружаем сцену
    data = load_scene(args.scene_path)
    scene = data["scene"]
    materials = data.get("materials", {})
    source = scene.get("source", {})

    xmin, xmax = WORLD["xmin"], WORLD["xmax"]
    ymin, ymax = WORLD["ymin"], WORLD["ymax"]

    # Source parameters (position, frequency, amplitude)
    # Параметры источника (позиция, частота, амплитуда)
    x0 = _as_float(source.get("x0", 0.8), 0.8)
    y0 = _as_float(source.get("y0", 1.0), 1.0)
    freq_hz = args.freq_hz if args.freq_hz is not None else _as_float(source.get("frequency_hz", 1e9), 1e9)
    A = args.amplitude if args.amplitude is not None else _as_float(source.get("amplitude", 1.0), 1.0)

    # Scene objects list
    # Список объектов сцены
    objects = scene.get("objects", [])
    if not isinstance(objects, list):
        objects = []

    # Define which materials are barriers and build disks list
    # Определяем материалы-барьеры и собираем диски
    barrier_set = {m.strip().lower() for m in args.barrier_materials}
    disks = build_disks(objects, barrier_set)

    # Transmission coefficients map
    # Карта коэффициентов прохождения
    trans_map = {
        "metal": float(args.trans_metal),
        "concrete": float(args.trans_concrete),
        "brick": float(args.trans_brick),
        "glass": float(args.trans_glass),
        "wood": float(args.trans_wood),
        "water": float(args.trans_water),
        "plastic": float(args.trans_plastic),
        "rubber": float(args.trans_rubber),
        "sand": float(args.trans_sand),
        "asphalt": float(args.trans_asphalt),
        "foam": float(args.trans_foam),
        "ice": float(args.trans_ice),
        "air": float(args.trans_air),
    }

    # Build computational grid
    # Строим вычислительную сетку
    nx, ny = int(args.nx), int(args.ny)
    gx = np.linspace(xmin, xmax, nx, dtype=np.float32)
    gy = np.linspace(ymin, ymax, ny, dtype=np.float32)
    X, Y = np.meshgrid(gx, gy)
    pts = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1).astype(np.float32)

    # Compute TRUE first (ratio=1.0) and cache wall mask
    # Сначала считаем TRUE (ratio=1.0) и кэшируем маску барьеров
    U_true, wall = compute_field_variant(
        X=X, Y=Y, pts=pts,
        x0=x0, y0=y0, freq_hz=freq_hz, A=A,
        materials=materials, disks=disks, trans_map=trans_map,
        air_abs=float(args.air_abs),
        steps_base=int(args.steps),
        edge_smooth_base=float(args.edge_smooth_m),
        occl_metal=float(args.occlusion_metal),
        occl_conc=float(args.occlusion_concrete),
        shadow_steps_base=int(args.shadow_steps),
        blur_sigma_base=float(args.blur_sigma),
        blur_kernel_base=int(args.blur_kernel),
        edge_amp_base=float(args.edge_amp),
        edge_phase=float(args.edge_phase),
        edge_decay=float(args.edge_decay),
        edge_softness=float(args.edge_softness),
        spectral_mix_base=float(args.spectral_mix),
        spectral_bw=float(args.spectral_bw),
        spectral_soft=float(args.spectral_soft),
        keep_low_ratio=float(args.keep_low_ratio),
        zero_inside_barrier=bool(args.zero_inside_barrier),
        barrier_wall_mask_cache=None,
        ratio=1.0,
    )
# # sanity checks (commented out)
# if not (0.0 <= ratio <= 1.0):
#     raise ValueError(f"ratio must be in [0..1], got {ratio}")
# if X.shape != Y.shape:
#     raise ValueError(f"X and Y shapes must match, got {X.shape} vs {Y.shape}")
# if wall is not None and wall.shape != X.shape:
#     raise ValueError(f"wall mask shape must match grid, got {wall.shape} vs {X.shape}")
# if freq_hz <= 0:
#     raise ValueError(f"freq_hz must be > 0, got {freq_hz}")
# if A < 0:
#     raise ValueError(f"A must be >= 0, got {A}")
# if int(args.steps) <= 0:
#     raise ValueError(f"steps must be > 0, got {args.steps}")
# if int(args.shadow_steps) <= 0:
#     raise ValueError(f"shadow_steps must be > 0, got {args.shadow_steps}")
# if float(args.blur_sigma) <= 0:
#     raise ValueError(f"blur_sigma must be > 0, got {args.blur_sigma}")
# if int(args.blur_kernel) < 3 or (int(args.blur_kernel) % 2) == 0:
#     raise ValueError(f"blur_kernel must be odd and >= 3, got {args.blur_kernel}")
# if not (0.0 <= float(args.spectral_mix) <= 1.0):
#     raise ValueError(f"spectral_mix must be in [0..1], got {args.spectral_mix}")
# if float(args.spectral_bw) < 0:
#     raise ValueError(f"spectral_bw must be >= 0, got {args.spectral_bw}")
# if float(args.spectral_soft) < 0:
#     raise ValueError(f"spectral_soft must be >= 0, got {args.spectral_soft}")
# if not (0.0 <= float(args.keep_low_ratio) <= 1.0):
#     raise ValueError(f"keep_low_ratio must be in [0..1], got {args.keep_low_ratio}")
    # Compute PRED with ratio based on epochs
    # Считаем PRED с ratio, зависящим от epochs
    U_pred, _ = compute_field_variant(
        X=X, Y=Y, pts=pts,
        x0=x0, y0=y0, freq_hz=freq_hz, A=A,
        materials=materials, disks=disks, trans_map=trans_map,
        air_abs=float(args.air_abs),
        steps_base=int(args.steps),
        edge_smooth_base=float(args.edge_smooth_m),
        occl_metal=float(args.occlusion_metal),
        occl_conc=float(args.occlusion_concrete),
        shadow_steps_base=int(args.shadow_steps),
        blur_sigma_base=float(args.blur_sigma),
        blur_kernel_base=int(args.blur_kernel),
        edge_amp_base=float(args.edge_amp),
        edge_phase=float(args.edge_phase),
        edge_decay=float(args.edge_decay),
        edge_softness=float(args.edge_softness),
        spectral_mix_base=float(args.spectral_mix),
        spectral_bw=float(args.spectral_bw),
        spectral_soft=float(args.spectral_soft),
        keep_low_ratio=float(args.keep_low_ratio),
        zero_inside_barrier=bool(args.zero_inside_barrier),
        barrier_wall_mask_cache=wall,   # same mask / та же маска
        ratio=ratio,
    )

    # Absolute error field
    # Поле абсолютной ошибки
    U_err = np.abs(U_true - U_pred).astype(np.float32)

    # Output directory with timestamp
    # Папка результатов с таймштампом
    out_dir = os.path.join("simple_results", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    # Unified symmetric scale for pred/true
    # Единая симметричная шкала для pred/true
    vmax = float(max(np.max(np.abs(U_true)), np.max(np.abs(U_pred))) + 1e-12) * float(args.vmax_scale)
    vmin = -vmax

    def draw_symbols():
        """
        Draw material symbols at object locations (if enabled).
        Нарисовать символы материалов в позициях объектов (если включено).
        """
        if not args.symbols:
            return
        for obj in objects:
            ox = _as_float(obj.get("x", 0.0), 0.0)
            oy = _as_float(obj.get("y", 0.0), 0.0)
            mat = str(obj.get("material", "air")).strip().lower()
            sym = MAT_SYMBOLS.get(mat, "?")
            # shadow layer (black)
            # подложка (чёрная)
            plt.text(ox, oy, sym, color="black", fontsize=12, ha="center", va="center", alpha=0.55, weight="bold")
            # top layer (white)
            # верхний слой (белый)
            plt.text(ox, oy, sym, color="white", fontsize=12, ha="center", va="center", alpha=0.90, weight="bold")

    def save(arr, title, fname, symmetric=True):
        """
        Save a field image to PNG using matplotlib.
        Сохранить картинку поля в PNG через matplotlib.

        symmetric=True uses common vmin/vmax for signed fields.
        symmetric=True использует общие vmin/vmax для знаковых полей.
        """
        plt.figure(figsize=(12, 5))
        if symmetric:
            plt.imshow(
                arr,
                extent=[xmin, xmax, ymin, ymax],
                origin="lower",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                alpha=1.0
            )
        else:
            plt.imshow(
                arr,
                extent=[xmin, xmax, ymin, ymax],
                origin="lower",
                cmap=str(args.cmap),
                alpha=float(args.img_alpha)
            )
        plt.colorbar()
        plt.title(title)
        # mark source position
        # отмечаем положение источника
        plt.scatter([x0], [y0], color="red", s=90)
        draw_symbols()
        plt.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
        plt.close()

    # Save required triple: pred / true / err
    # Сохраняем тройку: pred / true / err
    save(U_pred, f"field_pred (epochs={pred_epochs}, ratio={ratio:.3f})  freq={float(freq_hz)/1e9:.2f} GHz", "field_pred.png", symmetric=True)
    save(U_true, f"field_true (true_epochs={true_epochs})  freq={float(freq_hz)/1e9:.2f} GHz", "field_true.png", symmetric=True)
    save(U_err, "field_err = |true - pred|", "field_err.png", symmetric=False)

    # Save metrics text file for debugging / reproducibility
    # Сохраняем метрики для отладки / воспроизводимости
    with open(os.path.join(out_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write("mode=fast_analytic_wave_smoothing_pastel_symbols_pred_true_err\n")
        f.write(f"freq_hz={float(freq_hz)}\n")
        f.write(f"amplitude={float(A)}\n")
        f.write(f"air_abs={float(args.air_abs)}\n")
        f.write(f"steps_base={int(args.steps)}\n")
        f.write(f"edge_smooth_m={float(args.edge_smooth_m)}\n")
        f.write(f"blur_sigma_base={float(args.blur_sigma)}\n")
        f.write(f"blur_kernel_base={int(args.blur_kernel)}\n")
        f.write(f"edge_amp_base={float(args.edge_amp)}\n")
        f.write(f"spectral_mix_base={float(args.spectral_mix)}\n")
        f.write(f"spectral_bw={float(args.spectral_bw)}\n")
        f.write(f"spectral_soft={float(args.spectral_soft)}\n")
        f.write(f"keep_low_ratio={float(args.keep_low_ratio)}\n")
        f.write(f"epochs_pred={pred_epochs}\n")
        f.write(f"epochs_true={true_epochs}\n")
        f.write(f"ratio={ratio}\n")
        f.write(f"objects={len(objects)}\n")
        f.write(f"vmax={vmax}\n")

    # Print output directory so GUI can parse it ("Saved results: ...")
    # Печатаем путь к папке результатов (GUI парсит строку "Saved results: ...")

 
    np.save(os.path.join(out_dir, "U_pred.npy"), U_pred.astype(np.float32))
    np.save(os.path.join(out_dir, "U_true.npy"), U_true.astype(np.float32))
    np.save(os.path.join(out_dir, "U_err.npy"),  U_err.astype(np.float32))

    print("Saved results:", out_dir, flush=True)


if __name__ == "__main__":
    main()
