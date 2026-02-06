
"""
scene_builder.py

Чистый модуль для построения аналитического выражения поля u(x,y) для произвольного числа объектов.
Используется в solver'е (Modulus) и может использоваться UI для валидации.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from sympy import And, Piecewise, exp, pi, sin, sqrt, symbols

# Скорость света в вакууме (м/с)
C0 = 299_792_458.0

# Символы, которые использует Modulus (Key("x"), Key("y"))
x, y = symbols("x y", real=True)


@dataclass(frozen=True)
class Material:
    """Параметры материала для упрощённой модели."""
    absorption: float  # коэффициент затухания
    R: float           # отражение
    T: float           # прохождение


@dataclass(frozen=True)
class RectObject:
    """Прямоугольный объект."""
    x1: float
    x2: float
    y1: float
    y2: float
    material: str


@dataclass(frozen=True)
class Source:
    """Источник/антенна."""
    x0: float
    y0: float
    amplitude: float
    frequency_hz: float


@dataclass(frozen=True)
class Air:
    """Параметры воздуха."""
    absorption: float


def _in_rect(obj: RectObject):
    # Важно: Piecewise у sympy работает с булевыми условиями
    return And(x >= obj.x1, x <= obj.x2, y >= obj.y1, y <= obj.y2)


def build_u_expr(
    source: Source,
    air: Air,
    objects: List[RectObject],
    materials: Dict[str, Material],
):
    """
    Строит символьное выражение u(x,y) как Piecewise:
      - внутри каждого прямоугольника: u_obj(...)
      - иначе: u_air

    Это "демо-цель" для обучения PINN через PointwiseInteriorConstraint(outvar={"u": u_expr}).
    """
    # Частота -> длина волны -> волновое число
    lam = C0 / float(source.frequency_hz)
    k = 2 * pi / lam

    # Воздух: прямая волна от источника
    r_air = sqrt((x - source.x0) ** 2 + (y - source.y0) ** 2)
    u_air = (
        source.amplitude
        * sin(k * r_air)
        * exp(-air.absorption * r_air)
    )

    def u_inside(obj: RectObject):
        m = materials[obj.material]

        # "центр" объекта как простая точка рассеяния
        xc = (obj.x1 + obj.x2) / 2
        yc = (obj.y1 + obj.y2) / 2
        r_obj = sqrt((x - xc) ** 2 + (y - yc) ** 2)

        # Трансмиссия: ослабленная версия прямой волны (затухание материала)
        u_trans = (
            source.amplitude * m.T * sin(k * r_air) * exp(-m.absorption * r_air)
        )

        # Рассеяние от объекта (очень упрощенно, но визуально работает)
        sc_amp = source.amplitude * (m.absorption / (m.absorption + 1.0))
        u_scat = sc_amp * sin(k * r_obj) * exp(-m.absorption * r_obj)

        # Отражение (также грубо): добавляем отрицательный вклад от прямой волны
        u_ref = -source.amplitude * m.R * sin(k * r_air) * exp(-air.absorption * r_air)

        return u_trans + u_scat + u_ref

    # Собираем Piecewise. Приоритет: первый попавшийся (по порядку).
    # Если нужно, чтобы "последний перекрывал предыдущие", просто переверни список objects.
    pieces = [(u_inside(obj), _in_rect(obj)) for obj in objects]
    pieces.append((u_air, True))
    return Piecewise(*pieces)
