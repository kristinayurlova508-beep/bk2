
"""
streamlit_app.py

Простой UI на Streamlit:
- выбор частоты
- выбор количества объектов N
- карточки объектов (координаты + материал)
- генерация scene.yaml
- запуск solver через subprocess (опционально)

Запуск:
  streamlit run streamlit_app.py
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List

import streamlit as st
import yaml


DEFAULT_MATERIALS = {
    "metal": {"absorption": 5.0, "R": 0.9, "T": 0.1},
    "wood":  {"absorption": 1.0, "R": 0.3, "T": 0.7},
    "water": {"absorption": 0.5, "R": 0.1, "T": 0.9},
}

st.set_page_config(page_title="PINN EM Scene Builder", layout="wide")

st.title("Прототип: сцена для PINN (много объектов + частота)")

with st.sidebar:
    st.header("Параметры сигнала")
    freq_ghz = st.number_input("Частота (ГГц)", min_value=0.01, max_value=100.0, value=1.0, step=0.1)
    amplitude = st.number_input("Амплитуда", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    x0 = st.number_input("Источник x0", value=1.0, step=0.1)
    y0 = st.number_input("Источник y0", value=1.0, step=0.1)

    st.header("Воздух")
    air_abs = st.number_input("Absorption (air)", min_value=0.0, max_value=10.0, value=0.1, step=0.05)

    st.header("Объекты")
    n = st.number_input("Количество объектов N", min_value=0, max_value=50, value=3, step=1)

st.divider()

# Храним объекты в session_state
if "objects" not in st.session_state:
    st.session_state["objects"] = []

# Подгоняем список под N
objs: List[Dict] = st.session_state["objects"]
while len(objs) < n:
    objs.append({"x1": 0.5, "x2": 1.0, "y1": 0.5, "y2": 1.0, "material": "wood"})
while len(objs) > n:
    objs.pop()

cols = st.columns(2)

with cols[0]:
    st.subheader("Материалы")
    st.caption("Можно редактировать параметры материалов (абсорбция, отражение R, прохождение T).")
    materials = {}
    for name, params in DEFAULT_MATERIALS.items():
        with st.expander(f"Материал: {name}", expanded=False):
            absorption = st.number_input(f"{name}.absorption", value=float(params["absorption"]), key=f"m_{name}_abs", step=0.1)
            R = st.number_input(f"{name}.R", value=float(params["R"]), key=f"m_{name}_R", step=0.05, min_value=0.0, max_value=1.0)
            T = st.number_input(f"{name}.T", value=float(params["T"]), key=f"m_{name}_T", step=0.05, min_value=0.0, max_value=1.0)
            materials[name] = {"absorption": float(absorption), "R": float(R), "T": float(T)}

with cols[1]:
    st.subheader("Список объектов")
    st.caption("Каждый объект — прямоугольник: x1<x2, y1<y2 и выбранный материал.")

    for i, obj in enumerate(objs):
        with st.expander(f"Объект #{i+1}", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                obj["material"] = st.selectbox("Материал", list(materials.keys()), index=list(materials.keys()).index(obj["material"]) if obj["material"] in materials else 0, key=f"obj_{i}_mat")
            with c2:
                obj["x1"] = st.number_input("x1", value=float(obj["x1"]), key=f"obj_{i}_x1", step=0.1)
                obj["x2"] = st.number_input("x2", value=float(obj["x2"]), key=f"obj_{i}_x2", step=0.1)
            with c3:
                obj["y1"] = st.number_input("y1", value=float(obj["y1"]), key=f"obj_{i}_y1", step=0.1)
                obj["y2"] = st.number_input("y2", value=float(obj["y2"]), key=f"obj_{i}_y2", step=0.1)

st.divider()

scene = {
    "scene": {
        "source": {
            "x0": float(x0),
            "y0": float(y0),
            "amplitude": float(amplitude),
            "frequency_hz": float(freq_ghz) * 1e9,
        },
        "air": {"absorption": float(air_abs)},
        "objects": [
            {
                "x1": float(o["x1"]),
                "x2": float(o["x2"]),
                "y1": float(o["y1"]),
                "y2": float(o["y2"]),
                "material": str(o["material"]),
            }
            for o in objs
        ],
    },
    "materials": materials,
}

cA, cB, cC = st.columns([1, 1, 2])

with cA:
    if st.button("💾 Сохранить scene.yaml", use_container_width=True):
        out = Path("scene.yaml")
        out.write_text(yaml.safe_dump(scene, sort_keys=False, allow_unicode=True), encoding="utf-8")
        st.success(f"Сохранено: {out.resolve()}")

with cB:
    st.download_button(
        "⬇️ Скачать scene.yaml",
        data=yaml.safe_dump(scene, sort_keys=False, allow_unicode=True),
        file_name="scene.yaml",
        mime="text/yaml",
        use_container_width=True,
    )

with cC:
    st.info("Опционально: можно запускать solver_modulus.py из UI (если окружение настроено).")

# Опциональный запуск (если Modulus + config доступны)
run_solver = st.checkbox("Запускать solver_modulus.py после сохранения", value=False)

if run_solver and st.button("🚀 Run solver", use_container_width=True):
    Path("scene.yaml").write_text(yaml.safe_dump(scene, sort_keys=False, allow_unicode=True), encoding="utf-8")
    try:
        # Команда зависит от твоего проекта. Пример:
        # python solver_modulus.py custom.scene_path=scene.yaml
        cmd = ["python", "solver_modulus.py", "custom.scene_path=scene.yaml"]
        st.code(" ".join(cmd))
        p = subprocess.run(cmd, capture_output=True, text=True)
        st.text(p.stdout[-4000:])
        if p.returncode != 0:
            st.error(p.stderr[-4000:])
    except Exception as e:
        st.error(str(e))
