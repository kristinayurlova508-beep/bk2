import streamlit as st
import yaml
from pathlib import Path

st.set_page_config(page_title="EM PINN Scene Builder", layout="centered")

st.title("EM Wave Simulation (PINN)")
st.write("Настройка сцены: частота и объекты")

# --------------------
# Global parameters
# --------------------
frequency_ghz = st.number_input(
    "Частота (GHz)",
    min_value=0.1,
    max_value=100.0,
    value=1.0,
    step=0.1
)

num_objects = st.slider("Количество объектов", 0, 5, 1)

st.divider()

objects = []
materials = ["metal", "wood", "water"]

for i in range(num_objects):
    st.subheader(f"Объект {i+1}")
    col1, col2 = st.columns(2)

    with col1:
        x1 = st.number_input(f"x1_{i}", value=0.5)
        x2 = st.number_input(f"x2_{i}", value=1.0)
        y1 = st.number_input(f"y1_{i}", value=0.5)
        y2 = st.number_input(f"y2_{i}", value=1.0)

    with col2:
        material = st.selectbox(f"Материал_{i}", materials)

    objects.append({
        "x1": x1, "x2": x2,
        "y1": y1, "y2": y2,
        "material": material
    })

st.divider()

if st.button("Save scene.yaml"):
    scene = {
        "scene": {
            "source": {
                "frequency_hz": frequency_ghz * 1e9,
                "x0": 1.0,
                "y0": 1.0,
                "amplitude": 1.0
            },
            "air": {"absorption": 0.1},
            "objects": objects
        },
        "materials": {
            "metal": {"absorption": 5.0, "R": 0.9, "T": 0.1},
            "wood":  {"absorption": 1.0, "R": 0.3, "T": 0.7},
            "water": {"absorption": 0.5, "R": 0.1, "T": 0.9}
        }
    }

    path = Path("scene.yaml")
    with open(path, "w") as f:
        yaml.dump(scene, f)

    st.success("scene.yaml сохранён успешно!")
