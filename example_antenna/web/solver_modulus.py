
"""
solver_modulus.py

Solver на Modulus Sym, который читает сцену из YAML/JSON и строит u_expr через scene_builder.py.
Параметры сцены задаются пользователем в UI.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import modulus.sym
from modulus.sym.domain import Domain
from modulus.sym.domain.constraint import PointwiseInteriorConstraint
from modulus.sym.geometry.primitives_2d import Rectangle
from modulus.sym.key import Key
from modulus.sym.models.fully_connected import FullyConnectedArch
from modulus.sym.solver import Solver

from scene_builder import Air, Material, RectObject, Source, build_u_expr


def _load_scene(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scene file not found: {path}")

    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml  # pip install pyyaml
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))

    raise ValueError("Scene must be .yaml/.yml or .json")


def _parse_scene(scene_dict: Dict[str, Any]):
    scene = scene_dict["scene"]
    mats = scene_dict["materials"]

    source = Source(**scene["source"])
    air = Air(**scene["air"])

    materials = {name: Material(**params) for name, params in mats.items()}

    objects = [RectObject(**obj) for obj in scene.get("objects", [])]

    return source, air, objects, materials


@modulus.sym.main(config_path="config", config_name="config")
def run(cfg) -> None:
    # В config.yaml добавь параметр: custom.scene_path: "scene.yaml"
    scene_path = cfg.custom.scene_path

    scene_dict = _load_scene(scene_path)
    source, air, objects, materials = _parse_scene(scene_dict)

    # Символьная цель
    u_expr = build_u_expr(source=source, air=air, objects=objects, materials=materials)

    # Геометрия области решения
    geom = Rectangle((0, 0), (3, 2))

    # PINN
    input_keys = [Key("x"), Key("y")]
    output_keys = [Key("u")]
    net = FullyConnectedArch(input_keys=input_keys, output_keys=output_keys)
    nodes = [net.make_node("wave")]

    # Обучение "подгонкой" к u_expr (как у тебя в metal.py/wood.py/water.py)
    wave_constraint = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geom,
        outvar={"u": u_expr},
        batch_size=cfg.batch_size.Interior,
    )

    domain = Domain()
    domain.add_constraint(wave_constraint, "wave_constraint")

    slv = Solver(cfg, domain)
    slv.solve()


if __name__ == "__main__":
    run()
