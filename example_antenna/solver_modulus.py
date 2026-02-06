import sys
import subprocess
from pathlib import Path

def find_solver_script() -> Path:
    """
    Ищем существующий solver в проекте.
    Приоритет: metal.py -> wood.py -> water.py -> time.py
    """
    candidates = ["metal.py", "wood.py", "water.py", "time.py"]
    for name in candidates:
        p = Path(__file__).with_name(name)
        if p.exists():
            return p
    raise FileNotFoundError(
        "Не найден solver. Ожидались файлы: metal.py / wood.py / water.py / time.py "
        "в папке example_antenna."
    )

def main() -> int:
    solver = find_solver_script()

    # Передаём ВСЕ аргументы как есть (например custom.scene_path=scene.yaml)
    cmd = [sys.executable, str(solver), *sys.argv[1:]]

    print(f"[solver_modulus] Using solver: {solver.name}", flush=True)
    print(f"[solver_modulus] Running: {' '.join(cmd)}", flush=True)

    return subprocess.call(cmd)

if __name__ == "__main__":
    raise SystemExit(main())
