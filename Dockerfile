FROM nvidia/cuda:12.3.2-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Python 3.10 (Ubuntu 22.04), компилятор и базовые либы
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    build-essential \
    git ca-certificates \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# ВАЖНО: Modulus Sym старый плохо дружит с NumPy 2.x -> фиксируем NumPy < 2
RUN pip install "numpy<2" "setuptools<70" wheel

# PyTorch под CUDA 12.1 (работает и на твоём драйвере)
RUN pip install --index-url https://download.pytorch.org/whl/cu121 torch

# Ставим Modulus Sym ТАК, чтобы появился `import modulus.sym`
# Берём стабильный wheel из NVIDIA PyPI (а не кривые sdist с pypi.org)
RUN pip install --extra-index-url https://pypi.nvidia.com "nvidia-modulus-sym==1.6.0"

# Частые зависимости (если твоему проекту надо)
RUN pip install sympy scipy pandas pyvista

WORKDIR /workspace/project
