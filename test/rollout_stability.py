# tests/test_05_rollout_stability.py
import torch

@torch.no_grad()
def test_rollout_500_steps_no_nan(model, init_state):
    s = init_state
    energies = []
    for t in range(500):
        s = model.step(s)          # <-- подставь своё
        assert torch.isfinite(s).all()
        energies.append((s**2).mean().item())

    # грубая проверка: энергия не взорвалась в 1000 раз
    assert max(energies) < 1000.0 * max(energies[0], 1e-12)
