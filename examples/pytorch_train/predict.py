"""FunctionJob entrypoint: arguments and return value are plain JSON (no pickling)."""

import torch


def predict(points: list[list[float]]) -> dict:
    """Closed-form least squares fit of the same synthetic problem, then predict."""
    torch.manual_seed(0)
    x = torch.randn(4096, 2)
    y = 3 * x[:, 0] - 2 * x[:, 1] + 0.5
    a = torch.cat([x, torch.ones(4096, 1)], dim=1)
    w = torch.linalg.lstsq(a, y.unsqueeze(1)).solution.squeeze(1)
    p = torch.tensor(points, dtype=torch.float32)
    preds = (p @ w[:2] + w[2]).tolist()
    return {"weights": [round(v, 4) for v in w.tolist()], "predictions": preds}
