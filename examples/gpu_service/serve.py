"""A small PyTorch model served over HTTP from a GPU, as a long-running GPUlayer service:

    gpulayer submit --target remote --service --accelerator cuda --gpus 1 \
        --framework torch.cuda --source examples/gpu_service -- python serve.py

The worker sets $PORT and reports host:port; `POST /predict` with a JSON list of 64 numbers
returns the model's output and the GPU time it took.
"""

import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
model = (
    nn.Sequential(
        nn.Linear(64, 2048), nn.GELU(), nn.Linear(2048, 2048), nn.GELU(), nn.Linear(2048, 10)
    )
    .to(device)
    .eval()
    .half()
)
name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # health check and device
        self._send({"device": name})

    def do_POST(self):
        x = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        t0 = time.perf_counter()
        with torch.inference_mode():
            y = model(torch.tensor([x], device=device, dtype=torch.half)).float().cpu()
        self._send(
            {"output": y[0].tolist(), "ms": (time.perf_counter() - t0) * 1000, "device": name}
        )

    def _send(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


print(f"serving on port {os.environ['PORT']} with {name}", flush=True)
ThreadingHTTPServer(("0.0.0.0", int(os.environ["PORT"])), Handler).serve_forever()
