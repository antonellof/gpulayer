"""Self-signed TLS for `gpulayer up`: generated once, pinned by clients via its fingerprint."""

from __future__ import annotations

import hashlib
import os
import shutil
import ssl
import subprocess
from pathlib import Path

from gpulayer.errors import GpuLayerError


def certificate_fingerprint(cert_path: str | Path) -> str:
    der = ssl.PEM_cert_to_DER_cert(Path(cert_path).read_text())
    return "sha256:" + hashlib.sha256(der).hexdigest()


def ensure_self_signed(directory: str | Path, common_name: str = "gpulayer") -> tuple[Path, Path]:
    """Return (cert, key) in ``directory``, creating an ECDSA P-256 pair if missing."""
    d = Path(directory)
    cert, key = d / "cert.pem", d / "key.pem"
    if cert.exists() and key.exists():
        return cert, key
    openssl = shutil.which("openssl")
    if openssl is None:
        raise GpuLayerError(
            "openssl is needed to create a TLS certificate; install it, or pass "
            "--tls-cert/--tls-key, or use --insecure-http on a trusted network"
        )
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    subprocess.run(
        [
            openssl,
            "req",
            "-x509",
            "-newkey",
            "ec",
            "-pkeyopt",
            "ec_paramgen_curve:prime256v1",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "825",
            "-subj",
            f"/CN={common_name}",
        ],
        check=True,
        capture_output=True,
    )
    os.chmod(key, 0o600)
    return cert, key
