"""Exception hierarchy shared by SDK, control plane and workers."""


class GpuLayerError(Exception):
    """Base class. ``code`` is stable and sent over the wire."""

    code = "error"
    http_status = 500


class ValidationFailed(GpuLayerError):
    code = "invalid_request"
    http_status = 400


class NotFound(GpuLayerError):
    code = "not_found"
    http_status = 404


class Unauthorized(GpuLayerError):
    code = "unauthorized"
    http_status = 401


class Forbidden(GpuLayerError):
    code = "forbidden"
    http_status = 403


class Conflict(GpuLayerError):
    code = "conflict"
    http_status = 409


class ProtocolMismatch(GpuLayerError):
    code = "protocol_mismatch"
    http_status = 400


ERRORS_BY_CODE = {
    c.code: c
    for c in (
        GpuLayerError,
        ValidationFailed,
        NotFound,
        Unauthorized,
        Forbidden,
        Conflict,
        ProtocolMismatch,
    )
}
