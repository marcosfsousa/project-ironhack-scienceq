from starlette.requests import Request
from slowapi import Limiter


def _client_ip(request: Request) -> str:
    # Cloud Run's load balancer sets X-Forwarded-For to the real client IP.
    # Fall back to request.client.host for local development.
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"


limiter = Limiter(key_func=_client_ip)
