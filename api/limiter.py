from starlette.requests import Request
from slowapi import Limiter


def _client_ip(request: Request) -> str:
    # Cloud Run's GFE appends the real client IP as the rightmost entry.
    # Taking [-1] prevents spoofing via a client-supplied X-Forwarded-For header.
    # Fall back to request.client.host for local development.
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[-1].strip()
    return request.client.host if request.client else "127.0.0.1"


limiter = Limiter(key_func=_client_ip)
