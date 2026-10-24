"""Bounded transport-only retry for saved judge requests. No checkpoint changes."""
import http.client
import json
import socket
import ssl
import time
import urllib.error
import urllib.request

def transient(exc):
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in (408, 429, 502, 503, 504)
    if isinstance(exc, urllib.error.URLError):
        return transient(exc.reason)
    if isinstance(exc, ssl.SSLError):
        return False
    return isinstance(exc, (TimeoutError, socket.timeout, ConnectionError, http.client.RemoteDisconnected, http.client.IncompleteRead))

def post_batch(payload, *, base_url, api_key, timeout=1200.0):
    if timeout != 1200.0:
        raise ValueError("Recovery request timeout must be 1200 seconds")
    base = base_url.rstrip("/").removesuffix("/v1")
    request = urllib.request.Request(base + "/v1/chat/completions/batch", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}, method="POST")
    health = urllib.request.Request(base + "/health", headers={"Authorization": f"Bearer {api_key}"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as handle:
                return json.loads(handle.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as exc:
            if attempt == 2 or not transient(exc):
                raise
            time.sleep((5, 15)[attempt])
            # A dead/unhealthy server ends this job; no watchdog restarts hide errors.
            with urllib.request.urlopen(health, timeout=10) as handle:
                if handle.status != 200:
                    raise RuntimeError("Judge server health check failed") from exc
