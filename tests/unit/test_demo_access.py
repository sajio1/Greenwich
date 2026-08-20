import base64
import json
from dataclasses import dataclass
from urllib.parse import urlsplit

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import FastAPI

from alphamotion.service.access import DemoAccessMiddleware


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _b64e(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode().rstrip("=")


@dataclass
class _Response:
    status_code: int
    headers: dict[str, str]
    content: bytes

    @property
    def text(self):
        return self.content.decode()

    def json(self):
        return json.loads(self.content)


class _ASGIClient:
    """Tiny ASGI harness; current Starlette TestClient requires httpx2."""

    def __init__(self, app):
        self.app = app
        self.cookie = ""

    async def request(self, method: str, url: str, *, json_body=None):
        parsed = urlsplit(url)
        body = json.dumps(json_body).encode() if json_body is not None else b""
        headers = [(b"host", b"test")]
        if json_body is not None:
            headers.append((b"content-type", b"application/json"))
        if self.cookie:
            headers.append((b"cookie", self.cookie.encode()))
        received = False
        sent = []

        async def receive():
            nonlocal received
            if received:
                return {"type": "http.disconnect"}
            received = True
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message):
            sent.append(message)

        await self.app({
            "type": "http", "asgi": {"version": "3.0"},
            "http_version": "1.1", "scheme": "http", "method": method,
            "path": parsed.path, "raw_path": parsed.path.encode(),
            "query_string": parsed.query.encode(), "root_path": "",
            "headers": headers, "client": ("test", 1),
            "server": ("test", 80),
        }, receive, send)
        start = next(item for item in sent
                     if item["type"] == "http.response.start")
        response_headers = {
            key.decode().lower(): value.decode()
            for key, value in start.get("headers", [])
        }
        if "set-cookie" in response_headers:
            self.cookie = response_headers["set-cookie"].split(";", 1)[0]
        content = b"".join(item.get("body", b"") for item in sent
                           if item["type"] == "http.response.body")
        return _Response(start["status"], response_headers, content)

    async def get(self, url: str, **_kwargs):
        return await self.request("GET", url)

    async def post(self, url: str, *, json=None):
        return await self.request("POST", url, json_body=json)


async def _device_payload(client: _ASGIClient, token: str, name: str):
    private = ec.generate_private_key(ec.SECP256R1())
    public = private.public_key().public_numbers()
    jwk = {
        "kty": "EC", "crv": "P-256",
        "x": _b64e(public.x.to_bytes(32, "big")),
        "y": _b64e(public.y.to_bytes(32, "big")),
    }
    challenge = (await client.post(
        "/api/access/challenge", json={"public_key": jwk})).json()
    nonce = base64.urlsafe_b64decode(
        challenge["nonce"] + "=" * (-len(challenge["nonce"]) % 4))
    signature = private.sign(nonce, ec.ECDSA(hashes.SHA256()))
    return {
        "token": token, "device_name": name, "public_key": jwk,
        "challenge_id": challenge["challenge_id"],
        "signature": _b64e(signature),
    }


def _app(monkeypatch, tmp_path):
    monkeypatch.setenv("ALPHAMOTION_ACCESS_TOKEN", "shared-demo-token")
    monkeypatch.setenv("ALPHAMOTION_ACCESS_MAX_DEVICES", "3")
    monkeypatch.setenv("ALPHAMOTION_ACCESS_REGISTRY",
                       str(tmp_path / "access.json"))
    monkeypatch.setenv("ALPHAMOTION_COOKIE_SECURE", "0")
    app = FastAPI()
    app.add_middleware(DemoAccessMiddleware)

    @app.get("/api/private")
    async def private():
        return {"ok": True}

    @app.get("/")
    async def home():
        return {"studio": True}

    return app


@pytest.mark.anyio
async def test_shared_token_registers_three_devices_and_rejects_fourth(
        monkeypatch, tmp_path):
    client = _ASGIClient(_app(monkeypatch, tmp_path))
    assert (await client.get("/", follow_redirects=False)).status_code == 307
    assert (await client.get("/api/private")).status_code == 401

    registered = []
    for index in range(3):
        payload = await _device_payload(
            client, "shared-demo-token", f"Demo laptop {index + 1}")
        response = await client.post("/api/access/register", json=payload)
        assert response.status_code == 200, response.text
        registered.append(response.json()["device"])
    assert (await client.get("/api/private")).json() == {"ok": True}

    fourth = await _device_payload(
        client, "shared-demo-token", "Fourth laptop")
    response = await client.post("/api/access/register", json=fourth)
    assert response.status_code == 409
    assert "3 registered devices" in response.json()["detail"]

    listed = await client.post("/api/access/devices",
                               json={"token": "shared-demo-token"})
    assert [item["name"] for item in listed.json()["devices"]] == [
        "Demo laptop 1", "Demo laptop 2", "Demo laptop 3"]

    revoked = await client.post("/api/access/revoke", json={
        "token": "shared-demo-token",
        "device_id": registered[0]["id"],
    })
    assert revoked.status_code == 200
    replacement = await _device_payload(
        client, "shared-demo-token", "Replacement laptop")
    assert (await client.post(
        "/api/access/register", json=replacement)).status_code == 200


@pytest.mark.anyio
async def test_access_rejects_bad_token_and_bad_signature(monkeypatch, tmp_path):
    client = _ASGIClient(_app(monkeypatch, tmp_path))
    payload = await _device_payload(client, "wrong", "Unknown")
    assert (await client.post(
        "/api/access/register", json=payload)).status_code == 401

    payload = await _device_payload(client, "shared-demo-token", "Unknown")
    payload["signature"] = _b64e(b"not-a-signature")
    assert (await client.post(
        "/api/access/register", json=payload)).status_code == 400
