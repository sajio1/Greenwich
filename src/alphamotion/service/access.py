"""Small-demo access gate with one or more tokens and device slots.

The browser never receives the Hugging Face service token. It exchanges the
operator-provided demo token plus a non-exportable WebCrypto device signature
for an HttpOnly session cookie. The device registry is durable under
``ALPHAMOTION_DATA`` so EBS/Space restarts do not release slots unexpectedly.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from http.cookies import SimpleCookie
from pathlib import Path
from urllib.parse import parse_qs, quote

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

from ..paths import data_dir

COOKIE = "alphamotion_demo"


def _b64e(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode().rstrip("=")


def _b64d(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _canonical_jwk(value: object) -> bytes:
    if not isinstance(value, dict):
        raise ValueError("device public key is required")
    if value.get("kty") != "EC" or value.get("crv") != "P-256":
        raise ValueError("device key must be ECDSA P-256")
    x, y = value.get("x"), value.get("y")
    if not isinstance(x, str) or not isinstance(y, str):
        raise ValueError("device key coordinates are missing")
    if len(_b64d(x)) != 32 or len(_b64d(y)) != 32:
        raise ValueError("device key coordinates are invalid")
    return json.dumps({"crv": "P-256", "kty": "EC", "x": x, "y": y},
                      sort_keys=True, separators=(",", ":")).encode()


def _fingerprint(jwk: object) -> str:
    return hashlib.sha256(_canonical_jwk(jwk)).hexdigest()


class DemoAccessStore:
    def __init__(self) -> None:
        raw_tokens = os.environ.get("ALPHAMOTION_ACCESS_TOKENS", "")
        tokens = [item.strip() for item in raw_tokens.replace("\n", ",").split(",")
                  if item.strip()]
        legacy_token = os.environ.get("ALPHAMOTION_ACCESS_TOKEN", "").strip()
        if legacy_token and legacy_token not in tokens:
            tokens.insert(0, legacy_token)
        self.tokens = tuple(tokens)
        self.shared_token = self.tokens[0] if self.tokens else ""
        self.enabled = bool(self.tokens)
        self.max_devices = max(1, int(os.environ.get(
            "ALPHAMOTION_ACCESS_MAX_DEVICES", "3")))
        self.session_days = max(1, int(os.environ.get(
            "ALPHAMOTION_ACCESS_SESSION_DAYS", "30")))
        self.secure_cookie = os.environ.get(
            "ALPHAMOTION_COOKIE_SECURE", "1") not in ("0", "false", "False")
        self.path = Path(os.environ.get(
            "ALPHAMOTION_ACCESS_REGISTRY", "") or
            data_dir() / "demo_access.json")
        self._key = hashlib.sha256(
            b"alphamotion-demo-session\0" + b"\0".join(
                token.encode() for token in self.tokens)).digest()
        self._lock = threading.RLock()
        self._challenges: dict[str, dict] = {}

    def _read(self) -> dict:
        if not self.path.is_file():
            return {"version": 1, "devices": {}}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and isinstance(value.get("devices"), dict):
                return value
        except (OSError, ValueError, TypeError):
            pass
        return {"version": 1, "devices": {}}

    def _write(self, value: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(value, indent=2, sort_keys=True),
                       encoding="utf-8")
        tmp.replace(self.path)

    def _token_id(self, value: object) -> str | None:
        if not isinstance(value, str):
            return None
        for token in self.tokens:
            if hmac.compare_digest(value.encode(), token.encode()):
                return hashlib.sha256(token.encode()).hexdigest()[:16]
        return None

    def challenge(self, jwk: object) -> dict:
        fingerprint = _fingerprint(jwk)
        challenge_id = secrets.token_urlsafe(24)
        nonce = secrets.token_bytes(32)
        now = time.time()
        with self._lock:
            self._challenges = {
                key: value for key, value in self._challenges.items()
                if float(value["expires"]) > now
            }
            self._challenges[challenge_id] = {
                "fingerprint": fingerprint, "nonce": nonce,
                "expires": now + 120,
            }
        return {"challenge_id": challenge_id, "nonce": _b64e(nonce)}

    def _verify_signature(self, jwk: dict, challenge_id: str,
                          signature: str) -> str:
        fingerprint = _fingerprint(jwk)
        with self._lock:
            challenge = self._challenges.pop(challenge_id, None)
        if not challenge or challenge["expires"] < time.time():
            raise ValueError("device challenge expired; try again")
        if challenge["fingerprint"] != fingerprint:
            raise ValueError("device key changed during sign-in")
        raw = _b64d(signature)
        if len(raw) == 64:  # WebCrypto emits IEEE-P1363 r||s.
            raw = encode_dss_signature(
                int.from_bytes(raw[:32], "big"),
                int.from_bytes(raw[32:], "big"))
        x = int.from_bytes(_b64d(jwk["x"]), "big")
        y = int.from_bytes(_b64d(jwk["y"]), "big")
        public = ec.EllipticCurvePublicNumbers(x, y, ec.SECP256R1()).public_key()
        try:
            public.verify(raw, challenge["nonce"], ec.ECDSA(hashes.SHA256()))
        except Exception as exc:  # cryptography exposes several backend errors
            raise ValueError("device signature is invalid") from exc
        return fingerprint

    def register(self, body: dict) -> tuple[str, dict]:
        token_id = self._token_id(body.get("token"))
        if token_id is None:
            raise PermissionError("access token is invalid")
        jwk = body.get("public_key")
        fingerprint = self._verify_signature(
            jwk, str(body.get("challenge_id") or ""),
            str(body.get("signature") or ""))
        name = str(body.get("device_name") or "Browser").strip()[:80] or "Browser"
        now = int(time.time())
        with self._lock:
            registry = self._read()
            devices = registry["devices"]
            device_key = f"{token_id}:{fingerprint}"
            existing = devices.get(device_key)
            active = [item for item in devices.values()
                      if item.get("token_id") == token_id and
                      not item.get("revoked")]
            if existing is None and len(active) >= self.max_devices:
                raise OverflowError(
                    f"this token already has {self.max_devices} registered devices")
            devices[device_key] = {
                "id": fingerprint[:12], "name": name,
                "token_id": token_id,
                "created_at": (existing or {}).get("created_at", now),
                "last_seen_at": now, "revoked": False,
            }
            self._write(registry)
        return self.make_session(token_id, fingerprint), devices[device_key]

    def make_session(self, token_id: str, fingerprint: str) -> str:
        payload = _b64e(json.dumps({
            "device": fingerprint, "token_id": token_id,
            "expires": int(time.time()) + self.session_days * 86400,
        }, separators=(",", ":")).encode())
        signature = _b64e(hmac.new(
            self._key, payload.encode(), hashlib.sha256).digest())
        return payload + "." + signature

    def verify_session(self, value: str | None) -> bool:
        if not self.enabled:
            return True
        try:
            payload, signature = (value or "").split(".", 1)
            expected = _b64e(hmac.new(
                self._key, payload.encode(), hashlib.sha256).digest())
            if not hmac.compare_digest(signature, expected):
                return False
            decoded = json.loads(_b64d(payload))
            if int(decoded["expires"]) < int(time.time()):
                return False
            fingerprint = decoded["device"]
            token_id = decoded["token_id"]
            with self._lock:
                device = self._read()["devices"].get(
                    f"{token_id}:{fingerprint}")
            return bool(device and not device.get("revoked"))
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            return False

    def devices(self, token: object) -> list[dict]:
        token_id = self._token_id(token)
        if token_id is None:
            raise PermissionError("access token is invalid")
        with self._lock:
            values = [item for item in self._read()["devices"].values()
                      if item.get("token_id") == token_id]
        return sorted(values, key=lambda item: (
            int(item.get("created_at", 0)), str(item.get("name", "")),
            str(item.get("id", ""))))

    def revoke(self, token: object, device_id: object) -> None:
        token_id = self._token_id(token)
        if token_id is None:
            raise PermissionError("access token is invalid")
        wanted = str(device_id or "")
        with self._lock:
            registry = self._read()
            matched = [key for key, item in registry["devices"].items()
                       if item.get("token_id") == token_id and
                       item.get("id") == wanted]
            if not matched:
                raise KeyError("device not found")
            del registry["devices"][matched[0]]
            self._write(registry)

    def cookie(self, session: str) -> str:
        secure = "; Secure" if self.secure_cookie else ""
        return (f"{COOKIE}={session}; Path=/; Max-Age={self.session_days * 86400}; "
                f"HttpOnly; SameSite=Lax{secure}")


def _cookie(scope: dict) -> str | None:
    headers = {key.lower(): value for key, value in scope.get("headers", [])}
    raw = headers.get(b"cookie", b"").decode(errors="ignore")
    parsed = SimpleCookie()
    try:
        parsed.load(raw)
        return parsed[COOKIE].value if COOKIE in parsed else None
    except Exception:
        return None


def _access_html(max_devices: int, next_path: str) -> bytes:
    safe_next = next_path if next_path.startswith("/") and not next_path.startswith("//") else "/"
    template = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AlphaMotion Access</title><style>
:root{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,sans-serif;background:#0d0e10;color:#f3f1ec}
*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;background:radial-gradient(circle at 50% 20%,#29231b 0,#111214 42%,#090a0c 100%)}
.card{width:min(460px,calc(100vw - 32px));padding:34px;border:1px solid #38322a;border-radius:14px;background:rgba(21,22,24,.95);box-shadow:0 24px 80px #0009}
.brand{color:#ff9418;font-weight:750;letter-spacing:.02em}.sub{color:#aaa39a;line-height:1.55;margin:10px 0 26px}
label{display:block;font-size:12px;color:#bbb3a8;margin:14px 0 7px}input{width:100%;padding:13px 14px;border-radius:7px;border:1px solid #47433d;background:#101113;color:#fff;font:inherit;outline:none}input:focus{border-color:#f08c24;box-shadow:0 0 0 3px #f08c2428}
button{width:100%;margin-top:18px;padding:13px;border:0;border-radius:7px;background:#ee881c;color:#111;font-weight:760;cursor:pointer}button:disabled{opacity:.55;cursor:wait}.secondary{margin-top:8px;background:#292a2d;color:#d6d0c8;border:1px solid #45413b}.msg{min-height:22px;color:#f3a55a;font-size:13px;margin-top:14px}.small{font-size:12px;color:#77736d;margin-top:22px}.devices{display:grid;gap:8px;margin-top:14px}.device{display:grid;grid-template-columns:1fr auto;gap:10px;align-items:center;padding:10px 11px;border:1px solid #373532;border-radius:7px;font-size:13px}.device small{display:block;color:#817c75;margin-top:3px}.device button{width:auto;margin:0;padding:7px 10px;background:#3a2926;color:#ffb49d}
</style></head><body><main class="card"><div class="brand">AlphaMotion</div><h1>Studio access</h1><p class="sub">Enter the shared demo token. This token can register up to __MAX__ browser devices.</p>
<form id="form"><label for="device">Device name</label><input id="device" maxlength="80" autocomplete="off"><label for="token">Demo token</label><input id="token" type="password" autocomplete="current-password" required><button id="submit">Open Studio</button><button class="secondary" type="button" id="manage">Manage registered devices</button><div class="msg" id="msg"></div><div class="devices" id="devices"></div></form>
<p class="small">The device key stays in this browser. Clearing browser storage may require the operator to release the old slot.</p></main><script>
const nextPath=__NEXT__, enc=new TextEncoder();
const b64u=b=>{let s='';new Uint8Array(b).forEach(x=>s+=String.fromCharCode(x));return btoa(s).replaceAll('+','-').replaceAll('/','_').replaceAll('=','')};
const db=()=>new Promise((ok,no)=>{const r=indexedDB.open('alphamotion-demo-access',1);r.onupgradeneeded=()=>r.result.createObjectStore('keys');r.onsuccess=()=>ok(r.result);r.onerror=()=>no(r.error)});
async function getKeys(){const d=await db();let keys=await new Promise((ok,no)=>{const r=d.transaction('keys').objectStore('keys').get('device');r.onsuccess=()=>ok(r.result);r.onerror=()=>no(r.error)});if(!keys){keys=await crypto.subtle.generateKey({name:'ECDSA',namedCurve:'P-256'},false,['sign','verify']);await new Promise((ok,no)=>{const r=d.transaction('keys','readwrite').objectStore('keys').put(keys,'device');r.onsuccess=ok;r.onerror=()=>no(r.error)})}return keys}
document.querySelector('#device').value=(navigator.platform||'Browser')+' · '+(navigator.userAgent.includes('Chrome')?'Chrome':'Browser');
async function loadDevices(){const msg=document.querySelector('#msg'),box=document.querySelector('#devices'),token=document.querySelector('#token').value;if(!token){msg.textContent='Enter the demo token first.';return}msg.textContent='Loading registered devices…';let r=await fetch('/api/access/devices',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({token})}),out=await r.json();if(!r.ok){msg.textContent=out.detail||'Could not load devices';return}msg.textContent=`${out.devices.length} of ${out.max_devices} device slots are in use.`;box.replaceChildren(...out.devices.map(d=>{const row=document.createElement('div');row.className='device';const label=document.createElement('div');label.textContent=d.name;const meta=document.createElement('small');meta.textContent='Device '+d.id;label.append(meta);const remove=document.createElement('button');remove.type='button';remove.textContent='Release';remove.onclick=async()=>{if(!confirm(`Release ${d.name}? That browser will be signed out.`))return;const res=await fetch('/api/access/revoke',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({token,device_id:d.id})});const value=await res.json();if(!res.ok){msg.textContent=value.detail||'Could not release device';return}await loadDevices()};row.append(label,remove);return row}))}
document.querySelector('#manage').addEventListener('click',loadDevices);
document.querySelector('#form').addEventListener('submit',async e=>{e.preventDefault();const btn=document.querySelector('#submit'),msg=document.querySelector('#msg');btn.disabled=true;msg.textContent='Verifying this device…';try{const keys=await getKeys(),pub=await crypto.subtle.exportKey('jwk',keys.publicKey);let r=await fetch('/api/access/challenge',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({public_key:pub})});let out=await r.json();if(!r.ok)throw Error(out.detail||'Challenge failed');const nonce=Uint8Array.from(atob(out.nonce.replaceAll('-','+').replaceAll('_','/')),c=>c.charCodeAt(0));const sig=await crypto.subtle.sign({name:'ECDSA',hash:'SHA-256'},keys.privateKey,nonce);r=await fetch('/api/access/register',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({token:document.querySelector('#token').value,device_name:document.querySelector('#device').value,public_key:pub,challenge_id:out.challenge_id,signature:b64u(sig)})});out=await r.json();if(!r.ok)throw Error(out.detail||'Access denied');location.replace(nextPath)}catch(err){msg.textContent=err.message||String(err);btn.disabled=false}});
</script></body></html>'''
    return template.replace("__MAX__", str(max_devices)).replace(
        "__NEXT__", json.dumps(safe_next)).encode()


class DemoAccessMiddleware:
    """Pure ASGI middleware so HTTP, mounted static files and WS share a gate."""

    def __init__(self, app) -> None:
        self.app = app
        self.store = DemoAccessStore()

    async def _json_body(self, receive) -> dict:
        chunks = []
        more = True
        while more:
            message = await receive()
            if message["type"] == "http.disconnect":
                break
            if message["type"] != "http.request":
                break
            chunks.append(message.get("body", b""))
            more = bool(message.get("more_body"))
        try:
            value = json.loads(b"".join(chunks) or b"{}")
            return value if isinstance(value, dict) else {}
        except (ValueError, TypeError):
            return {}

    async def _response(self, send, status: int, body: bytes, content_type: str,
                        headers: list[tuple[bytes, bytes]] | None = None) -> None:
        base = [(b"content-type", content_type.encode()),
                (b"content-length", str(len(body)).encode()),
                (b"cache-control", b"no-store")]
        await send({"type": "http.response.start", "status": status,
                    "headers": base + (headers or [])})
        await send({"type": "http.response.body", "body": body})

    async def _json(self, send, status: int, value: dict,
                    headers: list[tuple[bytes, bytes]] | None = None) -> None:
        await self._response(send, status,
                             json.dumps(value).encode(),
                             "application/json", headers)

    async def __call__(self, scope, receive, send) -> None:
        if not self.store.enabled:
            await self.app(scope, receive, send)
            return
        kind, path = scope["type"], scope.get("path", "")
        if kind == "http" and path == "/access":
            query = parse_qs(scope.get("query_string", b"").decode())
            next_path = (query.get("next") or ["/"])[0]
            await self._response(send, 200,
                                 _access_html(self.store.max_devices, next_path),
                                 "text/html; charset=utf-8")
            return
        if kind == "http" and path.startswith("/api/access/"):
            body = await self._json_body(receive)
            try:
                if path == "/api/access/challenge":
                    await self._json(send, 200,
                                     self.store.challenge(body.get("public_key")))
                elif path == "/api/access/register":
                    session, device = self.store.register(body)
                    await self._json(
                        send, 200, {"ok": True, "device": device},
                        [(b"set-cookie", self.store.cookie(session).encode())])
                elif path == "/api/access/devices":
                    await self._json(send, 200, {
                        "devices": self.store.devices(body.get("token")),
                        "max_devices": self.store.max_devices})
                elif path == "/api/access/revoke":
                    self.store.revoke(body.get("token"), body.get("device_id"))
                    await self._json(send, 200, {"ok": True})
                elif path == "/api/access/logout":
                    await self._json(send, 200, {"ok": True}, [
                        (b"set-cookie",
                         f"{COOKIE}=; Path=/; Max-Age=0; HttpOnly; SameSite=Lax".encode())])
                else:
                    await self._json(send, 404, {"detail": "not found"})
            except PermissionError as exc:
                await self._json(send, 401, {"detail": str(exc)})
            except OverflowError as exc:
                await self._json(send, 409, {"detail": str(exc)})
            except KeyError as exc:
                await self._json(send, 404, {"detail": str(exc).strip("'")})
            except (ValueError, TypeError) as exc:
                await self._json(send, 400, {"detail": str(exc)})
            return
        authorized = self.store.verify_session(_cookie(scope))
        if authorized:
            await self.app(scope, receive, send)
            return
        if kind == "websocket":
            await send({"type": "websocket.close", "code": 4401,
                        "reason": "AlphaMotion access required"})
            return
        if kind == "http":
            method = scope.get("method", "GET")
            accepts_html = method == "GET" and not path.startswith("/api/")
            if accepts_html:
                destination = path
                if scope.get("query_string"):
                    destination += "?" + scope["query_string"].decode()
                location = "/access?next=" + quote(destination, safe="")
                await self._response(send, 307, b"", "text/plain", [
                    (b"location", location.encode())])
            else:
                await self._json(send, 401, {
                    "detail": "AlphaMotion demo access required"})
            return
        await self.app(scope, receive, send)
