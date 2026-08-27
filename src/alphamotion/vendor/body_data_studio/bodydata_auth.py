from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import parse_qs


LOGIN_PAGE = b"""<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>Body Data Studio</title><style>
:root{color-scheme:light;--orange:#ff7417;--ink:#111827;--muted:#667085;--line:#d9dde4;--panel:#fff;--bg:#f5f6f7}
*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;background:var(--bg);color:var(--ink);font:15px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}
.card{width:min(430px,calc(100vw - 32px));padding:34px;border:1px solid var(--line);border-radius:22px;background:var(--panel);box-shadow:0 20px 70px #0002}
.mark{display:grid;place-items:center;width:52px;height:52px;overflow:hidden;border:1px solid var(--line);border-radius:12px;background:#fff;margin-bottom:22px}.mark img{display:block;width:70px;height:70px;max-width:none;transform:translate(-.8%,4.4%)}
h1{font-size:27px;margin:0 0 7px}.sub{color:var(--muted);margin:0 0 25px}label{display:block;font-weight:700;margin-bottom:8px}
input,button{width:100%;height:48px;border-radius:11px;font:inherit}input{border:1px solid var(--line);padding:0 14px;background:transparent;color:var(--ink);outline:none}input:focus{border-color:var(--orange);box-shadow:0 0 0 3px #ff741724}
button{margin-top:13px;border:0;background:var(--orange);color:#111;font-weight:850;cursor:pointer}button:disabled{opacity:.55;cursor:wait}.error{min-height:22px;margin-top:12px;color:#d92d20;font-size:13px}.note{margin-top:22px;color:var(--muted);font-size:12px}
</style></head><body><main class=\"card\"><div class=\"mark\" aria-label=\"Body Data Studio\"><img src=\"/assets/brand/body-data-mark.png?v=ea28e43\" alt=\"\"></div><h1>Body Data Studio</h1><p class=\"sub\">Private motion-data review workspace</p>
<form id=\"login\"><label for=\"key\">Access token</label><input id=\"key\" name=\"access_key\" type=\"password\" autocomplete=\"current-password\" autofocus required><button>Continue</button><div class=\"error\" id=\"error\"></div></form>
<p class=\"note\">Each access token can activate up to three browser devices.</p></main><script>
const form=document.querySelector('#login'),button=form.querySelector('button'),error=document.querySelector('#error');
form.addEventListener('submit',async event=>{event.preventDefault();button.disabled=true;error.textContent='';try{const response=await fetch('/api/auth/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({access_key:form.access_key.value})});const data=await response.json();if(!response.ok)throw new Error(data.error||'Access denied');location.replace('/');}catch(exc){error.textContent=exc.message;}finally{button.disabled=false;}});
</script></body></html>"""


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


class AccessGate:
    """Small, file-backed access-token gate intended to sit behind a TLS proxy."""

    def __init__(self, key_file: Path | None, binding_file: Path | None, secret: bytes | None):
        self.key_file = key_file
        self.binding_file = binding_file
        self.secret = secret
        self.enabled = bool(key_file)
        self._lock = threading.Lock()
        self._attempts: dict[str, list[float]] = {}
        self._keys: list[dict] = []
        if not self.enabled:
            return
        if not secret or len(secret) < 32:
            raise RuntimeError("BODY_DATA_AUTH_SECRET must contain at least 32 bytes")
        payload = json.loads(key_file.read_text(encoding="utf-8"))
        keys = payload.get("keys", [])
        if not keys:
            raise RuntimeError("The Body Data Studio access-key file is empty")
        for item in keys:
            identity = str(item.get("id", "")).strip()
            digest = str(item.get("sha256", "")).strip().lower()
            if not identity or len(digest) != 64:
                raise RuntimeError("Every access key needs an id and SHA-256 digest")
            self._keys.append({"id": identity, "sha256": digest})
        if self.binding_file:
            self.binding_file.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_environment(cls) -> "AccessGate":
        raw_key_file = os.environ.get("BODY_DATA_ACCESS_KEYS_FILE", "").strip()
        if not raw_key_file:
            return cls(None, None, None)
        key_file = Path(raw_key_file).expanduser().resolve()
        binding_file = Path(
            os.environ.get("BODY_DATA_ACCESS_BINDINGS_FILE", str(key_file.with_name("access_bindings.json")))
        ).expanduser().resolve()
        raw_secret = os.environ.get("BODY_DATA_AUTH_SECRET", "")
        secret_file = os.environ.get("BODY_DATA_AUTH_SECRET_FILE", "").strip()
        if secret_file:
            raw_secret = Path(secret_file).expanduser().read_text(encoding="utf-8").strip()
        return cls(key_file, binding_file, raw_secret.encode("utf-8"))

    def _load_bindings(self) -> dict[str, dict]:
        if not self.binding_file or not self.binding_file.exists():
            return {}
        try:
            value = json.loads(self.binding_file.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_bindings(self, bindings: dict[str, dict]) -> None:
        assert self.binding_file is not None
        temporary = self.binding_file.with_suffix(self.binding_file.suffix + ".tmp")
        temporary.write_text(json.dumps(bindings, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.binding_file)

    def _limited(self, fingerprint: str) -> bool:
        now = time.time()
        recent = [stamp for stamp in self._attempts.get(fingerprint, []) if now - stamp < 300]
        self._attempts[fingerprint] = recent
        return len(recent) >= 8

    def _record_failure(self, fingerprint: str) -> None:
        self._attempts.setdefault(fingerprint, []).append(time.time())

    def login(self, access_key: str, device_id: str = "") -> tuple[bool, str, str, str]:
        fingerprint = hashlib.sha256((device_id or "new-device").encode("utf-8")).hexdigest()
        if self._limited(fingerprint):
            return False, "Too many attempts. Try again later.", "", ""
        supplied = hashlib.sha256(access_key.encode("utf-8")).hexdigest()
        matched = next((item for item in self._keys if hmac.compare_digest(item["sha256"], supplied)), None)
        if not matched:
            self._record_failure(fingerprint)
            return False, "Invalid access token.", "", ""
        identity = matched["id"]
        device_id = device_id if len(device_id) >= 24 else "dev_" + secrets.token_urlsafe(24)
        device_hash = hashlib.sha256(device_id.encode("utf-8")).hexdigest()
        with self._lock:
            bindings = self._load_bindings()
            existing = bindings.get(identity, {"devices": []})
            devices = existing.get("devices", []) if isinstance(existing, dict) else []
            matched_device = next((item for item in devices if hmac.compare_digest(str(item.get("hash", "")), device_hash)), None)
            if not matched_device and len(devices) >= 3:
                self._record_failure(fingerprint)
                return False, "This access token has already activated three devices.", "", ""
            now = int(time.time())
            if matched_device:
                matched_device["last_seen"] = now
            else:
                devices.append({"hash": device_hash, "activated_at": now, "last_seen": now})
            bindings[identity] = {"devices": devices}
            self._write_bindings(bindings)
        self._attempts.pop(fingerprint, None)
        return True, "", self.make_token(identity, device_id), device_id

    def make_token(self, identity: str, device_id: str, lifetime: int = 60 * 60 * 24 * 21) -> str:
        payload = _b64(json.dumps({"id": identity, "device": device_id, "exp": int(time.time()) + lifetime}, separators=(",", ":")).encode("utf-8"))
        signature = _b64(hmac.new(self.secret or b"", payload.encode("ascii"), hashlib.sha256).digest())
        return f"{payload}.{signature}"

    def token_identity(self, token: str, device_id: str) -> str:
        try:
            payload, signature = token.split(".", 1)
            expected = _b64(hmac.new(self.secret or b"", payload.encode("ascii"), hashlib.sha256).digest())
            if not hmac.compare_digest(signature, expected):
                return ""
            value = json.loads(_unb64(payload))
            if int(value.get("exp", 0)) < time.time() or not hmac.compare_digest(str(value.get("device", "")), device_id):
                return ""
            identity = str(value.get("id", ""))
            device_hash = hashlib.sha256(device_id.encode("utf-8")).hexdigest()
            devices = self._load_bindings().get(identity, {}).get("devices", [])
            if not any(hmac.compare_digest(str(item.get("hash", "")), device_hash) for item in devices):
                return ""
            return identity
        except (ValueError, TypeError, json.JSONDecodeError):
            return ""

    @staticmethod
    def cookies(handler) -> dict[str, str]:
        cookies = {}
        for part in handler.headers.get("Cookie", "").split(";"):
            name, separator, value = part.strip().partition("=")
            if separator:
                cookies[name] = value
        return cookies

    def identity(self, handler) -> str:
        if not self.enabled:
            return "local"
        cookies = self.cookies(handler)
        return self.token_identity(cookies.get("bds_session", ""), cookies.get("bds_device", ""))

    def cookie_header(self, token: str) -> str:
        secure = "; Secure" if os.environ.get("BODY_DATA_SECURE_COOKIE", "") == "1" else ""
        return f"bds_session={token}; Path=/; Max-Age=1814400; HttpOnly; SameSite=Strict{secure}"

    def device_cookie_header(self, device_id: str) -> str:
        secure = "; Secure" if os.environ.get("BODY_DATA_SECURE_COOKIE", "") == "1" else ""
        return f"bds_device={device_id}; Path=/; Max-Age=31536000; HttpOnly; SameSite=Strict{secure}"

    @staticmethod
    def expired_cookie_header() -> str:
        return "bds_session=; Path=/; Max-Age=0; HttpOnly; SameSite=Strict"

    @staticmethod
    def parse_form(body: bytes, content_type: str) -> dict:
        if content_type.split(";", 1)[0].strip() == "application/json":
            value = json.loads(body or b"{}")
            return value if isinstance(value, dict) else {}
        return {key: values[-1] for key, values in parse_qs(body.decode("utf-8")).items()}


def generate_key_config(count: int = 8) -> tuple[dict, list[tuple[str, str]]]:
    """Deployment helper: returns a hash-only config and one-time plaintext keys."""
    plain: list[tuple[str, str]] = []
    config = {"keys": []}
    for index in range(1, count + 1):
        identity = f"vc-{index:02d}"
        value = "bds_" + secrets.token_urlsafe(18)
        plain.append((identity, value))
        config["keys"].append({"id": identity, "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest()})
    return config, plain
