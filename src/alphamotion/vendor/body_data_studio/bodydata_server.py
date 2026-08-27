from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, unquote, urlparse
import webbrowser

from bodydata_decode import PREVIEW_CACHE_VERSION, PreviewError, prepare
from bodydata_index import CACHE_ROOT, DATA_ROOT, DB_PATH, build_index, connect
from bodydata_auth import AccessGate, LOGIN_PAGE


APP_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = APP_ROOT / "bodydata_web"
NODE_MODULES = APP_ROOT / "node_modules"
MEDIA: dict[str, Path] = {}
MEDIA_LOCK = threading.Lock()
MEDIA_INDEX_PATH = CACHE_ROOT / "media_tokens.jsonl"
INDEX_THREAD: threading.Thread | None = None
PREVIEW_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="bodydata-preview")
PREVIEW_JOBS: dict[str, Future] = {}
PREVIEW_JOBS_LOCK = threading.Lock()
BATCH_THREAD: threading.Thread | None = None
BATCH_STOP = threading.Event()
BATCH_LOCK = threading.Lock()
BATCH_STATE = {"running": False, "scope": "", "total": 0, "completed": 0, "skipped": 0, "failed": 0, "current": ""}
CLUSTER_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bodydata-cluster")
CLUSTER_JOBS: dict[str, Future] = {}
CLUSTER_PROGRESS: dict[str, dict] = {}
CLUSTER_JOBS_LOCK = threading.Lock()
EXPORT_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bodydata-export")
EXPORT_JOBS: dict[str, Future] = {}
EXPORT_JOBS_LOCK = threading.Lock()
AUGMENT_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bodydata-augment")
AUGMENT_JOBS: dict[str, Future] = {}
AUGMENT_JOBS_LOCK = threading.Lock()
CONTACT_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bodydata-contact")
CONTACT_JOBS: dict[str, Future] = {}
CONTACT_JOBS_LOCK = threading.Lock()
PROCESS_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bodydata-process")
PROCESS_JOBS: dict[str, Future] = {}
PROCESS_JOBS_LOCK = threading.Lock()
ACCESS_GATE = AccessGate.from_environment()

# Draft augmentations need asset records so the comparison viewer can decode
# them, but they are not library assets until the user explicitly saves them.
LIBRARY_VISIBLE_SQL = """(
    source NOT LIKE 'Augmentation Cache / %'
    AND (source NOT LIKE 'Augmented / %' OR metadata_json LIKE '%\"augmentation_saved\": true%')
)"""


def json_value(value: str | None, default):
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def public_asset(row: sqlite3.Row | dict) -> dict:
    item = dict(row)
    item["animated"] = bool(item["animated"])
    item["favorite"] = bool(item.pop("favorite", False))
    item["aux"] = json_value(item.pop("aux_json", "{}"), {})
    item["metadata"] = json_value(item.pop("metadata_json", "{}"), {})
    item.pop("container", None)
    item.pop("inner_path", None)
    item.pop("locator_type", None)
    return item


def library_motion_totals(db: sqlite3.Connection) -> dict:
    rows = db.execute(
        f"SELECT metadata_json FROM assets WHERE {LIBRARY_VISIBLE_SQL} AND status='ready' AND kind='smplh_motion'"
    ).fetchall()
    total_frames = 0
    total_seconds = 0.0
    measured = 0
    for row in rows:
        metadata = json_value(row["metadata_json"], {})
        frames = int(metadata.get("frame_count") or 0)
        duration = float(metadata.get("duration_seconds") or 0.0)
        if frames <= 0 or duration < 0:
            continue
        total_frames += frames
        total_seconds += duration
        measured += 1
    return {
        "sequence_assets": len(rows),
        "measured_assets": measured,
        "frames": total_frames,
        "duration_seconds": total_seconds,
        "complete": measured == len(rows),
    }


def media_url(path: str | Path, directory: bool = False) -> str:
    path = Path(path).resolve()
    allowed = (DATA_ROOT.resolve(), CACHE_ROOT.resolve())
    if not any(path == root or root in path.parents for root in allowed):
        raise PreviewError("Preview attempted to expose a path outside the configured data/cache roots.")
    identity = hashlib.sha256(str(path).lower().encode("utf-8")).hexdigest()[:32]
    with MEDIA_LOCK:
        if MEDIA.get(identity) != path:
            MEDIA[identity] = path
            MEDIA_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
            with MEDIA_INDEX_PATH.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"token": identity, "path": str(path)}, ensure_ascii=False) + "\n")
    if directory:
        return f"/media/{identity}/"
    return f"/media/{identity}/{quote(path.name)}"


def resolve_media_token(token: str) -> Path | None:
    """Recover a preview token after a server restart without widening file access."""
    with MEDIA_LOCK:
        cached = MEDIA.get(token)
        if cached is not None:
            return cached
        if not MEDIA_INDEX_PATH.is_file():
            return None
        lines = MEDIA_INDEX_PATH.read_text(encoding="utf-8").splitlines()
        allowed = (DATA_ROOT.resolve(), CACHE_ROOT.resolve())
        for line in reversed(lines):
            try:
                record = json.loads(line)
                if record.get("token") != token:
                    continue
                candidate = Path(record["path"]).resolve()
            except (json.JSONDecodeError, KeyError, TypeError, OSError):
                continue
            expected = hashlib.sha256(str(candidate).lower().encode("utf-8")).hexdigest()[:32]
            if expected != token or not any(candidate == root or root in candidate.parents for root in allowed):
                return None
            MEDIA[token] = candidate
            return candidate
    return None


def expose_preview(result: dict) -> dict:
    output: dict = {}
    for key, value in result.items():
        if key == "resource_root_path" and value:
            output["resource_root_url"] = media_url(value, directory=True)
        elif key.endswith("_path") and value:
            output[key.removesuffix("_path") + "_url"] = media_url(value)
        else:
            output[key] = value
    return output


def start_index(full: bool) -> bool:
    global INDEX_THREAD
    if INDEX_THREAD and INDEX_THREAD.is_alive():
        return False

    def run():
        try:
            build_index(full=full)
        except Exception as exc:
            db = connect()
            with db:
                db.execute(
                    "INSERT INTO state(key,value) VALUES('index_error',?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (json.dumps({"time": time.time(), "error": f"{type(exc).__name__}: {exc}"}),),
                )
            db.close()

    INDEX_THREAD = threading.Thread(target=run, name="bodydata-index", daemon=True)
    INDEX_THREAD.start()
    return True


def source_fingerprint(asset: dict) -> str:
    source = Path(asset["container"])
    try:
        stat = source.stat()
        value = f"{source.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{asset.get('inner_path', '')}"
    except OSError:
        value = f"{source}|missing|{asset.get('inner_path', '')}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def cached_preview(asset: dict) -> tuple[str, dict | None, str]:
    fingerprint = source_fingerprint(asset)
    db = connect()
    row = db.execute("SELECT * FROM preview_artifacts WHERE asset_id=?", (asset["id"],)).fetchone()
    db.close()
    if not row or row["source_fingerprint"] != fingerprint or row["decoder_version"] != PREVIEW_CACHE_VERSION:
        return "missing", None, ""
    if row["status"] == "ready":
        result = json_value(row["result_json"], {})
        paths = [Path(v) for k, v in result.items() if k.endswith("_path") and v]
        if paths and all(p.exists() for p in paths):
            return "ready", result, ""
        return "missing", None, ""
    return row["status"], None, row["error"]


def build_preview(asset: dict) -> dict:
    fingerprint = source_fingerprint(asset)
    now = time.time()
    db = connect()
    with db:
        db.execute(
            """INSERT INTO preview_artifacts(asset_id,status,result_json,error,source_fingerprint,decoder_version,updated_at,last_accessed)
               VALUES(?, 'building', '{}', '', ?, ?, ?, ?)
               ON CONFLICT(asset_id) DO UPDATE SET status='building',error='',source_fingerprint=excluded.source_fingerprint,
                 decoder_version=excluded.decoder_version,updated_at=excluded.updated_at,last_accessed=excluded.last_accessed""",
            (asset["id"], fingerprint, PREVIEW_CACHE_VERSION, now, now),
        )
    db.close()
    try:
        result = prepare(asset)
        db = connect()
        with db:
            db.execute(
                "UPDATE preview_artifacts SET status='ready',result_json=?,error='',updated_at=?,last_accessed=? WHERE asset_id=?",
                (json.dumps(result, ensure_ascii=False), time.time(), time.time(), asset["id"]),
            )
        db.close()
        return result
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        db = connect()
        with db:
            db.execute(
                "UPDATE preview_artifacts SET status='failed',error=?,updated_at=? WHERE asset_id=?",
                (message, time.time(), asset["id"]),
            )
        db.close()
        raise


def enqueue_preview(asset: dict) -> str:
    status, _, _ = cached_preview(asset)
    if status == "ready":
        return "ready"
    with PREVIEW_JOBS_LOCK:
        current = PREVIEW_JOBS.get(asset["id"])
        if current is None or current.done():
            PREVIEW_JOBS[asset["id"]] = PREVIEW_EXECUTOR.submit(build_preview, asset)
    return "building"


def start_prebuild(source: str = "", collection: str = "") -> bool:
    global BATCH_THREAD, BATCH_STATE
    with BATCH_LOCK:
        if BATCH_THREAD and BATCH_THREAD.is_alive():
            return False
        BATCH_STOP.clear()
        scope = source or (f"collection:{collection}" if collection else "all")
        BATCH_STATE = {"running": True, "scope": scope, "total": 0, "completed": 0, "skipped": 0, "failed": 0, "current": ""}

    def run():
        global BATCH_STATE
        db = connect()
        if source:
            rows = db.execute("SELECT * FROM assets WHERE status='ready' AND source=? ORDER BY id", (source,)).fetchall()
        elif collection:
            rows = db.execute(
                "SELECT a.* FROM assets a JOIN collection_assets ca ON ca.asset_id=a.id WHERE a.status='ready' AND ca.collection_id=? ORDER BY a.id",
                (collection,),
            ).fetchall()
        else:
            rows = db.execute("SELECT * FROM assets WHERE status='ready' ORDER BY id").fetchall()
        db.close()
        with BATCH_LOCK:
            BATCH_STATE["total"] = len(rows)
        for row in rows:
            if BATCH_STOP.is_set():
                break
            asset = dict(row)
            with BATCH_LOCK:
                BATCH_STATE["current"] = asset["title"]
            status, _, _ = cached_preview(asset)
            if status == "ready":
                with BATCH_LOCK:
                    BATCH_STATE["skipped"] += 1
                    BATCH_STATE["completed"] += 1
                continue
            try:
                build_preview(asset)
            except Exception:
                with BATCH_LOCK:
                    BATCH_STATE["failed"] += 1
            finally:
                with BATCH_LOCK:
                    BATCH_STATE["completed"] += 1
        with BATCH_LOCK:
            BATCH_STATE["running"] = False
            BATCH_STATE["current"] = ""

    BATCH_THREAD = threading.Thread(target=run, name="bodydata-prebuild", daemon=True)
    BATCH_THREAD.start()
    return True


class Handler(BaseHTTPRequestHandler):
    server_version = "BodyDataStudio/1.0"

    def log_message(self, fmt, *args):
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def send_json(self, payload, status=HTTPStatus.OK):
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_login_page(self):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(LOGIN_PAGE)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", "default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(LOGIN_PAGE)

    def send_auth_status(self):
        identity = ACCESS_GATE.identity(self)
        self.send_json({"ok": True, "authenticated": bool(identity), "identity": identity})

    def handle_login(self):
        length = min(int(self.headers.get("Content-Length", "0") or 0), 16_384)
        payload = ACCESS_GATE.parse_form(self.rfile.read(length), self.headers.get("Content-Type", ""))
        device_id = ACCESS_GATE.cookies(self).get("bds_device", "")
        ok, error, token, device_id = ACCESS_GATE.login(str(payload.get("access_key", "")), device_id)
        if not ok:
            return self.send_error_json(HTTPStatus.UNAUTHORIZED, error)
        data = json.dumps({"ok": True}, separators=(",", ":")).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Set-Cookie", ACCESS_GATE.cookie_header(token))
        self.send_header("Set-Cookie", ACCESS_GATE.device_cookie_header(device_id))
        self.end_headers()
        self.wfile.write(data)

    def handle_logout(self):
        data = b'{"ok":true}'
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Set-Cookie", ACCESS_GATE.expired_cookie_header())
        self.end_headers()
        self.wfile.write(data)

    def send_error_json(self, status, message):
        self.send_json({"ok": False, "error": message}, status)

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/auth/status":
                return self.send_auth_status()
            if ACCESS_GATE.enabled and not ACCESS_GATE.identity(self):
                if parsed.path == "/assets/brand/body-data-mark.png":
                    return self.serve_static(parsed.path)
                if parsed.path in {"", "/"}:
                    return self.send_login_page()
                return self.send_error_json(HTTPStatus.UNAUTHORIZED, "Authentication required")
            if parsed.path == "/api/health":
                return self.send_json({
                    "ok": True,
                    "root": "Hosted library" if ACCESS_GATE.enabled else str(DATA_ROOT),
                    "db": "managed" if ACCESS_GATE.enabled else str(DB_PATH),
                    "hosted": ACCESS_GATE.enabled,
                })
            if parsed.path == "/api/status":
                return self.api_status()
            if parsed.path == "/api/sources":
                return self.api_sources()
            if parsed.path == "/api/assets":
                return self.api_assets(parse_qs(parsed.query))
            if parsed.path == "/api/inspect":
                return self.api_inspect(parse_qs(parsed.query))
            if parsed.path == "/api/dataset-health":
                return self.api_dataset_health(parse_qs(parsed.query))
            if parsed.path == "/api/folders":
                return self.api_folders(parse_qs(parsed.query))
            if parsed.path == "/api/preview":
                return self.api_preview(parse_qs(parsed.query))
            if parsed.path == "/api/recipe":
                return self.api_recipe_get(parse_qs(parsed.query))
            if parsed.path == "/api/collections":
                return self.api_collections(parse_qs(parsed.query))
            if parsed.path == "/api/favorites":
                return self.api_favorites_get(parse_qs(parsed.query))
            if parsed.path == "/api/prebuild":
                with BATCH_LOCK:
                    return self.send_json({"ok": True, "batch": dict(BATCH_STATE)})
            if parsed.path == "/api/cluster":
                return self.api_cluster_get(parse_qs(parsed.query))
            if parsed.path == "/api/export":
                return self.api_export_get(parse_qs(parsed.query))
            if parsed.path == "/api/augment":
                return self.api_augment_get(parse_qs(parsed.query))
            if parsed.path == "/api/contact-labels":
                return self.api_contact_labels_get(parse_qs(parsed.query))
            if parsed.path == "/api/process":
                return self.api_process_get(parse_qs(parsed.query))
            if parsed.path == "/api/process-download":
                return self.api_process_download(parse_qs(parsed.query))
            if parsed.path.startswith("/media/"):
                return self.serve_media(parsed.path)
            if parsed.path.startswith("/vendor/"):
                return self.serve_file(NODE_MODULES / unquote(parsed.path.removeprefix("/vendor/")))
            return self.serve_static(parsed.path)
        except BrokenPipeError:
            return
        except Exception as exc:
            self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/auth/login":
                return self.handle_login()
            if parsed.path == "/api/auth/logout":
                return self.handle_logout()
            if ACCESS_GATE.enabled and not ACCESS_GATE.identity(self):
                return self.send_error_json(HTTPStatus.UNAUTHORIZED, "Authentication required")
            length = int(self.headers.get("Content-Length", "0") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
            if parsed.path == "/api/prepare":
                return self.api_prepare(payload)
            if parsed.path == "/api/recipe":
                return self.api_recipe_save(payload)
            if parsed.path == "/api/collections":
                return self.api_collections_change(payload)
            if parsed.path == "/api/favorites":
                return self.api_favorites_change(payload)
            if parsed.path == "/api/prewarm":
                return self.api_prewarm(payload)
            if parsed.path == "/api/prebuild":
                if payload.get("action") == "stop":
                    BATCH_STOP.set()
                    return self.send_json({"ok": True, "stopping": True})
                started = start_prebuild(str(payload.get("source", "")), str(payload.get("collection", "")))
                return self.send_json({"ok": True, "started": started})
            if parsed.path == "/api/cluster":
                return self.api_cluster_start(payload)
            if parsed.path == "/api/cluster-label":
                return self.api_cluster_label(payload)
            if parsed.path == "/api/export":
                return self.api_export_start(payload)
            if parsed.path == "/api/augment":
                return self.api_augment_start(payload)
            if parsed.path == "/api/augmentation-save":
                return self.api_augmentation_save(payload)
            if parsed.path == "/api/contact-labels":
                return self.api_contact_labels_start(payload)
            if parsed.path == "/api/process":
                return self.api_process_change(payload)
            if parsed.path == "/api/rescan":
                full = bool(payload.get("full", False))
                started = start_index(full)
                return self.send_json({"ok": True, "started": started, "full": full})
            self.send_error_json(HTTPStatus.NOT_FOUND, "Unknown API endpoint")
        except PreviewError as exc:
            self.send_error_json(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
        except Exception as exc:
            self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    def api_status(self):
        db = connect()
        states = {r["key"]: json_value(r["value"], r["value"]) for r in db.execute("SELECT key,value FROM state")}
        total = db.execute(f"SELECT COUNT(*) FROM assets WHERE {LIBRARY_VISIBLE_SQL}").fetchone()[0]
        ready = db.execute(f"SELECT COUNT(*) FROM assets WHERE {LIBRARY_VISIBLE_SQL} AND status='ready'").fetchone()[0]
        motion_metrics = library_motion_totals(db)
        db.close()
        states["assets"] = {"total": total, "ready": ready}
        states["motion_metrics"] = motion_metrics
        states["index_thread_alive"] = bool(INDEX_THREAD and INDEX_THREAD.is_alive())
        self.send_json({"ok": True, "state": states})

    def api_sources(self):
        db = connect()
        rows = db.execute(
            """
            SELECT source, COUNT(*) AS total,
                   SUM(CASE WHEN status='ready' THEN 1 ELSE 0 END) AS ready,
                   SUM(CASE WHEN animated=1 AND status='ready' THEN 1 ELSE 0 END) AS animated
            FROM assets WHERE """ + LIBRARY_VISIBLE_SQL + """ GROUP BY source ORDER BY source COLLATE NOCASE
            """
        ).fetchall()
        db.close()
        self.send_json({"ok": True, "sources": [dict(r) for r in rows]})

    def api_assets(self, query):
        identities = [value for raw in (query.get("ids") or [])
                      for value in str(raw).split(",") if value]
        source = (query.get("source") or [""])[0]
        search = (query.get("q") or [""])[0].strip()
        status = (query.get("status") or ["all"])[0]
        collection = (query.get("collection") or [""])[0]
        folder = (query.get("folder") or [""])[0]
        cluster_run = (query.get("cluster_run") or [""])[0]
        cluster_id = (query.get("cluster_id") or [""])[0]
        favorites = str((query.get("favorites") or [""])[0]).lower() in {"1", "true", "yes"}
        motion = str((query.get("motion") or [""])[0]).lower()
        limit = min(max(int((query.get("limit") or [200])[0]), 1), 500)
        offset = max(int((query.get("offset") or [0])[0]), 0)
        clauses, params = [LIBRARY_VISIBLE_SQL], []
        if identities:
            clauses.append(f"id IN ({','.join('?' for _ in identities)})")
            params.extend(identities)
        if source:
            clauses.append("source = ?")
            params.append(source)
        if search:
            clauses.append("(title LIKE ? OR folder LIKE ? OR format LIKE ?)")
            wildcard = f"%{search}%"
            params.extend([wildcard, wildcard, wildcard])
        if status != "all":
            clauses.append("status = ?")
            params.append(status)
        if collection:
            clauses.append("id IN (SELECT asset_id FROM collection_assets WHERE collection_id=?)")
            params.append(collection)
        if favorites:
            clauses.append("id IN (SELECT asset_id FROM asset_favorites)")
        if motion == "static":
            clauses.append("animated = 0")
        elif motion == "animated":
            clauses.append("animated = 1")
        if folder:
            clauses.append("folder = ?")
            params.append(folder)
        if cluster_run and cluster_id != "":
            clauses.append("id IN (SELECT asset_id FROM cluster_members WHERE run_id=? AND cluster_id=?)")
            params.extend((cluster_run, int(cluster_id)))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        db = connect()
        total = db.execute(f"SELECT COUNT(*) FROM assets{where}", params).fetchone()[0]
        rows = db.execute(
            f"SELECT assets.*, EXISTS(SELECT 1 FROM asset_favorites WHERE asset_id=assets.id) AS favorite FROM assets{where} ORDER BY folder COLLATE NOCASE, title COLLATE NOCASE LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()
        db.close()
        self.send_json({"ok": True, "total": total, "offset": offset, "assets": [public_asset(r) for r in rows]})

    def api_favorites_get(self, query):
        identity = str((query.get("id") or [""])[0]).strip()
        db = connect()
        if identity:
            favorite = bool(db.execute("SELECT 1 FROM asset_favorites WHERE asset_id=?", (identity,)).fetchone())
            db.close()
            return self.send_json({"ok": True, "asset_id": identity, "favorite": favorite})
        count = int(db.execute("SELECT COUNT(*) FROM asset_favorites").fetchone()[0])
        db.close()
        self.send_json({"ok": True, "count": count})

    def api_favorites_change(self, payload):
        identity = str(payload.get("asset_id", "")).strip()
        if not identity:
            return self.send_error_json(HTTPStatus.BAD_REQUEST, "asset_id is required")
        db = connect()
        if not db.execute("SELECT 1 FROM assets WHERE id=?", (identity,)).fetchone():
            db.close()
            return self.send_error_json(HTTPStatus.NOT_FOUND, "Asset not found")
        requested = payload.get("favorite")
        current = bool(db.execute("SELECT 1 FROM asset_favorites WHERE asset_id=?", (identity,)).fetchone())
        favorite = (not current) if requested is None else bool(requested)
        with db:
            if favorite:
                db.execute("INSERT OR REPLACE INTO asset_favorites(asset_id,created_at) VALUES(?,?)", (identity, time.time()))
            else:
                db.execute("DELETE FROM asset_favorites WHERE asset_id=?", (identity,))
        db.close()
        self.send_json({"ok": True, "asset_id": identity, "favorite": favorite})

    def api_inspect(self, query):
        from bodydata_inspect import inspect_asset

        identity = str((query.get("id") or [""])[0]).strip()
        db = connect()
        row = db.execute("SELECT * FROM assets WHERE id=?", (identity,)).fetchone()
        recipe_row = db.execute("SELECT recipe_json FROM asset_recipes WHERE asset_id=?", (identity,)).fetchone() if row else None
        folder_recipe = db.execute(
            "SELECT recipe_json FROM folder_recipes WHERE source=? AND folder=?",
            (row["source"], row["folder"]),
        ).fetchone() if row else None
        db.close()
        if not row:
            return self.send_error_json(HTTPStatus.NOT_FOUND, "Asset not found")
        asset = dict(row)
        status, preview, _error = cached_preview(asset)
        if asset.get("kind") == "anymate_record" and status == "ready" and not (preview or {}).get("data_schema"):
            preview = build_preview(asset)
            status = "ready"
        recipe = {
            **(json_value(folder_recipe["recipe_json"], {}) if folder_recipe else {}),
            **(json_value(recipe_row["recipe_json"], {}) if recipe_row else {}),
        }
        result = inspect_asset(
            {**asset, "metadata": json_value(asset.get("metadata_json"), {})},
            preview if status == "ready" else None,
            recipe,
        )
        self.send_json({"ok": True, "inspection": result})

    def api_dataset_health(self, query):
        source = str((query.get("source") or [""])[0]).strip()
        clauses, params = [LIBRARY_VISIBLE_SQL], []
        if source:
            clauses.append("a.source=?")
            params.append(source)
        where = " WHERE " + " AND ".join(clauses)
        db = connect()
        totals = dict(db.execute(
            f"""SELECT COUNT(*) total,
                       SUM(CASE WHEN a.status='ready' THEN 1 ELSE 0 END) ready,
                       SUM(CASE WHEN a.status!='ready' THEN 1 ELSE 0 END) unavailable,
                       SUM(CASE WHEN a.animated=1 THEN 1 ELSE 0 END) animated,
                       SUM(CASE WHEN a.animated=0 THEN 1 ELSE 0 END) static,
                       COALESCE(SUM(a.size),0) bytes
                FROM assets a{where}""",
            params,
        ).fetchone())
        formats = [dict(row) for row in db.execute(
            f"SELECT UPPER(a.format) format,COUNT(*) count,COALESCE(SUM(a.size),0) bytes FROM assets a{where} GROUP BY UPPER(a.format) ORDER BY count DESC",
            params,
        ).fetchall()]
        sources = [dict(row) for row in db.execute(
            f"""SELECT a.source,COUNT(*) total,
                       SUM(CASE WHEN a.status='ready' THEN 1 ELSE 0 END) ready,
                       SUM(CASE WHEN a.animated=1 THEN 1 ELSE 0 END) animated,
                       COALESCE(SUM(a.size),0) bytes
                FROM assets a{where} GROUP BY a.source ORDER BY total DESC""",
            params,
        ).fetchall()]
        preview_where = "WHERE a.source=?" if source else ""
        preview_params = [source] if source else []
        previews = dict(db.execute(
            f"""SELECT COUNT(p.asset_id) cached,
                       SUM(CASE WHEN p.status='ready' THEN 1 ELSE 0 END) ready,
                       SUM(CASE WHEN p.status='failed' THEN 1 ELSE 0 END) failed,
                       SUM(CASE WHEN p.status='building' THEN 1 ELSE 0 END) building
                FROM preview_artifacts p JOIN assets a ON a.id=p.asset_id {preview_where}""",
            preview_params,
        ).fetchone())
        failures = [dict(row) for row in db.execute(
            f"""SELECT a.id,a.title,a.source,a.folder,p.error
                FROM preview_artifacts p JOIN assets a ON a.id=p.asset_id
                WHERE p.status='failed' {"AND a.source=?" if source else ""}
                ORDER BY p.updated_at DESC LIMIT 100""",
            preview_params,
        ).fetchall()]
        db.close()
        self.send_json({"ok": True, "source": source, "totals": totals, "previews": previews, "formats": formats, "sources": sources, "failures": failures})

    def api_folders(self, query):
        source = str((query.get("source") or [""])[0])
        if not source:
            return self.send_error_json(HTTPStatus.BAD_REQUEST, "source is required")
        db = connect()
        rows = db.execute(
            """SELECT folder,COUNT(*) AS total,
                      SUM(CASE WHEN status='ready' THEN 1 ELSE 0 END) AS ready,
                      SUM(CASE WHEN status='ready' AND animated=1 THEN 1 ELSE 0 END) AS animated
               FROM assets WHERE source=? AND """ + LIBRARY_VISIBLE_SQL + """ GROUP BY folder ORDER BY folder COLLATE NOCASE""",
            (source,),
        ).fetchall()
        db.close()
        self.send_json({"ok": True, "source": source, "folders": [dict(row) for row in rows]})

    def api_prepare(self, payload):
        identity = str(payload.get("id", ""))
        db = connect()
        row = db.execute("SELECT * FROM assets WHERE id=?", (identity,)).fetchone()
        db.close()
        if not row:
            return self.send_error_json(HTTPStatus.NOT_FOUND, "Asset not found")
        asset = dict(row)
        if bool(payload.get("async", False)):
            status, result, error = cached_preview(asset)
            if status == "ready" and result:
                return self.send_json({"ok": True, "status": "ready", "asset": public_asset(row), "preview": expose_preview(result)})
            status = enqueue_preview(asset)
            return self.send_json({"ok": True, "status": status, "asset": public_asset(row), "error": error})
        result = expose_preview(build_preview(asset))
        self.send_json({"ok": True, "status": "ready", "asset": public_asset(row), "preview": result})

    def api_preview(self, query):
        identity = str((query.get("id") or [""])[0])
        db = connect()
        row = db.execute("SELECT * FROM assets WHERE id=?", (identity,)).fetchone()
        db.close()
        if not row:
            return self.send_error_json(HTTPStatus.NOT_FOUND, "Asset not found")
        status, result, error = cached_preview(dict(row))
        if status == "ready" and result:
            return self.send_json({"ok": True, "status": "ready", "preview": expose_preview(result)})
        self.send_json({"ok": True, "status": status, "error": error})

    def api_prewarm(self, payload):
        ids = [str(value) for value in payload.get("ids", [])[:12]]
        if not ids:
            return self.send_json({"ok": True, "queued": 0})
        placeholders = ",".join("?" for _ in ids)
        db = connect()
        rows = db.execute(f"SELECT * FROM assets WHERE id IN ({placeholders})", ids).fetchall()
        db.close()
        queued = sum(enqueue_preview(dict(row)) == "building" for row in rows)
        self.send_json({"ok": True, "queued": queued})

    def api_recipe_get(self, query):
        identity = str((query.get("id") or [""])[0])
        db = connect()
        asset = db.execute("SELECT source,folder FROM assets WHERE id=?", (identity,)).fetchone()
        if not asset:
            db.close()
            return self.send_error_json(HTTPStatus.NOT_FOUND, "Asset not found")
        asset_row = db.execute("SELECT recipe_json,updated_at FROM asset_recipes WHERE asset_id=?", (identity,)).fetchone()
        folder_row = db.execute(
            "SELECT recipe_json,updated_at FROM folder_recipes WHERE source=? AND folder=?",
            (asset["source"], asset["folder"]),
        ).fetchone()
        db.close()
        asset_recipe = json_value(asset_row["recipe_json"], {}) if asset_row else {}
        folder_recipe = json_value(folder_row["recipe_json"], {}) if folder_row else {}
        self.send_json({
            "ok": True, "recipe": {**folder_recipe, **asset_recipe}, "asset_recipe": asset_recipe,
            "folder_recipe": folder_recipe, "source": asset["source"], "folder": asset["folder"],
            "asset_updated_at": asset_row["updated_at"] if asset_row else None,
            "folder_updated_at": folder_row["updated_at"] if folder_row else None,
        })

    def api_recipe_save(self, payload):
        identity = str(payload.get("id", ""))
        scope = str(payload.get("scope", "asset"))
        recipe = payload.get("recipe") or {}
        if not isinstance(recipe, dict):
            return self.send_error_json(HTTPStatus.BAD_REQUEST, "recipe must be an object")
        if scope not in {"asset", "folder"}:
            return self.send_error_json(HTTPStatus.BAD_REQUEST, "scope must be asset or folder")
        db = connect()
        asset = db.execute("SELECT source,folder FROM assets WHERE id=?", (identity,)).fetchone()
        if not asset:
            db.close()
            return self.send_error_json(HTTPStatus.NOT_FOUND, "Asset not found")
        with db:
            if scope == "folder":
                if recipe:
                    db.execute(
                        """INSERT INTO folder_recipes(source,folder,recipe_json,updated_at) VALUES(?,?,?,?)
                           ON CONFLICT(source,folder) DO UPDATE SET recipe_json=excluded.recipe_json,updated_at=excluded.updated_at""",
                        (asset["source"], asset["folder"], json.dumps(recipe, ensure_ascii=False), time.time()),
                    )
                else:
                    db.execute("DELETE FROM folder_recipes WHERE source=? AND folder=?", (asset["source"], asset["folder"]))
            elif scope == "asset":
                if recipe:
                    db.execute(
                        """INSERT INTO asset_recipes(asset_id,recipe_json,updated_at) VALUES(?,?,?)
                           ON CONFLICT(asset_id) DO UPDATE SET recipe_json=excluded.recipe_json,updated_at=excluded.updated_at""",
                        (identity, json.dumps(recipe, ensure_ascii=False), time.time()),
                    )
                else:
                    db.execute("DELETE FROM asset_recipes WHERE asset_id=?", (identity,))
        db.close()
        self.send_json({"ok": True, "scope": scope, "recipe": recipe})

    def api_cluster_start(self, payload):
        from bodydata_cluster import cluster_scope

        identity = str(payload.get("id", ""))
        requested_source = str(payload.get("source", ""))
        requested_folder = str(payload.get("folder", ""))
        db = connect()
        asset = db.execute("SELECT source,folder FROM assets WHERE id=?", (identity,)).fetchone() if identity else None
        source = requested_source or (asset["source"] if asset else "")
        folder = requested_folder if requested_source else (asset["folder"] if asset else "")
        if folder:
            count = db.execute(
                "SELECT COUNT(*) FROM assets WHERE source=? AND folder=? AND status='ready' AND animated=1 AND kind='smplh_motion'",
                (source, folder),
            ).fetchone()[0]
        else:
            count = db.execute(
                "SELECT COUNT(*) FROM assets WHERE source=? AND status='ready' AND animated=1 AND kind='smplh_motion'",
                (source,),
            ).fetchone()[0]
        db.close()
        if not source or count == 0:
            return self.send_error_json(HTTPStatus.NOT_FOUND, "No supported motion clips were found in this scope")
        cluster_count = payload.get("clusters")
        if cluster_count is not None:
            cluster_count = min(max(int(cluster_count), 2), 50)
        job_id = uuid.uuid4().hex[:20]
        def update_progress(value):
            with CLUSTER_JOBS_LOCK:
                CLUSTER_PROGRESS[job_id] = {**value, "updated_at": time.time()}
        with CLUSTER_JOBS_LOCK:
            CLUSTER_PROGRESS[job_id] = {"stage": "queued", "completed": 0, "total": count, "detail": "Waiting to start", "updated_at": time.time()}
            CLUSTER_JOBS[job_id] = CLUSTER_EXECUTOR.submit(cluster_scope, source, folder, cluster_count, update_progress)
        self.send_json({"ok": True, "job_id": job_id, "source": source, "folder": folder, "clips": count})

    def api_cluster_get(self, query):
        from bodydata_cluster import CLUSTER_ROOT, load_cluster

        run_id = str((query.get("run") or [""])[0])
        if run_id:
            result = load_cluster(run_id)
            if not result:
                return self.send_error_json(HTTPStatus.NOT_FOUND, "Cluster run not found")
            self.enrich_cluster_result(result)
            result["result_json_url"] = media_url(CLUSTER_ROOT / run_id / "result.json")
            result["members_csv_url"] = media_url(CLUSTER_ROOT / run_id / "members.csv")
            return self.send_json({"ok": True, "status": "ready", "result": result})
        job_id = str((query.get("job") or [""])[0])
        with CLUSTER_JOBS_LOCK:
            future = CLUSTER_JOBS.get(job_id)
            progress = dict(CLUSTER_PROGRESS.get(job_id, {}))
        if future is None:
            return self.send_error_json(HTTPStatus.NOT_FOUND, "Cluster job not found")
        if not future.done():
            return self.send_json({"ok": True, "status": "building", "progress": progress})
        try:
            result = future.result()
        except Exception as exc:
            return self.send_json({"ok": True, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        self.enrich_cluster_result(result)
        result["result_json_url"] = media_url(CLUSTER_ROOT / result["run_id"] / "result.json")
        result["members_csv_url"] = media_url(CLUSTER_ROOT / result["run_id"] / "members.csv")
        self.send_json({"ok": True, "status": "ready", "result": result})

    def enrich_cluster_result(self, result):
        ids = [member["asset_id"] for member in result.get("members", [])]
        db = connect()
        labels = db.execute("SELECT cluster_id,name FROM cluster_labels WHERE run_id=?", (result["run_id"],)).fetchall()
        assets = {}
        if ids:
            placeholders = ",".join("?" for _ in ids)
            rows = db.execute(f"SELECT * FROM assets WHERE id IN ({placeholders})", ids).fetchall()
            assets = {row["id"]: public_asset(row) for row in rows}
        db.close()
        result["labels"] = {str(row["cluster_id"]): row["name"] for row in labels}
        for member in result.get("members", []):
            member["asset"] = assets.get(member["asset_id"], {})

    def api_cluster_label(self, payload):
        run_id = str(payload.get("run_id", ""))
        cluster_id = int(payload.get("cluster_id", -1))
        name = str(payload.get("name", "")).strip()
        if not run_id or cluster_id < 0 or not name:
            return self.send_error_json(HTTPStatus.BAD_REQUEST, "run_id, cluster_id and a non-empty name are required")
        db = connect()
        run = db.execute("SELECT id FROM cluster_runs WHERE id=?", (run_id,)).fetchone()
        if not run:
            db.close()
            return self.send_error_json(HTTPStatus.NOT_FOUND, "Cluster run not found")
        with db:
            db.execute(
                """INSERT INTO cluster_labels(run_id,cluster_id,name,updated_at) VALUES(?,?,?,?)
                   ON CONFLICT(run_id,cluster_id) DO UPDATE SET name=excluded.name,updated_at=excluded.updated_at""",
                (run_id, cluster_id, name, time.time()),
            )
        db.close()
        self.send_json({"ok": True, "run_id": run_id, "cluster_id": cluster_id, "name": name})

    def api_export_start(self, payload):
        from bodydata_export import export_assets

        if ACCESS_GATE.enabled:
            return self.send_error_json(
                HTTPStatus.FORBIDDEN,
                "Direct source export is disabled on the hosted app; export processed results instead.",
            )
        asset_ids = [str(value) for value in payload.get("asset_ids", [])]
        destination = str(payload.get("destination", ""))
        if not asset_ids or not destination.strip():
            return self.send_error_json(HTTPStatus.BAD_REQUEST, "asset_ids and destination are required")
        job_id = uuid.uuid4().hex[:20]
        context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
        with EXPORT_JOBS_LOCK:
            EXPORT_JOBS[job_id] = EXPORT_EXECUTOR.submit(export_assets, asset_ids, destination, context)
        self.send_json({"ok": True, "job_id": job_id, "selected": len(asset_ids)})

    def api_export_get(self, query):
        job_id = str((query.get("job") or [""])[0])
        with EXPORT_JOBS_LOCK:
            future = EXPORT_JOBS.get(job_id)
        if future is None:
            return self.send_error_json(HTTPStatus.NOT_FOUND, "Export job not found")
        if not future.done():
            return self.send_json({"ok": True, "status": "building"})
        try:
            result = future.result()
        except Exception as exc:
            return self.send_json({"ok": True, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        self.send_json({"ok": True, "status": "ready", "result": result})

    def api_augment_start(self, payload):
        from bodydata_augment import MAX_DURATION_SCALE, MAX_VARIANTS, MIN_DURATION_SCALE, augment_assets

        asset_ids = [str(value) for value in payload.get("asset_ids", [])]
        augmentation_type = str(payload.get("augmentation_type", "time"))
        augmentation_methods = [str(value) for value in payload.get("augmentation_methods", ["duration_multiplier"])]
        raw_scales = payload.get("duration_scales", [1.15])
        try:
            duration_scales = list(dict.fromkeys(round(float(value), 4) for value in raw_scales))
        except (TypeError, ValueError):
            return self.send_error_json(HTTPStatus.BAD_REQUEST, "Duration multipliers must be numeric")
        if augmentation_type != "time":
            return self.send_error_json(HTTPStatus.BAD_REQUEST, "Only the validated time augmentation is available in this version")
        if augmentation_methods != ["duration_multiplier"]:
            return self.send_error_json(HTTPStatus.BAD_REQUEST, "Only Duration Multiplier is implemented; planned methods cannot be submitted")
        if not asset_ids:
            return self.send_error_json(HTTPStatus.BAD_REQUEST, "Select at least one motion asset")
        if not duration_scales or len(duration_scales) > MAX_VARIANTS:
            return self.send_error_json(HTTPStatus.BAD_REQUEST, f"Select between 1 and {MAX_VARIANTS} duration multipliers")
        if any(value < MIN_DURATION_SCALE or value > MAX_DURATION_SCALE for value in duration_scales):
            return self.send_error_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                f"Duration multipliers must be between {MIN_DURATION_SCALE:.2f} and {MAX_DURATION_SCALE:.2f}",
            )
        db = connect()
        placeholders = ",".join("?" for _ in asset_ids)
        eligible = db.execute(
            f"SELECT COUNT(*) FROM assets WHERE id IN ({placeholders}) AND kind='smplh_motion' AND format='npz' AND animated=1",
            asset_ids,
        ).fetchone()[0]
        db.close()
        if eligible != len(set(asset_ids)):
            return self.send_error_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "The selected set contains assets not supported by the first temporal augmentation; select indexed AMASS/SMPL-H NPZ motion clips",
            )
        job_id = uuid.uuid4().hex[:20]
        with AUGMENT_JOBS_LOCK:
            AUGMENT_JOBS[job_id] = AUGMENT_EXECUTOR.submit(augment_assets, asset_ids, duration_scales, None, 0)
        self.send_json({
            "ok": True,
            "job_id": job_id,
            "selected": len(asset_ids),
            "variants": len(duration_scales),
            "duration_scales": duration_scales,
            "augmentation_type": augmentation_type,
            "augmentation_methods": augmentation_methods,
            "output_fps": "same_as_each_source",
        })

    def api_augment_get(self, query):
        from bodydata_augment import AUGMENT_ROOT, load_augmentation

        run_id = str((query.get("run") or [""])[0])
        if run_id:
            result = load_augmentation(run_id)
            if not result:
                return self.send_error_json(HTTPStatus.NOT_FOUND, "Augmentation run not found")
            self.enrich_augmentation_result(result)
            result["result_json_url"] = media_url(AUGMENT_ROOT / run_id / "result.json")
            return self.send_json({"ok": True, "status": result.get("status", "ready"), "result": result})
        job_id = str((query.get("job") or [""])[0])
        with AUGMENT_JOBS_LOCK:
            future = AUGMENT_JOBS.get(job_id)
        if future is None:
            return self.send_error_json(HTTPStatus.NOT_FOUND, "Augmentation job not found")
        if not future.done():
            return self.send_json({"ok": True, "status": "building"})
        try:
            result = future.result()
        except Exception as exc:
            return self.send_json({"ok": True, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        self.enrich_augmentation_result(result)
        result["result_json_url"] = media_url(AUGMENT_ROOT / result["run_id"] / "result.json")
        self.send_json({"ok": True, "status": result.get("status", "ready"), "result": result})

    def api_augmentation_save(self, payload):
        from bodydata_augment import AUGMENT_ROOT, save_augmentation

        run_id = str(payload.get("run_id", "")).strip()
        if not run_id:
            return self.send_error_json(HTTPStatus.BAD_REQUEST, "run_id is required")
        result = save_augmentation(run_id)
        self.enrich_augmentation_result(result)
        result["result_json_url"] = media_url(AUGMENT_ROOT / run_id / "result.json")
        self.send_json({"ok": True, "result": result})

    def enrich_augmentation_result(self, result):
        ids = []
        for pair in result.get("pairs", []):
            ids.extend((pair["original_asset_id"], pair["augmented_asset_id"]))
        db = connect()
        assets = {}
        if ids:
            placeholders = ",".join("?" for _ in ids)
            rows = db.execute(f"SELECT * FROM assets WHERE id IN ({placeholders})", ids).fetchall()
            assets = {row["id"]: public_asset(row) for row in rows}
        db.close()
        for pair in result.get("pairs", []):
            pair["original"] = assets.get(pair["original_asset_id"], {})
            pair["augmented"] = assets.get(pair["augmented_asset_id"], {})
            if pair.get("quality_plot_path"):
                pair["quality_plot_url"] = media_url(pair["quality_plot_path"])

    def api_contact_labels_start(self, payload):
        from bodydata_contact import label_assets

        asset_ids = [str(value) for value in payload.get("asset_ids", [])]
        if not asset_ids:
            return self.send_error_json(HTTPStatus.BAD_REQUEST, "Select at least one SMPL-H motion asset")
        force = bool(payload.get("force", False))
        options = dict(payload.get("options") or {})
        job_id = uuid.uuid4().hex[:20]
        with CONTACT_JOBS_LOCK:
            CONTACT_JOBS[job_id] = CONTACT_EXECUTOR.submit(label_assets, asset_ids, force, options)
        self.send_json({"ok": True, "job_id": job_id, "selected": len(asset_ids), "force": force})

    def api_contact_labels_get(self, query):
        from bodydata_contact import load_contact_preview

        asset_id = str((query.get("id") or [""])[0]).strip()
        if asset_id:
            process_run = str((query.get("run") or [""])[0]).strip()
            if process_run:
                from bodydata_process import load_process_contact_preview
                result = load_process_contact_preview(process_run, asset_id)
            else:
                result = load_contact_preview(asset_id)
            if not result:
                return self.send_json({"ok": True, "status": "missing"})
            return self.send_json({"ok": True, "status": "ready", "labels": result})
        job_id = str((query.get("job") or [""])[0]).strip()
        with CONTACT_JOBS_LOCK:
            future = CONTACT_JOBS.get(job_id)
        if future is None:
            return self.send_error_json(HTTPStatus.NOT_FOUND, "Contact-label job not found")
        if not future.done():
            return self.send_json({"ok": True, "status": "building"})
        try:
            result = future.result()
        except Exception as exc:
            return self.send_json({"ok": True, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        self.send_json({"ok": True, "status": result.get("status", "ready"), "result": result})

    def enrich_process_result(self, result):
        augmentation = result.get("augmentation")
        if augmentation:
            self.enrich_augmentation_result(augmentation)
            from bodydata_augment import AUGMENT_ROOT
            augmentation["result_json_url"] = media_url(AUGMENT_ROOT / augmentation["run_id"] / "result.json")
        labeling = result.get("labeling")
        if labeling and labeling.get("items"):
            ids = [item["asset_id"] for item in labeling["items"]]
            placeholders = ",".join("?" for _ in ids)
            db = connect()
            rows = db.execute(f"SELECT * FROM assets WHERE id IN ({placeholders})", ids).fetchall()
            db.close()
            assets = {row["id"]: public_asset(row) for row in rows}
            for item in labeling["items"]:
                item["asset"] = assets.get(item["asset_id"], {})

    def api_process_get(self, query):
        from bodydata_process import list_process_runs, load_process_run

        run_id = str((query.get("run") or [""])[0]).strip()
        if run_id:
            result = load_process_run(run_id)
            if not result:
                return self.send_error_json(HTTPStatus.NOT_FOUND, "Processing run not found")
            self.enrich_process_result(result)
            return self.send_json({"ok": True, "status": result.get("status", "ready"), "result": result})
        job_id = str((query.get("job") or [""])[0]).strip()
        if job_id:
            with PROCESS_JOBS_LOCK:
                future = PROCESS_JOBS.get(job_id)
            if future is None:
                return self.send_error_json(HTTPStatus.NOT_FOUND, "Processing job not found")
            if not future.done():
                return self.send_json({"ok": True, "status": "processing"})
            try:
                result = future.result()
            except Exception as exc:
                return self.send_json({"ok": True, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            self.enrich_process_result(result)
            return self.send_json({"ok": True, "status": result.get("status", "ready"), "result": result})
        scope = str((query.get("scope") or ["temporary"])[0])
        search = str((query.get("q") or [""])[0])
        limit = int((query.get("limit") or [30])[0])
        runs = list_process_runs(scope, search, limit)
        return self.send_json({"ok": True, "scope": scope, "runs": runs, "total": len(runs)})

    def api_process_change(self, payload):
        from bodydata_process import delete_exported_process_run, discard_process_run, run_process, save_process_run, set_process_pinned

        action = str(payload.get("action", "start"))
        if action == "start":
            asset_ids = [str(value) for value in payload.get("asset_ids", [])]
            config = dict(payload.get("config") or {})
            if not asset_ids:
                return self.send_error_json(HTTPStatus.BAD_REQUEST, "Select at least one asset")
            job_id = uuid.uuid4().hex[:20]
            with PROCESS_JOBS_LOCK:
                PROCESS_JOBS[job_id] = PROCESS_EXECUTOR.submit(run_process, asset_ids, config)
            return self.send_json({"ok": True, "job_id": job_id, "selected": len(asset_ids)})
        run_id = str(payload.get("run_id", "")).strip()
        if not run_id:
            return self.send_error_json(HTTPStatus.BAD_REQUEST, "run_id is required")
        if action == "pin":
            return self.send_json({"ok": True, "result": set_process_pinned(run_id, bool(payload.get("pinned", True)))})
        if action == "discard":
            discard_process_run(run_id)
            return self.send_json({"ok": True, "discarded": run_id})
        if action == "delete_exported":
            delete_exported_process_run(run_id)
            return self.send_json({"ok": True, "deleted_exported": run_id})
        if action == "save":
            destination = str(payload.get("destination", "")).strip()
            include_original = bool(payload.get("include_original", False))
            if ACCESS_GATE.enabled or os.environ.get("BODY_DATA_INTEGRATED") == "1":
                destination = str(Path(os.environ.get("BODY_DATA_EXPORT_ROOT", "/data/exports")) / run_id)
                include_original = False
            if not destination:
                return self.send_error_json(HTTPStatus.BAD_REQUEST, "destination is required")
            result = save_process_run(run_id, destination, include_original)
            self.enrich_process_result(result)
            return self.send_json({"ok": True, "result": result})
        if action == "open_folder":
            from bodydata_process import load_process_run
            result = load_process_run(run_id)
            destination = Path((result or {}).get("destination", "")).resolve()
            if not result or result.get("storage_state") != "exported" or not destination.is_dir():
                return self.send_error_json(HTTPStatus.NOT_FOUND, "Exported output folder not found")
            if sys.platform == "win32":
                os.startfile(destination)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(destination)])
            else:
                subprocess.Popen(["xdg-open", str(destination)])
            return self.send_json({"ok": True, "destination": str(destination)})
        return self.send_error_json(HTTPStatus.BAD_REQUEST, "Unknown process action")

    def api_process_download(self, query):
        from bodydata_process import load_process_run

        run_id = str((query.get("run") or [""])[0]).strip()
        result = load_process_run(run_id)
        destination = Path((result or {}).get("destination", "")).resolve()
        export_root = Path(os.environ.get("BODY_DATA_EXPORT_ROOT", "/data/exports")).resolve()
        if not run_id or not result or result.get("storage_state") != "exported" or not destination.is_dir():
            return self.send_error_json(HTTPStatus.NOT_FOUND, "Exported output not found")
        if ACCESS_GATE.enabled and not (destination == export_root or export_root in destination.parents):
            return self.send_error_json(HTTPStatus.FORBIDDEN, "Hosted downloads are restricted to exported results")
        download_root = CACHE_ROOT / "downloads"
        download_root.mkdir(parents=True, exist_ok=True)
        archive = download_root / f"body-data-studio__{run_id}.zip"
        newest_source = max((path.stat().st_mtime for path in destination.rglob("*") if path.is_file()), default=0)
        if not archive.exists() or archive.stat().st_mtime < newest_source:
            temporary = archive.with_suffix(".zip.tmp")
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
                for path in sorted(destination.rglob("*")):
                    if path.is_file():
                        output.write(path, path.relative_to(destination))
            temporary.replace(archive)
        return self.serve_download(archive)

    def serve_download(self, path: Path):
        path = path.resolve()
        allowed = (CACHE_ROOT / "downloads").resolve()
        if not path.is_file() or not (path == allowed or allowed in path.parents):
            return self.send_error_json(HTTPStatus.NOT_FOUND, "Download not found")
        size = path.stat().st_size
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Length", str(size))
        self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.send_header("Cache-Control", "private, no-store")
        self.end_headers()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                self.wfile.write(chunk)

    def api_collections(self, query):
        db = connect()
        rows = db.execute(
            """SELECT c.*, COUNT(ca.asset_id) AS asset_count FROM collections c
               LEFT JOIN collection_assets ca ON ca.collection_id=c.id
               GROUP BY c.id ORDER BY c.updated_at DESC, c.name COLLATE NOCASE"""
        ).fetchall()
        db.close()
        self.send_json({"ok": True, "collections": [dict(row) for row in rows]})

    def api_collections_change(self, payload):
        action = str(payload.get("action", ""))
        db = connect()
        now = time.time()
        if action == "create":
            name = str(payload.get("name", "")).strip()
            if not name:
                db.close()
                return self.send_error_json(HTTPStatus.BAD_REQUEST, "Collection name is required")
            identity = uuid.uuid4().hex[:16]
            try:
                with db:
                    db.execute("INSERT INTO collections(id,name,created_at,updated_at) VALUES(?,?,?,?)", (identity, name, now, now))
            except sqlite3.IntegrityError:
                db.close()
                return self.send_error_json(HTTPStatus.CONFLICT, "A collection with this name already exists")
            db.close()
            return self.send_json({"ok": True, "id": identity, "name": name})
        collection_id = str(payload.get("collection_id", ""))
        asset_ids = [str(value) for value in payload.get("asset_ids", [])]
        if action == "add":
            with db:
                for position, asset_id in enumerate(asset_ids):
                    db.execute(
                        "INSERT OR IGNORE INTO collection_assets(collection_id,asset_id,position,added_at) VALUES(?,?,?,?)",
                        (collection_id, asset_id, position, now),
                    )
                db.execute("UPDATE collections SET updated_at=? WHERE id=?", (now, collection_id))
        elif action == "remove":
            with db:
                for asset_id in asset_ids:
                    db.execute("DELETE FROM collection_assets WHERE collection_id=? AND asset_id=?", (collection_id, asset_id))
                db.execute("UPDATE collections SET updated_at=? WHERE id=?", (now, collection_id))
        elif action == "delete":
            with db:
                db.execute("DELETE FROM collections WHERE id=?", (collection_id,))
        else:
            db.close()
            return self.send_error_json(HTTPStatus.BAD_REQUEST, "Unknown collection action")
        db.close()
        self.send_json({"ok": True})

    def serve_static(self, request_path: str):
        rel = "index.html" if request_path in {"", "/"} else unquote(request_path.lstrip("/"))
        return self.serve_file(STATIC_ROOT / rel)

    def serve_file(self, path: Path):
        try:
            path = path.resolve()
            allowed = [STATIC_ROOT.resolve(), NODE_MODULES.resolve()]
            if not any(path == root or root in path.parents for root in allowed):
                return self.send_error_json(HTTPStatus.FORBIDDEN, "Forbidden path")
        except OSError:
            return self.send_error_json(HTTPStatus.NOT_FOUND, "File not found")
        if not path.is_file():
            return self.send_error_json(HTTPStatus.NOT_FOUND, "File not found")
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def serve_media(self, request_path: str):
        parts = request_path.split("/", 3)
        if len(parts) < 3:
            return self.send_error_json(HTTPStatus.NOT_FOUND, "Invalid media URL")
        token = parts[2]
        suffix = unquote(parts[3]) if len(parts) > 3 else ""
        base = resolve_media_token(token)
        if base is None:
            return self.send_error_json(HTTPStatus.NOT_FOUND, "Media token expired; preview the asset again")
        path = base / suffix if base.is_dir() else base
        path = path.resolve()
        if base.is_dir() and not (path == base or base in path.parents):
            return self.send_error_json(HTTPStatus.FORBIDDEN, "Forbidden media path")
        if not path.is_file():
            return self.send_error_json(HTTPStatus.NOT_FOUND, "Media file not found")
        size = path.stat().st_size
        start, end = 0, size - 1
        range_header = self.headers.get("Range")
        status = HTTPStatus.OK
        if range_header and range_header.startswith("bytes="):
            spec = range_header[6:].split(",", 1)[0]
            left, right = spec.split("-", 1)
            start = int(left or 0)
            end = min(int(right) if right else size - 1, size - 1)
            status = HTTPStatus.PARTIAL_CONTENT
        length = max(0, end - start + 1)
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.name.endswith(".json.gz"):
            content_type = "application/json"
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if path.name.endswith(".json.gz"):
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def main():
    parser = argparse.ArgumentParser(description="Body Data Studio local server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open the local app in the default browser")
    parser.add_argument("--full-index", action="store_true", help="Start the slower AMASS index in the background")
    args = parser.parse_args()
    db = connect()
    count = db.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
    db.close()
    if count == 0:
        build_index(full=False)
    from bodydata_process import prune_temporary_runs
    prune_temporary_runs()
    if args.full_index:
        start_index(full=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Body Data Studio: {url}")
    print(f"Raw data (read-only): {DATA_ROOT}")
    print(f"Preview cache: {CACHE_ROOT}")
    if args.open:
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
