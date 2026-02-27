"""Public playground API with short-lived cookie session and guardrails."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import math
import secrets
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Body, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.api.v1.chat import ChatCompletionRequest, validate_request
from app.core.config import get_config
from app.core.logger import logger
from app.services.grok.chat import ChatService
from app.services.grok.model import ModelService


router = APIRouter(tags=["Public"])

_RUNTIME_PUBLIC_SECRET = secrets.token_bytes(32)
_state_lock = asyncio.Lock()


@dataclass
class _SessionState:
    sid: str
    ip: str
    ua_hash: str
    expires_at: int


@dataclass
class _WindowState:
    window_start: int
    count: int


_sessions: dict[str, _SessionState] = {}
_rate_windows: dict[str, dict[str, _WindowState]] = {
    "session_issue_ip": {},
    "chat_ip": {},
    "chat_session": {},
}
_last_prune_ts = 0


class PublicSessionRequest(BaseModel):
    captcha_token: Optional[str] = Field(default=None, description="Captcha token when enabled")


def _public_cfg(path: str, default: Any) -> Any:
    return get_config(f"public.{path}", default)


def _now_ts() -> int:
    return int(time.time())


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _ensure_public_enabled() -> None:
    if bool(_public_cfg("enabled", True)):
        return
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Public channel is disabled")


def _client_ip(request: Request) -> str:
    xff = str(request.headers.get("x-forwarded-for") or "").strip()
    if xff:
        return xff.split(",", 1)[0].strip() or "unknown"
    if request.client and request.client.host:
        return str(request.client.host)
    return "unknown"


def _ua_hash(request: Request) -> str:
    ua = str(request.headers.get("user-agent") or "")
    return hashlib.sha256(ua.encode("utf-8")).hexdigest()


def _cookie_name() -> str:
    name = str(_public_cfg("cookie_name", "grok2api_public_session") or "").strip()
    return name or "grok2api_public_session"


def _cookie_ttl() -> int:
    try:
        ttl = int(_public_cfg("session_ttl_seconds", 1800) or 1800)
    except Exception:
        ttl = 1800
    return max(60, min(24 * 3600, ttl))


def _public_secret() -> bytes:
    raw = str(_public_cfg("hmac_secret", "") or "").strip()
    if raw:
        return raw.encode("utf-8")
    app_key = str(get_config("app.app_key", "") or "").strip()
    if app_key:
        return app_key.encode("utf-8")
    return _RUNTIME_PUBLIC_SECRET


def _sign_payload(payload: str) -> str:
    mac = hmac.new(_public_secret(), payload.encode("utf-8"), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(mac).decode("ascii").rstrip("=")


def _encode_cookie_value(sid: str, exp: int) -> str:
    payload = f"{sid}.{exp}"
    sig = _sign_payload(payload)
    return f"{payload}.{sig}"


def _decode_cookie_value(raw: str) -> tuple[str, int] | None:
    parts = str(raw or "").strip().split(".")
    if len(parts) != 3:
        return None
    sid, exp_raw, sig = parts
    if not sid or not exp_raw or not sig:
        return None
    try:
        exp = int(exp_raw)
    except Exception:
        return None
    payload = f"{sid}.{exp}"
    expected = _sign_payload(payload)
    if not hmac.compare_digest(sig, expected):
        return None
    return sid, exp


async def _prune_state(now_ts: int) -> None:
    global _last_prune_ts
    if now_ts - _last_prune_ts < 30:
        return

    for sid, state in list(_sessions.items()):
        if state.expires_at <= now_ts:
            _sessions.pop(sid, None)

    for bucket in _rate_windows.values():
        for key, state in list(bucket.items()):
            if now_ts - state.window_start > 300:
                bucket.pop(key, None)

    _last_prune_ts = now_ts


async def _check_rate_limit(bucket_name: str, key: str, limit: int, window_sec: int = 60) -> bool:
    if limit <= 0:
        return True

    now_ts = _now_ts()
    async with _state_lock:
        await _prune_state(now_ts)
        bucket = _rate_windows[bucket_name]
        state = bucket.get(key)
        if state is None or now_ts - state.window_start >= window_sec:
            bucket[key] = _WindowState(window_start=now_ts, count=1)
            return True

        if state.count >= limit:
            return False

        state.count += 1
        return True


def _allowed_public_models() -> list[str]:
    configured = _public_cfg(
        "allowed_models",
        ["grok-4.20-beta", "grok-4", "grok-4-mini", "grok-3-mini", "grok-3"],
    )
    values = configured if isinstance(configured, list) else []
    out: list[str] = []
    for item in values:
        model_id = str(item or "").strip()
        if not model_id or not ModelService.valid(model_id):
            continue
        info = ModelService.get(model_id)
        if info and (info.is_image or info.is_video):
            continue
        out.append(model_id)
    if out:
        return out
    return ["grok-4.20-beta"]


def _model_objects(model_ids: list[str]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for model_id in model_ids:
        info = ModelService.get(model_id)
        items.append(
            {
                "id": model_id,
                "display_name": info.display_name if info else model_id,
            }
        )
    return items


def _estimate_input_size(messages: list[Any]) -> tuple[int, int]:
    total_chars = 0

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")

        if isinstance(content, str):
            total_chars += len(content)
            continue

        if isinstance(content, dict):
            total_chars += len(str(content))
            continue

        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    total_chars += len(str(block))
                    continue
                block_type = str(block.get("type") or "")
                if block_type == "text":
                    total_chars += len(str(block.get("text") or ""))
                elif block_type == "image_url":
                    image_url = block.get("image_url")
                    if isinstance(image_url, dict):
                        total_chars += len(str(image_url.get("url") or ""))
                    else:
                        total_chars += len(str(image_url or ""))
                else:
                    total_chars += len(str(block))
            continue

        total_chars += len(str(content))

    approx_tokens = max(1, math.ceil(total_chars / 4)) if total_chars > 0 else 0
    return total_chars, approx_tokens


async def _verify_captcha_if_needed(captcha_token: str, ip: str) -> None:
    enabled = bool(_public_cfg("captcha_enabled", False))
    if not enabled:
        return

    secret = str(_public_cfg("captcha_secret", "") or "").strip()
    if not secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Captcha is enabled but public.captcha_secret is not configured",
        )

    token = str(captcha_token or "").strip()
    if not token:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="captcha_token is required")

    verify_url = str(
        _public_cfg("captcha_verify_url", "https://challenges.cloudflare.com/turnstile/v0/siteverify")
        or "https://challenges.cloudflare.com/turnstile/v0/siteverify"
    ).strip()

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(
                verify_url,
                data={
                    "secret": secret,
                    "response": token,
                    "remoteip": ip,
                },
            )
            payload = resp.json()
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning(f"Public captcha verify failed: {exc}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="captcha verify failed") from exc

    if not bool(payload.get("success", False)):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="captcha verification failed")


async def _require_session(request: Request) -> _SessionState:
    raw = request.cookies.get(_cookie_name())
    decoded = _decode_cookie_value(str(raw or ""))
    if not decoded:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing public session")

    sid, exp = decoded
    now_ts = _now_ts()
    if exp <= now_ts:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Public session expired")

    ip = _client_ip(request)
    ua = _ua_hash(request)

    async with _state_lock:
        await _prune_state(now_ts)
        state = _sessions.get(sid)
        if not state:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid public session")
        if state.expires_at <= now_ts:
            _sessions.pop(sid, None)
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Public session expired")
        if state.ip != ip or state.ua_hash != ua:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Public session mismatch")
        return state


@router.post("/session")
async def create_public_session(request: Request, body: PublicSessionRequest | None = Body(default=None)):
    """Issue short-lived signed HttpOnly cookie for public playground."""

    _ensure_public_enabled()

    ip = _client_ip(request)
    issue_limit = _as_int(_public_cfg("session_issue_ip_rate_limit_per_min", 10), 10)
    if not await _check_rate_limit("session_issue_ip", ip, issue_limit):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Too many session requests")

    await _verify_captcha_if_needed((body.captcha_token if body else "") or "", ip)

    sid = secrets.token_urlsafe(24)
    ttl = _cookie_ttl()
    exp = _now_ts() + ttl
    state = _SessionState(sid=sid, ip=ip, ua_hash=_ua_hash(request), expires_at=exp)

    async with _state_lock:
        await _prune_state(_now_ts())
        _sessions[sid] = state

    models = _allowed_public_models()
    resp = JSONResponse(
        {
            "status": "ok",
            "expires_in": ttl,
            "models": _model_objects(models),
        }
    )
    resp.set_cookie(
        key=_cookie_name(),
        value=_encode_cookie_value(sid, exp),
        max_age=ttl,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="lax",
        path="/api/v1/public",
    )
    return resp


@router.post("/chat/completions")
async def public_chat_completions(request: Request, payload: ChatCompletionRequest):
    """Public chat endpoint that uses signed session cookie instead of API key."""

    _ensure_public_enabled()

    session_state = await _require_session(request)
    ip = _client_ip(request)

    ip_limit = _as_int(_public_cfg("ip_rate_limit_per_min", 60), 60)
    if not await _check_rate_limit("chat_ip", ip, ip_limit):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="IP rate limit exceeded")

    session_limit = _as_int(_public_cfg("session_rate_limit_per_min", 30), 30)
    if not await _check_rate_limit("chat_session", session_state.sid, session_limit):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Session rate limit exceeded")

    allowed_models = set(_allowed_public_models())
    if payload.model not in allowed_models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{payload.model}' is not allowed for public channel",
        )

    validate_request(payload)

    max_messages = _as_int(_public_cfg("max_messages", 24), 24)
    if len(payload.messages) > max_messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many messages: {len(payload.messages)} > {max_messages}",
        )

    total_chars, approx_tokens = _estimate_input_size([m.model_dump() for m in payload.messages])
    max_chars = _as_int(_public_cfg("max_input_chars", 12000), 12000)
    max_tokens = _as_int(_public_cfg("max_input_tokens", 3000), 3000)

    if total_chars > max_chars:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Input too large (chars={total_chars}, limit={max_chars})",
        )
    if approx_tokens > max_tokens:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Input too large (approx_tokens={approx_tokens}, limit={max_tokens})",
        )

    effective_thinking = payload.thinking
    if effective_thinking is None and payload.reasoning_effort is not None:
        effective_thinking = "disabled" if payload.reasoning_effort == "none" else "enabled"

    result = await ChatService.completions(
        model=payload.model,
        messages=[m.model_dump() for m in payload.messages],
        stream=payload.stream,
        thinking=effective_thinking,
    )

    if isinstance(result, dict):
        return JSONResponse(content=result)

    return StreamingResponse(
        result,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


__all__ = ["router"]
