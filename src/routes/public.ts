import { Hono } from "hono";
import type { Env } from "../env";
import { getSettings } from "../settings";
import { MODEL_CONFIG, isValidModel } from "../grok/models";
import { extractContent, buildConversationPayload, sendConversationRequest } from "../grok/conversation";
import { uploadImage } from "../grok/upload";
import { createMediaPost, createPost } from "../grok/create";
import { createOpenAiStreamFromGrokNdjson, parseOpenAiFromGrokNdjson } from "../grok/processor";
import { addRequestLog } from "../repo/logs";
import { applyCooldown, recordTokenFailure, selectBestToken } from "../repo/tokens";

function openAiError(message: string, code: string): Record<string, unknown> {
  return { error: { message, type: "invalid_request_error", code } };
}

function getClientIp(req: Request): string {
  return (
    req.headers.get("CF-Connecting-IP") ||
    req.headers.get("X-Forwarded-For")?.split(",")[0]?.trim() ||
    "0.0.0.0"
  );
}

function toBool(input: unknown): boolean {
  if (typeof input === "boolean") return input;
  if (typeof input === "number") return input === 1;
  if (typeof input !== "string") return false;
  const normalized = input.trim().toLowerCase();
  return normalized === "true" || normalized === "1" || normalized === "yes";
}

function parseIntSafe(v: string | undefined, fallback: number): number {
  const n = Number(v);
  if (!Number.isFinite(n)) return fallback;
  return Math.floor(n);
}

async function mapLimit<T, R>(
  items: T[],
  limit: number,
  fn: (item: T) => Promise<R>,
): Promise<R[]> {
  const results: R[] = [];
  const queue = items.slice();
  const workers = Array.from({ length: Math.max(1, limit) }, async () => {
    while (queue.length) {
      const item = queue.shift() as T;
      results.push(await fn(item));
    }
  });
  await Promise.all(workers);
  return results;
}

function normalizeSsoToken(raw: string): string {
  const token = String(raw ?? "").trim();
  if (!token) return "";
  return token.startsWith("sso=") ? token.slice(4).trim() : token;
}

function cookieName(settings: Awaited<ReturnType<typeof getSettings>>): string {
  const v = String(settings.public.cookie_name ?? "").trim();
  return v || "grok2api_public_session";
}

function cookieTtlSeconds(settings: Awaited<ReturnType<typeof getSettings>>): number {
  const raw = Number(settings.public.session_ttl_seconds ?? 1800);
  if (!Number.isFinite(raw)) return 1800;
  return Math.max(60, Math.min(24 * 3600, Math.floor(raw)));
}

function publicEnabled(settings: Awaited<ReturnType<typeof getSettings>>): boolean {
  return Boolean(settings.public.enabled);
}

function normalizeCfCookie(value: string): string {
  const v = String(value ?? "").trim();
  if (!v) return "";
  return v.startsWith("cf_clearance=") ? v : `cf_clearance=${v}`;
}

function bytesToBase64Url(bytes: Uint8Array): string {
  let binary = "";
  for (const b of bytes) binary += String.fromCharCode(b);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}

function base64UrlToBytes(input: string): Uint8Array {
  const normalized = input.replace(/-/g, "+").replace(/_/g, "/");
  const pad = normalized.length % 4 === 0 ? "" : "=".repeat(4 - (normalized.length % 4));
  const binary = atob(normalized + pad);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

function secureCompare(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a[i]! ^ b[i]!;
  return diff === 0;
}

function getSigningSecret(settings: Awaited<ReturnType<typeof getSettings>>): string {
  const fromPublic = String(settings.public.hmac_secret ?? "").trim();
  if (fromPublic) return fromPublic;
  const fromApp = String(settings.global.admin_password ?? "").trim();
  if (fromApp) return fromApp;
  return "grok2api-public-fallback-secret";
}

async function hmacSha256(message: string, secret: string): Promise<Uint8Array> {
  const enc = new TextEncoder();
  const keyData = enc.encode(secret);
  const msgData = enc.encode(message);
  const key = await crypto.subtle.importKey(
    "raw",
    keyData,
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const sig = await crypto.subtle.sign("HMAC", key, msgData);
  return new Uint8Array(sig);
}

function setCookieHeader(name: string, value: string, ttlSeconds: number, secure: boolean): string {
  const parts = [
    `${name}=${value}`,
    "HttpOnly",
    "Path=/api/v1/public",
    "SameSite=Lax",
    `Max-Age=${ttlSeconds}`,
  ];
  if (secure) parts.push("Secure");
  return parts.join("; ");
}

function parseCookieHeader(header: string | null): Record<string, string> {
  if (!header) return {};
  const out: Record<string, string> = {};
  const parts = header.split(";");
  for (const part of parts) {
    const idx = part.indexOf("=");
    if (idx <= 0) continue;
    const key = part.slice(0, idx).trim();
    const value = part.slice(idx + 1).trim();
    if (!key) continue;
    out[key] = value;
  }
  return out;
}

function localDayString(tsMs: number, tzOffsetMinutes: number): string {
  const local = new Date(tsMs + tzOffsetMinutes * 60_000);
  const y = local.getUTCFullYear();
  const m = String(local.getUTCMonth() + 1).padStart(2, "0");
  const d = String(local.getUTCDate()).padStart(2, "0");
  return `${y}-${m}-${d}`;
}

function nowMs(): number {
  return Date.now();
}

async function ensurePublicSchema(env: Env): Promise<void> {
  await env.DB.prepare(
    "CREATE TABLE IF NOT EXISTS public_sessions (sid TEXT PRIMARY KEY, ip TEXT NOT NULL, ua_hash TEXT NOT NULL, expires_at INTEGER NOT NULL, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL)",
  ).run();
  await env.DB.prepare(
    "CREATE INDEX IF NOT EXISTS idx_public_sessions_expires ON public_sessions(expires_at)",
  ).run();
  await env.DB.prepare(
    "CREATE TABLE IF NOT EXISTS public_rate_limits (bucket TEXT NOT NULL, rate_key TEXT NOT NULL, window_start INTEGER NOT NULL, count INTEGER NOT NULL DEFAULT 0, updated_at INTEGER NOT NULL, day TEXT, PRIMARY KEY (bucket, rate_key, window_start))",
  ).run();
  await env.DB.prepare(
    "CREATE INDEX IF NOT EXISTS idx_public_rate_limits_updated ON public_rate_limits(updated_at)",
  ).run();
}

async function checkAndConsumeRateLimit(args: {
  env: Env;
  bucket: string;
  key: string;
  limit: number;
  windowSec?: number;
}): Promise<boolean> {
  if (args.limit <= 0) return true;
  const windowSec = Math.max(1, Math.floor(args.windowSec ?? 60));
  const now = nowMs();
  const windowStart = now - now % (windowSec * 1000);
  const tz = parseIntSafe(args.env.CACHE_RESET_TZ_OFFSET_MINUTES, 0);
  const day = localDayString(now, tz);

  await args.env.DB.prepare(
    "INSERT INTO public_rate_limits(bucket, rate_key, window_start, count, updated_at, day) VALUES(?,?,?,?,?,?) ON CONFLICT(bucket, rate_key, window_start) DO NOTHING",
  )
    .bind(args.bucket, args.key, windowStart, 0, now, day)
    .run();

  const update = await args.env.DB.prepare(
    "UPDATE public_rate_limits SET count = count + 1, updated_at = ? WHERE bucket = ? AND rate_key = ? AND window_start = ? AND count < ?",
  )
    .bind(now, args.bucket, args.key, windowStart, args.limit)
    .run();

  const changed = Number((update as any)?.meta?.changes ?? 0);
  return changed > 0;
}

async function pruneExpiredPublicSessions(env: Env): Promise<void> {
  const now = Math.floor(Date.now() / 1000);
  await env.DB.prepare("DELETE FROM public_sessions WHERE expires_at <= ?").bind(now).run();
}

async function signSessionValue(args: {
  sid: string;
  exp: number;
  ip: string;
  uaHash: string;
  secret: string;
}): Promise<string> {
  const payload = `${args.sid}.${args.exp}.${args.ip}.${args.uaHash}`;
  const sig = await hmacSha256(payload, args.secret);
  return `${args.sid}.${args.exp}.${bytesToBase64Url(sig)}`;
}

async function verifySessionCookie(args: {
  cookieValue: string;
  ip: string;
  uaHash: string;
  secret: string;
}): Promise<{ sid: string; exp: number } | null> {
  const parts = String(args.cookieValue || "").trim().split(".");
  if (parts.length !== 3) return null;
  const sid = String(parts[0] ?? "").trim();
  const expRaw = String(parts[1] ?? "").trim();
  const sigRaw = String(parts[2] ?? "").trim();
  if (!sid || !expRaw || !sigRaw) return null;

  const exp = Number(expRaw);
  if (!Number.isFinite(exp)) return null;
  if (Math.floor(Date.now() / 1000) >= Math.floor(exp)) return null;

  const payload = `${sid}.${Math.floor(exp)}.${args.ip}.${args.uaHash}`;
  const expected = await hmacSha256(payload, args.secret);
  let provided: Uint8Array;
  try {
    provided = base64UrlToBytes(sigRaw);
  } catch {
    return null;
  }
  if (!secureCompare(provided, expected)) return null;
  return { sid, exp: Math.floor(exp) };
}

function isImageOrVideoModel(model: string): boolean {
  const cfg = MODEL_CONFIG[model];
  return Boolean(cfg?.is_image_model || cfg?.is_video_model);
}

function allowedPublicModels(settings: Awaited<ReturnType<typeof getSettings>>): string[] {
  const raw = Array.isArray(settings.public.allowed_models)
    ? settings.public.allowed_models
    : ["grok-4.20-beta", "grok-4", "grok-4-mini", "grok-3-mini", "grok-3"];
  const out: string[] = [];
  for (const item of raw) {
    const id = String(item ?? "").trim();
    if (!id) continue;
    if (!isValidModel(id)) continue;
    if (isImageOrVideoModel(id)) continue;
    out.push(id);
  }
  return out.length ? out : ["grok-4.20-beta"];
}

async function validatePublicSession(request: Request, env: Env): Promise<{ sid: string }> {
  const settings = await getSettings(env);
  if (!publicEnabled(settings)) {
    throw new Response(JSON.stringify({ error: "Public channel is disabled" }), {
      status: 404,
      headers: { "content-type": "application/json; charset=utf-8" },
    });
  }

  await pruneExpiredPublicSessions(env);

  const ip = getClientIp(request);
  const uaHash = await (async () => {
    const ua = String(request.headers.get("user-agent") ?? "");
    const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(ua));
    return bytesToBase64Url(new Uint8Array(digest));
  })();

  const cookies = parseCookieHeader(request.headers.get("Cookie"));
  const cookieVal = cookies[cookieName(settings)];
  if (!cookieVal) {
    throw new Response(JSON.stringify({ error: "Missing public session" }), {
      status: 401,
      headers: { "content-type": "application/json; charset=utf-8" },
    });
  }

  const verified = await verifySessionCookie({
    cookieValue: cookieVal,
    ip,
    uaHash,
    secret: getSigningSecret(settings),
  });
  if (!verified) {
    throw new Response(JSON.stringify({ error: "Invalid public session" }), {
      status: 401,
      headers: { "content-type": "application/json; charset=utf-8" },
    });
  }

  const sessionRow = await env.DB.prepare(
    "SELECT sid, expires_at, ip, ua_hash FROM public_sessions WHERE sid = ?",
  )
    .bind(verified.sid)
    .first<{ sid: string; expires_at: number; ip: string; ua_hash: string }>();

  if (!sessionRow) {
    throw new Response(JSON.stringify({ error: "Invalid public session" }), {
      status: 401,
      headers: { "content-type": "application/json; charset=utf-8" },
    });
  }

  const nowSec = Math.floor(Date.now() / 1000);
  if (Number(sessionRow.expires_at ?? 0) <= nowSec) {
    await env.DB.prepare("DELETE FROM public_sessions WHERE sid = ?").bind(verified.sid).run();
    throw new Response(JSON.stringify({ error: "Public session expired" }), {
      status: 401,
      headers: { "content-type": "application/json; charset=utf-8" },
    });
  }

  if (String(sessionRow.ip ?? "") !== ip || String(sessionRow.ua_hash ?? "") !== uaHash) {
    throw new Response(JSON.stringify({ error: "Public session mismatch" }), {
      status: 401,
      headers: { "content-type": "application/json; charset=utf-8" },
    });
  }

  return { sid: verified.sid };
}

export const publicRoutes = new Hono<{ Bindings: Env }>();

publicRoutes.post("/api/v1/public/session", async (c) => {
  await ensurePublicSchema(c.env);
  const settings = await getSettings(c.env);
  if (!publicEnabled(settings)) return c.json({ error: "Public channel is disabled" }, 404);

  const ip = getClientIp(c.req.raw);
  const ua = String(c.req.header("user-agent") ?? "");
  const uaDigest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(ua));
  const uaHash = bytesToBase64Url(new Uint8Array(uaDigest));

  const issueLimit = Math.max(1, Math.floor(Number(settings.public.session_issue_ip_rate_limit_per_min ?? 10) || 10));
  const issueAllowed = await checkAndConsumeRateLimit({
    env: c.env,
    bucket: "session_issue_ip",
    key: ip,
    limit: issueLimit,
    windowSec: 60,
  });
  if (!issueAllowed) return c.json({ error: "Too many session requests" }, 429);

  const sid = crypto.randomUUID().replaceAll("-", "");
  const ttl = cookieTtlSeconds(settings);
  const exp = Math.floor(Date.now() / 1000) + ttl;

  await c.env.DB.prepare(
    "INSERT INTO public_sessions(sid, ip, ua_hash, expires_at, created_at, updated_at) VALUES(?,?,?,?,?,?) ON CONFLICT(sid) DO UPDATE SET ip=excluded.ip, ua_hash=excluded.ua_hash, expires_at=excluded.expires_at, updated_at=excluded.updated_at",
  )
    .bind(sid, ip, uaHash, exp, Date.now(), Date.now())
    .run();

  const signed = await signSessionValue({ sid, exp, ip, uaHash, secret: getSigningSecret(settings) });

  const models = allowedPublicModels(settings).map((id) => ({
    id,
    display_name: MODEL_CONFIG[id]?.display_name ?? id,
  }));

  const headers = new Headers({ "content-type": "application/json; charset=utf-8" });
  headers.append("Set-Cookie", setCookieHeader(cookieName(settings), signed, ttl, c.req.url.startsWith("https://")));

  return new Response(
    JSON.stringify({ status: "ok", expires_in: ttl, models }),
    { status: 200, headers },
  );
});

publicRoutes.post("/api/v1/public/chat/completions", async (c) => {
  await ensurePublicSchema(c.env);
  let session: { sid: string };
  try {
    session = await validatePublicSession(c.req.raw, c.env);
  } catch (resp) {
    if (resp instanceof Response) return resp;
    return c.json({ error: "Invalid public session" }, 401);
  }

  const settingsBundle = await getSettings(c.env);
  const allowedModels = new Set(allowedPublicModels(settingsBundle));

  const body = (await c.req.json()) as {
    model?: string;
    messages?: any[];
    stream?: boolean;
    video_config?: {
      aspect_ratio?: string;
      video_length?: number;
      resolution?: string;
      preset?: string;
    };
  };

  const requestedModel = String(body.model ?? "");
  if (!requestedModel) return c.json(openAiError("Missing 'model'", "missing_model"), 400);
  if (!allowedModels.has(requestedModel)) {
    return c.json(openAiError(`Model '${requestedModel}' is not allowed for public channel`, "model_not_allowed"), 400);
  }
  if (!Array.isArray(body.messages)) return c.json(openAiError("Missing 'messages'", "missing_messages"), 400);
  if (!isValidModel(requestedModel)) return c.json(openAiError(`Model '${requestedModel}' not supported`, "model_not_supported"), 400);

  const maxMessages = Math.max(1, Math.floor(Number(settingsBundle.public.max_messages ?? 24) || 24));
  if (body.messages.length > maxMessages) {
    return c.json(openAiError(`Too many messages: ${body.messages.length} > ${maxMessages}`, "too_many_messages"), 400);
  }

  const { content } = extractContent(body.messages as any);
  const maxInputChars = Math.max(1, Math.floor(Number(settingsBundle.public.max_input_chars ?? 12000) || 12000));
  if (content.length > maxInputChars) {
    return c.json(openAiError(`Input too large (chars=${content.length}, limit=${maxInputChars})`, "input_too_large"), 400);
  }
  const approxTokens = Math.max(0, Math.ceil(content.length / 4));
  const maxInputTokens = Math.max(1, Math.floor(Number(settingsBundle.public.max_input_tokens ?? 3000) || 3000));
  if (approxTokens > maxInputTokens) {
    return c.json(openAiError(`Input too large (approx_tokens=${approxTokens}, limit=${maxInputTokens})`, "input_too_large"), 400);
  }

  const ip = getClientIp(c.req.raw);
  const ipRateLimit = Math.max(1, Math.floor(Number(settingsBundle.public.ip_rate_limit_per_min ?? 60) || 60));
  const ipAllowed = await checkAndConsumeRateLimit({
    env: c.env,
    bucket: "chat_ip",
    key: ip,
    limit: ipRateLimit,
    windowSec: 60,
  });
  if (!ipAllowed) return c.json({ error: "IP rate limit exceeded" }, 429);

  const sessionRateLimit = Math.max(1, Math.floor(Number(settingsBundle.public.session_rate_limit_per_min ?? 30) || 30));
  const sessionAllowed = await checkAndConsumeRateLimit({
    env: c.env,
    bucket: "chat_session",
    key: session.sid,
    limit: sessionRateLimit,
    windowSec: 60,
  });
  if (!sessionAllowed) return c.json({ error: "Session rate limit exceeded" }, 429);

  const start = Date.now();
  const origin = new URL(c.req.url).origin;
  const keyName = "Public Session";

  try {
    const retryCodes = Array.isArray(settingsBundle.grok.retry_status_codes)
      ? settingsBundle.grok.retry_status_codes
      : [401, 429];
    const stream = Boolean(body.stream);
    const maxRetry = 3;
    let lastErr: string | null = null;

    for (let attempt = 0; attempt < maxRetry; attempt++) {
      const chosen = await selectBestToken(c.env.DB, requestedModel);
      if (!chosen) return c.json(openAiError("No available token", "NO_AVAILABLE_TOKEN"), 503);

      const jwt = normalizeSsoToken(chosen.token);
      const cfCookie = normalizeCfCookie(settingsBundle.grok.cf_clearance ?? "");
      const cookie = cfCookie ? `sso-rw=${jwt};sso=${jwt};${cfCookie}` : `sso-rw=${jwt};sso=${jwt}`;

      const { content: mergedContent, images } = extractContent(body.messages as any);
      const cfg = MODEL_CONFIG[requestedModel]!;
      const isVideoModel = Boolean(cfg.is_video_model);
      const imgInputs = isVideoModel && images.length > 1 ? images.slice(0, 1) : images;

      try {
        const uploads = await mapLimit(imgInputs, 5, (u) => uploadImage(u, cookie, settingsBundle.grok));
        const uploadedRows = uploads as Array<{ fileId?: string; fileUri?: string }>;
        const imgIds = uploadedRows.map((u) => u.fileId).filter(Boolean) as string[];
        const imgUris = uploadedRows.map((u) => u.fileUri).filter(Boolean) as string[];

        let postId: string | undefined;
        if (isVideoModel) {
          if (imgUris.length) {
            const post = await createPost(imgUris[0]!, cookie, settingsBundle.grok);
            postId = post.postId || undefined;
          } else {
            const post = await createMediaPost(
              { mediaType: "MEDIA_POST_TYPE_VIDEO", prompt: mergedContent },
              cookie,
              settingsBundle.grok,
            );
            postId = post.postId || undefined;
          }
        }

        const { payload, referer } = buildConversationPayload({
          requestModel: requestedModel,
          content: mergedContent,
          imgIds,
          imgUris,
          ...(postId ? { postId } : {}),
          ...(isVideoModel && body.video_config ? { videoConfig: body.video_config } : {}),
          settings: settingsBundle.grok,
        });

        const upstream = await sendConversationRequest({
          payload,
          cookie,
          settings: settingsBundle.grok,
          ...(referer ? { referer } : {}),
        });

        if (!upstream.ok) {
          const txt = await upstream.text().catch(() => "");
          lastErr = `Upstream ${upstream.status}: ${txt.slice(0, 200)}`;
          await recordTokenFailure(c.env.DB, jwt, upstream.status, txt.slice(0, 200));
          await applyCooldown(c.env.DB, jwt, upstream.status);
          if (retryCodes.includes(upstream.status) && attempt < maxRetry - 1) continue;
          break;
        }

        if (stream) {
          const sse = createOpenAiStreamFromGrokNdjson(upstream, {
            cookie,
            settings: settingsBundle.grok,
            global: settingsBundle.global,
            origin,
            requestedModel,
            onFinish: async ({ status, duration }) => {
              await addRequestLog(c.env.DB, {
                ip,
                model: requestedModel,
                duration: Number(duration.toFixed(2)),
                status,
                key_name: keyName,
                token_suffix: jwt.slice(-6),
                error: status === 200 ? "" : "stream_error",
              });
            },
          });

          return new Response(sse, {
            status: 200,
            headers: {
              "Content-Type": "text/event-stream; charset=utf-8",
              "Cache-Control": "no-cache",
              Connection: "keep-alive",
              "X-Accel-Buffering": "no",
              "Access-Control-Allow-Origin": "*",
            },
          });
        }

        const json = await parseOpenAiFromGrokNdjson(upstream, {
          cookie,
          settings: settingsBundle.grok,
          global: settingsBundle.global,
          origin,
          requestedModel,
        });

        const duration = (Date.now() - start) / 1000;
        await addRequestLog(c.env.DB, {
          ip,
          model: requestedModel,
          duration: Number(duration.toFixed(2)),
          status: 200,
          key_name: keyName,
          token_suffix: jwt.slice(-6),
          error: "",
        });

        return c.json(json);
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        lastErr = msg;
        await recordTokenFailure(c.env.DB, jwt, 500, msg);
        await applyCooldown(c.env.DB, jwt, 500);
        if (attempt < maxRetry - 1) continue;
      }
    }

    const duration = (Date.now() - start) / 1000;
    await addRequestLog(c.env.DB, {
      ip,
      model: requestedModel,
      duration: Number(duration.toFixed(2)),
      status: 500,
      key_name: keyName,
      token_suffix: "",
      error: "upstream_error",
    });

    return c.json(openAiError("Upstream error", "upstream_error"), 500);
  } catch (e) {
    const duration = (Date.now() - start) / 1000;
    await addRequestLog(c.env.DB, {
      ip,
      model: requestedModel || "unknown",
      duration: Number(duration.toFixed(2)),
      status: 500,
      key_name: keyName,
      token_suffix: "",
      error: e instanceof Error ? e.message : String(e),
    });
    return c.json(openAiError("Internal error", "internal_error"), 500);
  }
});
