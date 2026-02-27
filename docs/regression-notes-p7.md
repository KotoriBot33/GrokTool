# Regression Notes (P7)

Date: 2026-02-27
Scope: P3 WS waterfall enhancement, P4 image edit alignment, P5 video stitching enhancement, P6 NSFW cross-page flow.

## Environment
- Runtime: local repository state on `main`
- Python compile check for touched Python files: none touched in P7
- Test runner: `pytest` not available in current shell environment

## Regression Checklist
- Chat page (`/chat`) still supports standard send/retry and stream/non-stream paths.
- Admin chat page (`/admin/chat`) still loads admin session config and can run chat/image/video flows.
- Register flow intentionally unchanged in this phase set.
- Cloudflare critical files intentionally untouched:
  - `README.cloudflare.md`
  - `wrangler.toml`
  - `app/static/_worker.js`
  - `.github/workflows/cloudflare-workers.yml`

## NSFW Cross-Page Flow Validation Notes
- NSFW-like upstream error text now triggers a visible hint card on chat pages.
- User can jump from chat/admin-chat to token management via `nsfw_refresh=all` query parameters.
- Token page detects that query and asks for confirmation before running full NSFW refresh.
- Query parameters are cleared after decision to avoid repeated auto-prompts.

## Residual Risk
- Automated regression test suite was not executed because `pytest` is unavailable in this environment.
- Validation here is based on code-path inspection and targeted behavior review.
