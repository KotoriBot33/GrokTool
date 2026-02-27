# CODEX Serial Plan

Execution mode: strict serial only.
Gate rule: complete and validate current phase before starting the next one.
Safety rule: do not break register flow or Cloudflare one-click deploy files (`README.cloudflare.md`, `wrangler.toml`, `app/static/_worker.js`, `.github/workflows/cloudflare-workers.yml`).

## Phase Order

1. [x] P0 Baseline Protection
- Lock deployment-sensitive files and registration-critical behavior.
- Confirm prior baseline commits are preserved: `50c5104` (responses bridge), `4797262` (chat compatibility params).

2. [x] P3 WS Waterfall Enhancement
- Apply minimal safe enhancement to imagine WS waterfall behavior.
- Keep API compatibility and preserve existing stop/ping/auth behavior.
- Add/adjust focused tests for WS waterfall behavior.

3. [x] P4 Image Edit Workbench Alignment
- Align image edit workspace flow and parameters with current API behavior.

4. [x] P5 Video Page Enhancement + Stitching
- Enhance video page UX and implement stitching-related flow alignment.

5. [x] P6 NSFW Cross-Page Full Flow
- Complete NSFW-related end-to-end behavior across admin and user pages.

6. [x] P7 Regression and Docs
- Run regression checks and update docs/changelog for completed phases.
