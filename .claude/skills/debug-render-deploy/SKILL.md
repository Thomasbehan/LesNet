---
name: debug-render-deploy
description: Diagnose and fix the LesNet live demo deployment on Render (build failures, cold-start timeouts, 5xx on /predict). Use when the demo at lesnet.onrender.com is broken, a Render build fails, or deployment dependencies/Dockerfile need changing.
---

# Debugging the LesNet Render deployment

The live demo (`lesnet.onrender.com`) runs the **Dockerfile** at the repo root and serves
`production.ini` via `pserve`. Render access is available through the project MCP server
(`.mcp.json` → `render`); run `/mcp` and authenticate if the tools aren't connected.

## Get the real evidence first

Do not guess from the code alone — pull the actual logs:

1. **Build logs** — did `pip install -e .` or the model `curl` fail? Build failures are usually
   memory/timeout (TensorFlow 2.21 is a very heavy base dependency) or a dead asset URL.
2. **Runtime logs** — a successful build with a failing app is almost always a missing dependency
   or a missing model artifact, and shows up on the first `/predict` rather than at boot.
3. **The distinction matters.** "Build failed" and "app 500s" have completely different causes;
   establish which one you have before changing anything.

## The failure modes seen so far

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError` on first `/predict`, boots fine | A serving dependency declared only in the optional `[jepa]` extra. The Dockerfile runs `pip install -e .` = **base requires only** | Add it to `requires` in `setup.py`, not to an extra |
| First request after a cold start times out | Model downloaded at runtime (`_ensure_jepa_model` pulls ~318 MB) | Bake the model into the image at build time |
| Build OOM / timeout | `tensorflow==2.21.0` in base requires; the demo path no longer needs it | Consider a minimal `[serve]` extra — but confirm from the build log first |
| 503 `model_unavailable` | Artifacts missing at `LESNET_JEPA_HOME` (default `models/jepa`) | Check the Dockerfile fetch step actually produced `models/jepa/<variant>/jepa_config.json` |

## Which predictor the app loads

`lesnet/views/api.py` → `_get_predictor()`, in precedence order:

1. `LESNET_JEPA_ARTIFACTS` set → load that directory directly
2. otherwise (default) → JEPA family, variant `LESNET_JEPA_VARIANT` (default `medium`), from
   `LESNET_JEPA_HOME` (default `models/jepa`), self-healing via `ModelConfig.JEPA_URLS`
3. `LESNET_USE_TF=1` → the legacy TensorFlow `TriagePredictor`

The JEPA path imports **onnxruntime**, never torch or TensorFlow.

## Always verify a Dockerfile change locally before pushing

Run the exact command from the Dockerfile, then load the predictor the way the container does —
with no environment overrides:

```bash
mkdir -p /tmp/dc/models/jepa
curl -fsSL <release-url>/lesnet-jepa-medium.tar.gz | tar xz -C /tmp/dc/models/jepa
test -f /tmp/dc/models/jepa/medium/jepa_config.json
LESNET_JEPA_HOME=/tmp/dc/models/jepa python -c "
import lesnet.views.api as api; print(api._get_predictor())"
```

A green CI run does **not** prove the deployment works: CI never builds the Dockerfile and never
installs base-only requires. That gap is exactly how the missing `onnxruntime` reached production.
