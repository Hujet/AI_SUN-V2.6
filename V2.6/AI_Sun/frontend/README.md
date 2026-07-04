# Frontend for DEPLOYMENT.md

Simple static frontend that renders `DEPLOYMENT.md` client-side using `marked` and `highlight.js`.

Usage:

- Serve the repository root with a static server (so `DEPLOYMENT.md` is reachable at `../DEPLOYMENT.md` from this folder).

Example (Python 3):

```bash
cd "d:\天文课题\AI_SUN\AI_Sun 2.0\AI_Sun"
python -m http.server 8001
# then open http://localhost:8001/frontend/deployment.html
```
