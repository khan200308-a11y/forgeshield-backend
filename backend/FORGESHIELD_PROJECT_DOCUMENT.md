# ForgeShield Project Note

This repository started as a hackathon project and has since evolved beyond the original backend-only / Claude-centric design.

## Current State

ForgeShield now includes:

- a built-in frontend served by the Express backend
- a Python detector that can operate without Claude
- CPU-friendly training and inference defaults
- cross-validated LightGBM fusion
- composite evidence-map visualization
- suspicious crops and field-level suspicious regions
- optional offline local LLM semantic audit

## Current Analysis Flow

```text
Upload
  -> Express backend
     -> PDF/image preprocessing
     -> Claude analysis (optional)
     -> Python detector
     -> merged response
  -> frontend renders verdict, flags, dashboard, crops, and reliability
```

## Why This File Exists

Older project notes in the repo described earlier experiments and older model choices. This file is now just a lightweight pointer so readers do not rely on outdated architectural assumptions.

For actual setup and usage, use:

- [README.md](../README.md)
- [backend/README.md](README.md)
