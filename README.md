---
title: GREXO Fire Rescue API
emoji: 🚁
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# GREXO Fire Rescue API

GREXO is a FastAPI-based Fire Rescue Drone AI simulator prepared for Hugging Face Docker Spaces and OpenEnvx-style submission.

## What the Space exposes

- Dashboard UI: `/`
- Dashboard alias: `/dashboard`
- Swagger UI: `/docs`
- OpenAPI schema: `/openapi.json`
- Core environment reset endpoint: `POST /reset`
- Core environment state endpoint: `GET /state`
- Core environment step endpoint: `POST /step`

## Difficulty support

The simulator supports three real difficulty levels connected to environment reward logic:

- `Easy`: fewer initial fires, higher tolerance, more forgiving rewards
- `Medium`: balanced mission difficulty and reward pressure
- `Hard`: more fire spread, lower tolerance, stricter penalties

Difficulty can be passed to `POST /reset` using either:

- `difficulty`
- `scenario` (backward-compatible)

## Core API examples

### `POST /reset`

```json
{
  "difficulty": "Medium"
}
