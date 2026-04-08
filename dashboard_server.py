from __future__ import annotations

import os

import uvicorn


if __name__ == "__main__":
    host = os.getenv("GREXO_HOST", "127.0.0.1")
    port = int(os.getenv("PORT", os.getenv("GREXO_PORT", "8000")))
    uvicorn.run("app:app", host=host, port=port, reload=False)
