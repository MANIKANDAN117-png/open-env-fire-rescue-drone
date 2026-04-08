from __future__ import annotations

import argparse
import os

import uvicorn

from app import app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GREXO Fire Rescue API server.")
    parser.add_argument("--host", default=os.getenv("GREXO_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", os.getenv("GREXO_PORT", "7860"))))
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
