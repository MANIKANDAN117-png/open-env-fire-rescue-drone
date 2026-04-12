from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from dashboard_backend import MissionCommander
from env import FireDroneSwarmEnv

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-20b")
HF_TOKEN = os.getenv("HF_TOKEN")

BENCHMARK_NAME = "grexo-fire-rescue"
DEFAULT_TASK_NAME = "scout_and_map"
TASK_SCENARIOS: Dict[str, str] = {
    "scout_and_map": "Easy",
    "fire_containment": "Medium",
    "coordinated_rescue": "Hard",
}
MAX_STEPS = 80

try:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )
except Exception:
    client = None


def format_bool(value: bool) -> str:
    return "true" if value else "false"


def format_reward(value: float) -> str:
    return f"{float(value):.2f}"


def format_rewards(values: List[float]) -> str:
    return ",".join(format_reward(value) for value in values)


def format_action(commands: List[str]) -> str:
    if not commands:
        return "noop"
    return "|".join(str(command).replace(" ", "_") for command in commands)


def resolve_task_name() -> str:
    requested = os.getenv("MY_ENV_V4_TASK", DEFAULT_TASK_NAME).strip()
    return requested if requested in TASK_SCENARIOS else DEFAULT_TASK_NAME


def get_last_action_error(env: FireDroneSwarmEnv, info: Optional[Dict[str, Any]]) -> str:
    raw_error = getattr(env, "last_action_error", None)
    if raw_error is None and isinstance(info, dict):
        raw_error = info.get("last_action_error")
    if raw_error in (None, "", []):
        return "null"
    return str(raw_error).replace("\n", " ").strip()


def warmup_proxy(task_name: str) -> None:
    if client is None or not HF_TOKEN:
        return
    try:
        client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=8,
            messages=[
                {"role": "system", "content": "Reply with READY."},
                {"role": "user", "content": f"task={task_name} env={BENCHMARK_NAME}"},
            ],
        )
    except Exception:
        pass


def main() -> None:
    task_name = resolve_task_name()
    scenario = TASK_SCENARIOS.get(task_name, "Easy")
    rewards: List[float] = []
    steps = 0
    success = False
    env: Optional[FireDroneSwarmEnv] = None

    print(f"[START] task={task_name} env={BENCHMARK_NAME} model={MODEL_NAME}", flush=True)

    try:
        warmup_proxy(task_name)

        env = FireDroneSwarmEnv(num_drones=6)
        observation = env.reset(scenario)
        commander = MissionCommander()

        for step_index in range(1, MAX_STEPS + 1):
            action_commands: List[str] = []
            reward = 0.0
            done = False
            info: Optional[Dict[str, Any]] = None
            error = "null"

            try:
                plan = commander.act(env, observation)
                action_commands = list(plan.get("commands", []))
                observation, reward, done, info = env.step({"commands": action_commands})
                error = get_last_action_error(env, info)
            except Exception as exc:
                done = True
                error = f"{type(exc).__name__}:{str(exc).strip()}"

            rewards.append(float(reward))
            steps = step_index

            print(
                f"[STEP] step={step_index} action={format_action(action_commands)} "
                f"reward={format_reward(reward)} done={format_bool(done)} error={error}",
                flush=True,
            )

            if done:
                success = error == "null"
                break

    except Exception:
        success = False
    finally:
        if env is not None:
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass

        print(
            f"[END] success={format_bool(success)} steps={steps} rewards={format_rewards(rewards)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
