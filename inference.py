from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI

from dashboard_backend import MissionCommander
from env import FireDroneSwarmEnv

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-20b")
TASK_NAME = os.getenv("MY_ENV_V4_TASK", "all")

TASK_RUNS: List[Dict[str, Any]] = [
    {
        "task": "scout_and_map",
        "difficulty": "easy",
        "scenario": "Easy",
        "grader_key": "task1_scout_map",
        "max_steps": 60,
    },
    {
        "task": "fire_containment",
        "difficulty": "medium",
        "scenario": "Medium",
        "grader_key": "task2_containment",
        "max_steps": 80,
    },
    {
        "task": "coordinated_rescue",
        "difficulty": "hard",
        "scenario": "Hard",
        "grader_key": "task3_coordinated_rescue",
        "max_steps": 100,
    },
]


def clamp_score(score: float) -> float:
    return float(round(max(0.01, min(0.99, float(score))), 3))


def format_token(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".") if "." in f"{value:.3f}" else f"{value:.3f}"
    if isinstance(value, (list, tuple)):
        if not value:
            return "none"
        return "|".join(format_token(item) for item in value)
    return str(value).replace(" ", "_")


def emit_fields(tag: str, **fields: Any) -> None:
    payload = " ".join(f"{key}={format_token(value)}" for key, value in fields.items())
    print(f"{tag} {payload}".rstrip(), flush=True)


def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def request_llm_brief(observation: Dict[str, Any], task_run: Dict[str, Any]) -> Dict[str, Any]:
    fallback = {
        "priority": "contain_fire_then_rescue",
        "risk": "interior_fire_and_single_civilian",
        "note": "proxy_call_not_attempted",
    }
    prompt = {
        "task": task_run["task"],
        "difficulty": task_run["difficulty"],
        "scenario": task_run["scenario"],
        "sensor_alert": observation.get("sensor_alert"),
        "drone_count": len(observation.get("drones", [])),
        "instruction": "Return compact JSON with keys priority, risk, note. Respond only with valid JSON.",
    }

    if not API_BASE_URL:
        return {
            "priority": fallback["priority"],
            "risk": fallback["risk"],
            "note": "proxy_call_skipped_missing_api_base_url",
        }

    if not API_KEY:
        return {
            "priority": fallback["priority"],
            "risk": fallback["risk"],
            "note": "proxy_call_skipped_missing_api_key",
        }

    try:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY,
        )
    except Exception as exc:
        return {
            "priority": fallback["priority"],
            "risk": fallback["risk"],
            "note": f"proxy_client_init_failed:{type(exc).__name__}",
        }

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=120,
            messages=[
                {"role": "system", "content": "You are an emergency planning assistant. Respond only with valid JSON."},
                {"role": "user", "content": json.dumps(prompt)},
            ],
        )
        content = response.choices[0].message.content or ""
    except Exception as exc:
        return {
            "priority": fallback["priority"],
            "risk": fallback["risk"],
            "note": f"proxy_call_failed:{type(exc).__name__}",
        }

    parsed = safe_json_loads(content)
    if not parsed:
        return {
            "priority": fallback["priority"],
            "risk": fallback["risk"],
            "note": "proxy_call_completed_invalid_json",
        }

    return {
        "priority": str(parsed.get("priority", fallback["priority"])),
        "risk": str(parsed.get("risk", fallback["risk"])),
        "note": str(parsed.get("note", "proxy_call_completed")),
    }


def selected_task_runs() -> List[Dict[str, Any]]:
    requested = TASK_NAME.strip().lower()
    if not requested or requested == "all":
        return list(TASK_RUNS)

    requested_parts = {part.strip().lower() for part in requested.split(",") if part.strip()}
    matched = [
        task_run
        for task_run in TASK_RUNS
        if task_run["task"].lower() in requested_parts or task_run["difficulty"].lower() in requested_parts
    ]
    return matched or list(TASK_RUNS)


def default_final_info() -> Dict[str, Any]:
    return {
        "graders": {
            "task1_scout_map": 0.01,
            "task2_containment": 0.01,
            "task3_coordinated_rescue": 0.01,
        },
        "fire_tiles_remaining": 0,
        "civilian_reached_exit": False,
    }


def run_single_task(task_run: Dict[str, Any]) -> None:
    total_reward = 0.0
    final_info = default_final_info()
    llm_brief = {
        "priority": "contain_fire_then_rescue",
        "risk": "interior_fire_and_single_civilian",
        "note": "proxy_call_not_attempted",
    }
    last_step = 0
    done = False
    status = "error"

    try:
        env = FireDroneSwarmEnv(num_drones=6)
        observation = env.reset(task_run["scenario"])
        commander = MissionCommander()
        final_info["fire_tiles_remaining"] = len(env.fire_intensity)
        llm_brief = request_llm_brief(observation, task_run)

        emit_fields(
            "[START]",
            task=task_run["task"],
            difficulty=task_run["difficulty"],
            scenario=task_run["scenario"],
            api_base_url=API_BASE_URL,
            model_name=MODEL_NAME,
            max_steps=task_run["max_steps"],
            llm_note=llm_brief["note"],
        )

        for step_index in range(1, int(task_run["max_steps"]) + 1):
            plan = commander.act(env, observation)
            observation, reward, done, info = env.step({"commands": plan["commands"]})
            total_reward += reward
            final_info = info
            last_step = step_index

            emit_fields(
                "[STEP]",
                task=task_run["task"],
                difficulty=task_run["difficulty"],
                step=step_index,
                reward=round(float(reward), 3),
                done=bool(done),
                task1_scout_map=clamp_score(info["graders"]["task1_scout_map"]),
                task2_containment=clamp_score(info["graders"]["task2_containment"]),
                task3_coordinated_rescue=clamp_score(info["graders"]["task3_coordinated_rescue"]),
                fire_tiles_remaining=int(info["fire_tiles_remaining"]),
                civilian_reached_exit=bool(info["civilian_reached_exit"]),
                commands=plan["commands"],
            )

            if done:
                status = "success"
                break

        if not done:
            status = "max_steps_reached"
    except Exception as exc:  # pragma: no cover - validator safety
        status = f"error:{type(exc).__name__}"

    final_score = clamp_score(final_info["graders"].get(task_run["grader_key"], 0.01))
    emit_fields(
        "[END]",
        task=task_run["task"],
        difficulty=task_run["difficulty"],
        score=final_score,
        steps=last_step,
        status=status,
        total_reward=round(float(total_reward), 3),
    )


def run_all_tasks(task_runs: Iterable[Dict[str, Any]]) -> None:
    for task_run in task_runs:
        run_single_task(task_run)


if __name__ == "__main__":
    run_all_tasks(selected_task_runs())
