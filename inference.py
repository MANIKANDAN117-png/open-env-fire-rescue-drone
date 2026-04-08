from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from dashboard_backend import MissionCommander
from env import FireDroneSwarmEnv


def emit(tag: str, payload: Dict[str, Any]) -> None:
    print(f"{tag} {json.dumps(payload, separators=(',', ':'), ensure_ascii=False)}", flush=True)


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


def request_llm_brief(observation: Dict[str, Any]) -> Dict[str, Any]:
    api_base_url = os.getenv("API_BASE_URL", "").strip()
    model_name = os.getenv("MODEL_NAME", "").strip()
    hf_token = os.getenv("HF_TOKEN", "").strip()

    fallback = {
        "priority": "contain_fire_then_rescue",
        "risk": "interior_fire_and_single_civilian",
        "note": "deterministic_baseline_active",
    }

    if not api_base_url or not model_name or not hf_token:
        return fallback

    try:
        from openai import OpenAI
    except Exception:
        return fallback

    client = OpenAI(base_url=api_base_url, api_key=hf_token)
    prompt = {
        "sensor_alert": observation["sensor_alert"],
        "drone_count": len(observation["drones"]),
        "difficulty": observation.get("scenario", "Medium"),
        "instruction": "Return compact JSON with keys priority, risk, note. Respond only with valid JSON.",
    }

    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0,
            max_tokens=120,
            messages=[
                {"role": "system", "content": "You are an emergency planning assistant. Respond only with valid JSON."},
                {"role": "user", "content": json.dumps(prompt)},
            ],
        )
        content = response.choices[0].message.content or ""
        parsed = safe_json_loads(content)
        if not parsed:
            return fallback
        return {
            "priority": str(parsed.get("priority", fallback["priority"])),
            "risk": str(parsed.get("risk", fallback["risk"])),
            "note": str(parsed.get("note", fallback["note"])),
        }
    except Exception as exc:  # pragma: no cover - runtime fallback
        return {
            "priority": fallback["priority"],
            "risk": fallback["risk"],
            "note": f"llm_unavailable:{type(exc).__name__}",
        }


def run_episode(scenario: str = "Medium", max_steps: int = 80) -> None:
    env = FireDroneSwarmEnv(num_drones=6)
    observation = env.reset(scenario)
    commander = MissionCommander()
    llm_brief = request_llm_brief(observation)

    emit(
        "[START]",
        {
            "scenario": scenario,
            "api_base_url": os.getenv("API_BASE_URL", ""),
            "model_name": os.getenv("MODEL_NAME", ""),
            "llm_brief": llm_brief,
            "max_steps": max_steps,
        },
    )

    total_reward = 0.0
    final_info: Dict[str, Any] = {
        "graders": {
            "task1_scout_map": 0.0,
            "task2_containment": 0.0,
            "task3_coordinated_rescue": 0.0,
        },
        "fire_tiles_remaining": len(env.fire_intensity),
        "civilian_reached_exit": False,
    }
    done = False
    last_step = 0

    for step_index in range(1, max_steps + 1):
        plan = commander.act(env, observation)
        observation, reward, done, info = env.step({"commands": plan["commands"]})
        total_reward += reward
        final_info = info
        last_step = step_index
        emit(
            "[STEP]",
            {
                "step": step_index,
                "commands": plan["commands"],
                "reward": round(reward, 3),
                "done": done,
                "task1_scout_map": round(info["graders"]["task1_scout_map"], 3),
                "task2_containment": round(info["graders"]["task2_containment"], 3),
                "task3_coordinated_rescue": round(info["graders"]["task3_coordinated_rescue"], 3),
                "fire_tiles_remaining": info["fire_tiles_remaining"],
                "civilian_reached_exit": info["civilian_reached_exit"],
            },
        )
        if done:
            break

    emit(
        "[END]",
        {
            "status": "success" if done else "max_steps_reached",
            "total_reward": round(total_reward, 3),
            "steps": last_step,
            "task1_scout_map": round(final_info["graders"]["task1_scout_map"], 3),
            "task2_containment": round(final_info["graders"]["task2_containment"], 3),
            "task3_coordinated_rescue": round(final_info["graders"]["task3_coordinated_rescue"], 3),
            "fire_tiles_remaining": final_info["fire_tiles_remaining"],
            "civilian_reached_exit": final_info["civilian_reached_exit"],
        },
    )


if __name__ == "__main__":
    run_episode()
