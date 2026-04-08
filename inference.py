from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from openai import OpenAI

from dashboard_backend import MissionCommander
from env import FireDroneSwarmEnv

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-20b")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")


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
    fallback = {
        "priority": "contain_fire_then_rescue",
        "risk": "interior_fire_and_single_civilian",
        "note": "proxy_call_not_attempted",
    }
    prompt = {
        "sensor_alert": observation.get("sensor_alert"),
        "drone_count": len(observation.get("drones", [])),
        "difficulty": observation.get("scenario", "Medium"),
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


def run_episode(scenario: str = "Medium", max_steps: int = 80) -> None:
    total_reward = 0.0
    final_info: Dict[str, Any] = {
        "graders": {
            "task1_scout_map": 0.01,
            "task2_containment": 0.01,
            "task3_coordinated_rescue": 0.01,
        },
        "fire_tiles_remaining": 0,
        "civilian_reached_exit": False,
    }
    llm_brief = {
        "priority": "contain_fire_then_rescue",
        "risk": "interior_fire_and_single_civilian",
        "note": "proxy_call_not_attempted",
    }
    last_step = 0
    done = False
    start_emitted = False

    try:
        env = FireDroneSwarmEnv(num_drones=6)
        observation = env.reset(scenario)
        commander = MissionCommander()
        final_info["fire_tiles_remaining"] = len(env.fire_intensity)
        llm_brief = request_llm_brief(observation)

        emit(
            "[START]",
            {
                "scenario": scenario,
                "api_base_url": API_BASE_URL,
                "model_name": MODEL_NAME,
                "llm_brief": llm_brief,
                "max_steps": max_steps,
            },
        )
        start_emitted = True

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

        end_payload = {
            "status": "success" if done else "max_steps_reached",
            "total_reward": round(total_reward, 3),
            "steps": last_step,
            "task1_scout_map": round(final_info["graders"]["task1_scout_map"], 3),
            "task2_containment": round(final_info["graders"]["task2_containment"], 3),
            "task3_coordinated_rescue": round(final_info["graders"]["task3_coordinated_rescue"], 3),
            "fire_tiles_remaining": final_info["fire_tiles_remaining"],
            "civilian_reached_exit": final_info["civilian_reached_exit"],
        }
    except Exception as exc:  # pragma: no cover - final validator safeguard
        if not start_emitted:
            emit(
                "[START]",
                {
                    "scenario": scenario,
                    "api_base_url": API_BASE_URL,
                    "model_name": MODEL_NAME,
                    "llm_brief": llm_brief,
                    "max_steps": max_steps,
                },
            )
        end_payload = {
            "status": "error",
            "total_reward": round(total_reward, 3),
            "steps": last_step,
            "task1_scout_map": round(final_info["graders"]["task1_scout_map"], 3),
            "task2_containment": round(final_info["graders"]["task2_containment"], 3),
            "task3_coordinated_rescue": round(final_info["graders"]["task3_coordinated_rescue"], 3),
            "fire_tiles_remaining": final_info["fire_tiles_remaining"],
            "civilian_reached_exit": final_info["civilian_reached_exit"],
            "error_type": type(exc).__name__,
        }

    emit("[END]", end_payload)


if __name__ == "__main__":
    run_episode()
