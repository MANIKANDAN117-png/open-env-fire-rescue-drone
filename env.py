from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

try:
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - lightweight fallback for minimal environments
    class BaseModel:
        def __init__(self, **data: Any) -> None:
            for key in getattr(self, "__annotations__", {}):
                if key in data:
                    setattr(self, key, data[key])
                elif hasattr(self.__class__, key):
                    default = getattr(self.__class__, key)
                    setattr(self, key, default() if callable(default) else default)
                else:
                    raise TypeError(f"Missing required field: {key}")

        def model_dump(self) -> Dict[str, Any]:
            return {key: getattr(self, key) for key in getattr(self, "__annotations__", {})}

    def Field(default_factory: Any) -> Any:
        return default_factory


Coordinate = Tuple[int, int]
Direction = str


def manhattan(a: Coordinate, b: Coordinate) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class Observation(BaseModel):
    city_grid: List[List[int]]
    drones: List[Dict[str, Any]]
    sensor_alert: Coordinate
    scenario: str
    difficulty_profile: Dict[str, Any]


class Action(BaseModel):
    commands: List[str] = Field(default_factory=list)


class FireDroneSwarmEnv:
    EMPTY = 0
    WALL = 1
    FIRE = 2
    CIVILIAN = 3
    EXIT = 4

    DIRECTIONS: Dict[Direction, Coordinate] = {
        "up": (0, -1),
        "down": (0, 1),
        "left": (-1, 0),
        "right": (1, 0),
    }

    SCENARIO_PROFILES: Dict[str, Dict[str, Any]] = {
        "Easy": {
            "fire_map": {
                (8, 8): 1,
                (9, 7): 1,
                (10, 7): 1,
            },
            "civilian_pos": (10, 8),
            "step_penalty": 0.5,
            "idle_battery_cost": 0.10,
            "action_battery_cost": 0.75,
            "spray_power": 2,
            "drop_power": 3,
            "scan_reward": 14.0,
            "spray_reward": 6.0,
            "extinguish_reward": 18.0,
            "path_reward": 20.0,
            "rescue_progress_reward": 8.0,
            "rescue_reward": 120.0,
            "danger_penalty": 0.0,
            "battery_start": 100,
            "payload_start": 10,
            "max_steps": 220,
            "extra_walls": [],
        },
        "Medium": {
            "fire_map": {
                (9, 8): 3,
                (8, 8): 2,
                (9, 7): 2,
                (10, 7): 1,
                (8, 7): 1,
                (10, 9): 1,
            },
            "civilian_pos": (10, 8),
            "step_penalty": 1.0,
            "idle_battery_cost": 0.20,
            "action_battery_cost": 1.00,
            "spray_power": 1,
            "drop_power": 2,
            "scan_reward": 10.0,
            "spray_reward": 4.0,
            "extinguish_reward": 15.0,
            "path_reward": 12.0,
            "rescue_progress_reward": 6.0,
            "rescue_reward": 100.0,
            "danger_penalty": 2.0,
            "battery_start": 100,
            "payload_start": 8,
            "max_steps": 180,
            "extra_walls": [],
        },
        "Hard": {
            "fire_map": {
                (8, 7): 2,
                (9, 7): 2,
                (10, 7): 2,
                (8, 8): 2,
                (9, 8): 3,
                (10, 8): 2,
                (11, 8): 2,
                (9, 9): 2,
                (10, 9): 2,
            },
            "civilian_pos": (11, 9),
            "step_penalty": 1.6,
            "idle_battery_cost": 0.35,
            "action_battery_cost": 1.25,
            "spray_power": 1,
            "drop_power": 2,
            "scan_reward": 6.0,
            "spray_reward": 3.0,
            "extinguish_reward": 12.0,
            "path_reward": 8.0,
            "rescue_progress_reward": 4.0,
            "rescue_reward": 82.0,
            "danger_penalty": 4.0,
            "battery_start": 92,
            "payload_start": 7,
            "max_steps": 150,
            "extra_walls": [],
        },
    }

    TASK_DIFFICULTIES = {
        "task1_scout_map": "easy",
        "task2_containment": "medium",
        "task3_coordinated_rescue": "hard",
    }
    APPROVE_BUG_PENALTY = {"easy": 0.40, "medium": 0.50, "hard": 0.60}
    MISSED_BUG_PENALTY = {"easy": 0.04, "medium": 0.045, "hard": 0.05}
    FALSE_POSITIVE_PENALTY = 0.02
    CONSISTENCY_BONUS = {"easy": 0.05, "medium": 0.10, "hard": 0.15}
    EXPLANATION_BONUS = {"easy": 0.0, "medium": 0.01, "hard": 0.04}

    def __init__(self, num_drones: int = 6, max_steps: int = 180) -> None:
        self.width = 15
        self.height = 15
        self.num_drones = num_drones
        self.default_max_steps = max_steps
        self.max_steps = max_steps

        self.base_station: Coordinate = (1, 1)
        self.sensor_alert: Coordinate = (9, 8)
        self.building_bounds: Tuple[int, int, int, int] = (5, 3, 12, 11)
        self.entry_point: Coordinate = (5, 7)
        self.exit_pos: Coordinate = (5, 7)

        self.city_grid_true: List[List[int]] = []
        self.visible_mask: List[List[bool]] = []
        self.drones: Dict[str, Dict[str, Any]] = {}
        self.fire_intensity: Dict[Coordinate, int] = {}
        self.civilian_pos: Coordinate = (10, 8)
        self.projected_paths: Set[Coordinate] = set()

        self.step_count = 0
        self.initial_fire_tiles = 0
        self.civilian_discovered = False
        self.civilian_discovered_with_power = False
        self.civilian_reached_exit = False
        self.fire_blocking_civilian_cleared = False
        self.fire_suppression_drone_ids: Set[str] = set()
        self.path_projection_drone_ids: Set[str] = set()
        self.task_step_history: Dict[str, List[Dict[str, Any]]] = {}
        self.last_action_error: Optional[str] = None
        self.scenario = "Medium"
        self.profile = deepcopy(self.SCENARIO_PROFILES[self.scenario])

        self.reset("Medium")

    def reset(self, scenario: str = "Medium") -> Dict[str, Any]:
        resolved = scenario if scenario in self.SCENARIO_PROFILES else "Medium"
        self.scenario = resolved
        self.profile = deepcopy(self.SCENARIO_PROFILES[self.scenario])
        self.max_steps = int(self.profile.get("max_steps", self.default_max_steps))

        self.step_count = 0
        self.projected_paths = set()
        self.fire_suppression_drone_ids = set()
        self.path_projection_drone_ids = set()
        self.civilian_discovered = False
        self.civilian_discovered_with_power = False
        self.civilian_reached_exit = False
        self.fire_blocking_civilian_cleared = False
        self.last_action_error = None
        self.task_step_history = {
            "task1_scout_map": [],
            "task2_containment": [],
            "task3_coordinated_rescue": [],
        }

        self.city_grid_true = [[self.EMPTY for _ in range(self.width)] for _ in range(self.height)]
        self.visible_mask = [[False for _ in range(self.width)] for _ in range(self.height)]
        self.civilian_pos = tuple(self.profile["civilian_pos"])

        self._build_city_and_building()
        self._place_entities()
        self._init_drones()

        self.initial_fire_tiles = len(self.fire_intensity)
        self._reveal_around_point(self.base_station, 3)
        self._refresh_visibility()
        return self.state()

    def close(self) -> None:
        return None

    def state(self) -> Dict[str, Any]:
        observation = Observation(
            city_grid=self._observed_city_grid(),
            drones=[self._drone_to_dict(drone) for drone in self.drones.values()],
            sensor_alert=self.sensor_alert,
            scenario=self.scenario,
            difficulty_profile=self._difficulty_profile_payload(),
        )
        return observation.model_dump()

    def step(
        self,
        action: Action | Dict[str, Any] | List[str],
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self.step_count += 1
        self.last_action_error = None
        reward = -float(self.profile["step_penalty"])
        parsed = self._normalize_action(action)
        commands_by_drone = self._parse_commands(parsed.commands)
        resolved_command_names = self._resolved_command_names(commands_by_drone)
        before = self._capture_snapshot()
        info: Dict[str, Any] = {"events": []}

        for drone_id, drone in self.drones.items():
            command_name, args = commands_by_drone.get(drone_id, ("noop", []))
            delta_reward, events = self._apply_drone_command(drone, command_name, args)
            reward += delta_reward
            info["events"].extend(events)

        if any(self._is_fire_adjacent_to_civilian(pos) for pos in self.fire_intensity):
            reward -= float(self.profile["danger_penalty"])

        civilian_reward, civilian_events = self._move_civilian_if_possible()
        reward += civilian_reward
        info["events"].extend(civilian_events)

        self._refresh_visibility()
        after = self._capture_snapshot()
        self._record_task_rewards(resolved_command_names, before, after)

        scout_score = self.grade_task1_scout_and_map()
        containment_score = self.grade_task2_containment()
        rescue_score = self.grade_task3_coordinated_rescue()
        info["graders"] = {
            "scout_and_map": scout_score,
            "fire_containment": containment_score,
            "coordinated_rescue": rescue_score,
            "task1_scout_map": scout_score,
            "task2_containment": containment_score,
            "task3_coordinated_rescue": rescue_score,
        }
        info["fire_tiles_remaining"] = len(self.fire_intensity)
        info["civilian_position"] = self.civilian_pos
        info["civilian_reached_exit"] = self.civilian_reached_exit
        info["last_action_error"] = self.last_action_error
        info["coordination"] = {
            "blocking_fire_cleared": self.fire_blocking_civilian_cleared,
            "suppression_drones": sorted(self.fire_suppression_drone_ids),
            "path_drones": sorted(self.path_projection_drone_ids),
        }

        done = self._is_done()
        return self.state(), float(round(reward, 3)), done, info

    def grade_task1_scout_and_map(self) -> float:
        return self._aggregate_task_score("task1_scout_map")

    def grade_task2_containment(self) -> float:
        return self._aggregate_task_score("task2_containment")

    def grade_task3_coordinated_rescue(self) -> float:
        return self._aggregate_task_score("task3_coordinated_rescue")

    def _aggregate_task_score(self, task_key: str) -> float:
        history = self.task_step_history.get(task_key, [])
        if not history:
            return self._bounded_score(0.01)

        difficulty = self.TASK_DIFFICULTIES[task_key]
        step_scores = [float(entry["score"]) for entry in history]
        base_score = sum(step_scores) / len(step_scores)

        approve_bug_count = sum(1 for entry in history if entry["category"] == "catastrophic")
        missed_bug_count = sum(1 for entry in history if entry["category"] == "missed_bug")
        false_positive_count = sum(1 for entry in history if entry["category"] == "false_positive")
        correct_steps = sum(1 for entry in history if float(entry["score"]) >= 0.70)
        perfect_steps = sum(1 for entry in history if abs(float(entry["score"]) - 0.90) < 1e-9)

        score = base_score
        score -= min(0.45, approve_bug_count * self.APPROVE_BUG_PENALTY[difficulty])
        score -= min(0.20, missed_bug_count * self.MISSED_BUG_PENALTY[difficulty])
        score -= min(0.10, false_positive_count * self.FALSE_POSITIVE_PENALTY)

        if correct_steps / len(history) >= 0.80:
            score += self.CONSISTENCY_BONUS[difficulty]
        if perfect_steps / len(history) >= 0.80:
            score += self.EXPLANATION_BONUS[difficulty]

        return self._bounded_score(score)

    def _bounded_score(self, raw_score: float) -> float:
        return float(round(max(0.01, min(0.99, raw_score)), 3))

    def _difficulty_profile_payload(self) -> Dict[str, Any]:
        return {
            "difficulty": self.scenario,
            "step_penalty": float(self.profile["step_penalty"]),
            "battery_start": int(self.profile["battery_start"]),
            "payload_start": int(self.profile["payload_start"]),
            "active_fire_tiles": len(self.profile["fire_map"]),
            "rescue_reward": float(self.profile["rescue_reward"]),
            "containment_reward": float(self.profile["extinguish_reward"]),
        }

    def _capture_snapshot(self) -> Dict[str, Any]:
        return {
            "discovered": bool(self.civilian_discovered_with_power or self.civilian_discovered),
            "fire_count": len(self.fire_intensity),
            "fire_positions": list(self.fire_intensity.keys()),
            "fire_near_civilian": any(self._is_fire_adjacent_to_civilian(pos) for pos in self.fire_intensity),
            "civilian_pos": self.civilian_pos,
            "civilian_reached_exit": self.civilian_reached_exit,
            "projected_count": len(self.projected_paths),
            "rescue_near": self._role_near_civilian({"rescue"}),
            "guide_near": self._role_near_civilian({"guide"}),
            "drone_positions": {
                drone_id: (drone["x"], drone["y"])
                for drone_id, drone in self.drones.items()
            },
        }

    def _role_near_civilian(self, roles: Set[str]) -> bool:
        return any(
            drone["role"] in roles
            and manhattan((drone["x"], drone["y"]), self.civilian_pos) <= 1
            for drone in self.drones.values()
        )

    def _average_role_distance(
        self,
        positions: Dict[str, Coordinate],
        roles: Set[str],
        target: Coordinate,
    ) -> float:
        distances = [
            float(manhattan(positions[drone_id], target))
            for drone_id, drone in self.drones.items()
            if drone["role"] in roles and drone_id in positions
        ]
        return sum(distances) / len(distances) if distances else float("inf")

    def _average_fire_distance(
        self,
        positions: Dict[str, Coordinate],
        roles: Set[str],
        fire_positions: Sequence[Coordinate],
    ) -> float:
        if not fire_positions:
            return 0.0
        distances: List[float] = []
        for drone_id, drone in self.drones.items():
            if drone["role"] not in roles or drone_id not in positions:
                continue
            distances.append(
                float(min(manhattan(positions[drone_id], fire_pos) for fire_pos in fire_positions))
            )
        return sum(distances) / len(distances) if distances else float("inf")

    def _record_task_rewards(
        self,
        command_names: List[str],
        before: Dict[str, Any],
        after: Dict[str, Any],
    ) -> None:
        if not before["discovered"]:
            self.task_step_history["task1_scout_map"].append(
                self._evaluate_task1_step(command_names, before, after)
            )
        if before["fire_count"] > 0:
            self.task_step_history["task2_containment"].append(
                self._evaluate_task2_step(command_names, before, after)
            )
        if before["fire_count"] == 0 and not before["civilian_reached_exit"]:
            self.task_step_history["task3_coordinated_rescue"].append(
                self._evaluate_task3_step(command_names, before, after)
            )

    def _evaluate_task1_step(
        self,
        command_names: List[str],
        before: Dict[str, Any],
        after: Dict[str, Any],
    ) -> Dict[str, Any]:
        has_scan = "scan" in command_names
        has_move = "move" in command_names
        has_suppress = any(name in {"spray", "drop_ball"} for name in command_names)
        has_path = "project_path" in command_names

        prev_distance = self._average_role_distance(before["drone_positions"], {"rescue", "guide"}, before["civilian_pos"])
        next_distance = self._average_role_distance(after["drone_positions"], {"rescue", "guide"}, before["civilian_pos"])
        moved_closer = next_distance < prev_distance

        if after["discovered"] and has_scan:
            return {"score": 0.90, "category": "perfect"}
        if after["discovered"]:
            return {"score": 0.88, "category": "near_perfect"}
        if has_scan:
            return {"score": 0.82, "category": "partial"}
        if has_move and moved_closer:
            return {"score": 0.88, "category": "near_perfect"}
        if has_move:
            return {"score": 0.76, "category": "partial"}
        if has_suppress or has_path:
            return {"score": 0.15, "category": "false_positive"}
        if all(name == "noop" for name in command_names):
            return {"score": 0.30, "category": "missed_bug"}
        return {"score": 0.50, "category": "cautious"}

    def _evaluate_task2_step(
        self,
        command_names: List[str],
        before: Dict[str, Any],
        after: Dict[str, Any],
    ) -> Dict[str, Any]:
        has_suppress = any(name in {"spray", "drop_ball"} for name in command_names)
        has_move = "move" in command_names
        fire_delta = before["fire_count"] - after["fire_count"]

        prev_distance = self._average_fire_distance(before["drone_positions"], {"extinguish"}, before["fire_positions"])
        next_distance = self._average_fire_distance(after["drone_positions"], {"extinguish"}, after["fire_positions"])
        moved_closer = next_distance < prev_distance

        if has_suppress and fire_delta > 0:
            if after["fire_count"] == 0 or fire_delta >= 2:
                return {"score": 0.90, "category": "perfect"}
            return {"score": 0.88, "category": "near_perfect"}
        if before["fire_near_civilian"] and not has_suppress and not moved_closer:
            return {"score": 0.10, "category": "catastrophic"}
        if has_suppress:
            return {"score": 0.15, "category": "false_positive"}
        if has_move and moved_closer:
            return {"score": 0.76, "category": "partial"}
        if has_move:
            return {"score": 0.70, "category": "partial"}
        if "scan" in command_names or "project_path" in command_names:
            return {"score": 0.50, "category": "cautious"}
        if all(name == "noop" for name in command_names):
            return {"score": 0.30, "category": "missed_bug"}
        return {"score": 0.30, "category": "missed_bug"}

    def _evaluate_task3_step(
        self,
        command_names: List[str],
        before: Dict[str, Any],
        after: Dict[str, Any],
    ) -> Dict[str, Any]:
        has_move = "move" in command_names
        has_suppress = any(name in {"spray", "drop_ball"} for name in command_names)
        path_growth = after["projected_count"] > before["projected_count"]
        civilian_advanced = after["civilian_pos"] != before["civilian_pos"]
        escort_near = after["rescue_near"] or after["guide_near"]

        prev_distance = self._average_role_distance(before["drone_positions"], {"rescue", "guide"}, before["civilian_pos"])
        next_distance = self._average_role_distance(after["drone_positions"], {"rescue", "guide"}, after["civilian_pos"])
        escort_moved_closer = next_distance < prev_distance

        if after["civilian_reached_exit"]:
            if civilian_advanced and (path_growth or escort_near):
                return {"score": 0.90, "category": "perfect"}
            return {"score": 0.88, "category": "near_perfect"}
        if path_growth and escort_near:
            return {"score": 0.90, "category": "perfect"}
        if civilian_advanced and escort_near:
            return {"score": 0.88, "category": "near_perfect"}
        if path_growth or escort_moved_closer or escort_near:
            return {"score": 0.76, "category": "partial"}
        if has_suppress:
            return {"score": 0.15, "category": "false_positive"}
        if has_move:
            return {"score": 0.70, "category": "partial"}
        if "scan" in command_names:
            return {"score": 0.50, "category": "cautious"}
        if all(name == "noop" for name in command_names):
            return {"score": 0.30, "category": "missed_bug"}
        return {"score": 0.30, "category": "missed_bug"}

    def _resolved_command_names(self, commands_by_drone: Dict[str, Tuple[str, List[str]]]) -> List[str]:
        return [
            commands_by_drone.get(drone_id, ("noop", []))[0]
            for drone_id in sorted(self.drones)
        ]

    def _build_city_and_building(self) -> None:
        for x in range(self.width):
            self.city_grid_true[0][x] = self.WALL
            self.city_grid_true[self.height - 1][x] = self.WALL
        for y in range(self.height):
            self.city_grid_true[y][0] = self.WALL
            self.city_grid_true[y][self.width - 1] = self.WALL

        bx0, by0, bx1, by1 = self.building_bounds
        for x in range(bx0, bx1 + 1):
            self.city_grid_true[by0][x] = self.WALL
            self.city_grid_true[by1][x] = self.WALL
        for y in range(by0, by1 + 1):
            self.city_grid_true[y][bx0] = self.WALL
            self.city_grid_true[y][bx1] = self.WALL

        self.city_grid_true[self.entry_point[1]][self.entry_point[0]] = self.EXIT

        for x in range(7, 11):
            self.city_grid_true[6][x] = self.WALL
        for y in range(7, 10):
            self.city_grid_true[y][9] = self.WALL
        self.city_grid_true[6][9] = self.EMPTY
        self.city_grid_true[8][9] = self.EMPTY

        for wall in self.profile.get("extra_walls", []):
            if wall == self.entry_point or wall == self.civilian_pos:
                continue
            if self._in_bounds(wall):
                self.city_grid_true[wall[1]][wall[0]] = self.WALL

    def _place_entities(self) -> None:
        self.fire_intensity = dict(self.profile["fire_map"])
        self.city_grid_true[self.civilian_pos[1]][self.civilian_pos[0]] = self.CIVILIAN
        for (x, y), _intensity in self.fire_intensity.items():
            if self.city_grid_true[y][x] != self.WALL:
                self.city_grid_true[y][x] = self.FIRE

    def _init_drones(self) -> None:
        roles = [
            "extinguish",
            "rescue",
            "extinguish",
            "guide",
            "extinguish",
            "rescue",
        ]
        positions = [(1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1)]
        battery_start = int(self.profile["battery_start"])
        payload_start = int(self.profile["payload_start"])
        self.drones = {}
        for idx in range(self.num_drones):
            drone_id = f"drone_{idx}"
            x, y = positions[idx % len(positions)]
            self.drones[drone_id] = {
                "id": drone_id,
                "x": x,
                "y": y,
                "battery_level": float(battery_start),
                "payload_capacity": float(payload_start),
                "role": roles[idx % len(roles)],
            }

    def _drone_to_dict(self, drone: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": drone["id"],
            "x": drone["x"],
            "y": drone["y"],
            "battery_level": int(round(drone["battery_level"])),
            "payload_capacity": int(round(drone["payload_capacity"])),
            "role": drone["role"],
        }

    def _observed_city_grid(self) -> List[List[int]]:
        grid: List[List[int]] = []
        for y in range(self.height):
            row: List[int] = []
            for x in range(self.width):
                if self.visible_mask[y][x]:
                    row.append(self.city_grid_true[y][x])
                else:
                    row.append(self.EMPTY)
            grid.append(row)
        return grid

    def _normalize_action(self, action: Action | Dict[str, Any] | List[str]) -> Action:
        if isinstance(action, Action):
            return action
        if isinstance(action, dict):
            return Action(**action)
        if isinstance(action, list):
            return Action(commands=action)
        raise TypeError("Action must be an Action, dict, or list of command strings")

    def _parse_commands(self, commands: List[str]) -> Dict[str, Tuple[str, List[str]]]:
        parsed: Dict[str, Tuple[str, List[str]]] = {}
        for raw in commands:
            raw = raw.strip()
            if not raw.endswith(")") or "(" not in raw:
                self.last_action_error = raw
                continue
            name, remainder = raw.split("(", 1)
            args_text = remainder[:-1]
            args = [item.strip() for item in args_text.split(",") if item.strip()]
            if not args:
                self.last_action_error = raw
                continue
            drone_id = args[0]
            if drone_id not in self.drones:
                self.last_action_error = raw
                continue
            parsed[drone_id] = (name.strip(), args[1:])
        return parsed

    def _apply_drone_command(
        self,
        drone: Dict[str, Any],
        command_name: str,
        args: List[str],
    ) -> Tuple[float, List[str]]:
        events: List[str] = []
        reward = 0.0
        drone_id = drone["id"]

        if drone["battery_level"] <= 0:
            self.last_action_error = self.last_action_error or f"{drone_id}_out_of_battery"
            return -10.0, [f"{drone_id} is out of battery."]

        if command_name == "noop":
            drone["battery_level"] = max(0.0, drone["battery_level"] - float(self.profile["idle_battery_cost"]))
            return reward, events

        drone["battery_level"] = max(0.0, drone["battery_level"] - float(self.profile["action_battery_cost"]))

        if command_name == "move":
            if not args:
                self.last_action_error = self.last_action_error or "move_missing_direction"
                return reward, events
            direction = args[0]
            delta = self.DIRECTIONS.get(direction)
            if not delta:
                self.last_action_error = self.last_action_error or f"invalid_direction_{direction}"
                return reward, events
            nxt = (drone["x"] + delta[0], drone["y"] + delta[1])
            if self._can_enter(nxt):
                drone["x"], drone["y"] = nxt
                events.append(f"{drone_id} moved {direction}.")
            return reward, events

        if command_name == "scan":
            zoom = 2
            if args:
                try:
                    zoom = max(1, min(3, int(args[0])))
                except ValueError:
                    zoom = 2
            self._reveal_around_point((drone["x"], drone["y"]), zoom + 1)
            discovered_now = self.visible_mask[self.civilian_pos[1]][self.civilian_pos[0]] and not self.civilian_discovered
            if discovered_now:
                self.civilian_discovered = True
                self.civilian_discovered_with_power = True
                reward += float(self.profile["scan_reward"])
                events.append(f"{drone_id} discovered the civilian.")
            else:
                events.append(f"{drone_id} scanned with zoom {zoom}.")
            return reward, events

        if command_name == "spray":
            if drone["payload_capacity"] <= 0 or not args:
                self.last_action_error = self.last_action_error or "spray_unavailable"
                return reward, events
            direction = args[0]
            delta = self.DIRECTIONS.get(direction)
            if not delta:
                self.last_action_error = self.last_action_error or f"invalid_direction_{direction}"
                return reward, events
            target = (drone["x"] + delta[0], drone["y"] + delta[1])
            drone["payload_capacity"] = max(0.0, drone["payload_capacity"] - 1.0)
            reward += self._reduce_fire(target, amount=int(self.profile["spray_power"]), drone_id=drone_id)
            if target not in self.fire_intensity:
                events.append(f"{drone_id} sprayed {direction}.")
            return reward, events

        if command_name == "drop_ball":
            if drone["payload_capacity"] <= 0:
                self.last_action_error = self.last_action_error or "drop_ball_unavailable"
                return reward, events
            extinguished_before = len(self.fire_intensity)
            for y in range(drone["y"] - 1, drone["y"] + 2):
                for x in range(drone["x"] - 1, drone["x"] + 2):
                    reward += self._reduce_fire((x, y), amount=int(self.profile["drop_power"]), drone_id=drone_id)
            if extinguished_before == len(self.fire_intensity):
                events.append(f"{drone_id} dropped an extinguish ball.")
            drone["payload_capacity"] = max(0.0, drone["payload_capacity"] - 2.0)
            return reward, events

        if command_name == "project_path":
            if not args:
                self.last_action_error = self.last_action_error or "project_path_missing_direction"
                return reward, events
            direction = args[0]
            delta = self.DIRECTIONS.get(direction)
            if not delta:
                self.last_action_error = self.last_action_error or f"invalid_direction_{direction}"
                return reward, events
            cursor = (drone["x"], drone["y"])
            for _ in range(4):
                cursor = (cursor[0] + delta[0], cursor[1] + delta[1])
                if not self._in_bounds(cursor):
                    break
                tile = self.city_grid_true[cursor[1]][cursor[0]]
                if tile == self.WALL:
                    break
                if cursor in self.fire_intensity:
                    break
                self.projected_paths.add(cursor)
                if cursor == self.exit_pos:
                    break
            self.path_projection_drone_ids.add(drone_id)
            events.append(f"{drone_id} projected a safe path {direction}.")
            return reward, events

        self.last_action_error = self.last_action_error or f"unsupported_command_{command_name}"
        return reward, events

    def _reduce_fire(self, target: Coordinate, amount: int, drone_id: str) -> float:
        if target not in self.fire_intensity:
            return 0.0
        self.fire_intensity[target] = max(0, self.fire_intensity[target] - amount)
        self.fire_suppression_drone_ids.add(drone_id)
        if self.fire_intensity[target] == 0:
            del self.fire_intensity[target]
            if self.city_grid_true[target[1]][target[0]] == self.FIRE:
                self.city_grid_true[target[1]][target[0]] = self.EMPTY
            if self._is_fire_adjacent_to_civilian(target):
                self.fire_blocking_civilian_cleared = True
            return float(self.profile["extinguish_reward"])
        return float(self.profile["spray_reward"])

    def _move_civilian_if_possible(self) -> Tuple[float, List[str]]:
        if self.civilian_reached_exit:
            return 0.0, []

        escort_nearby = any(
            drone["role"] in {"rescue", "guide"}
            and drone["battery_level"] > 0
            and manhattan((drone["x"], drone["y"]), self.civilian_pos) <= 1
            for drone in self.drones.values()
        )
        if not escort_nearby:
            return 0.0, []
        if not self.projected_paths:
            return 0.0, []

        path = self._bfs_path(self.civilian_pos, self.exit_pos)
        if not path or len(path) < 2:
            return 0.0, []

        nxt = path[1]
        reward = float(self.profile["rescue_progress_reward"])
        events: List[str] = []

        if nxt in self.fire_intensity:
            return 0.0, events

        current_x, current_y = self.civilian_pos
        if self.city_grid_true[current_y][current_x] == self.CIVILIAN:
            self.city_grid_true[current_y][current_x] = self.EMPTY

        self.civilian_pos = nxt
        if nxt == self.exit_pos:
            self.civilian_reached_exit = True
            reward += float(self.profile["rescue_reward"])
            self.city_grid_true[nxt[1]][nxt[0]] = self.EXIT
            events.append("Civilian reached the exit safely.")
        else:
            self.city_grid_true[nxt[1]][nxt[0]] = self.CIVILIAN
            events.append("Civilian advanced along the rescue corridor.")

        if nxt in self.projected_paths:
            reward += float(self.profile["path_reward"])
        return reward, events

    def _refresh_visibility(self) -> None:
        self._reveal_around_point(self.base_station, 3)
        for drone in self.drones.values():
            self._reveal_around_point((drone["x"], drone["y"]), 2)
        if self.visible_mask[self.civilian_pos[1]][self.civilian_pos[0]]:
            self.civilian_discovered = True
            self.civilian_discovered_with_power = True

    def _reveal_around_point(self, center: Coordinate, radius: int) -> None:
        for y in range(center[1] - radius, center[1] + radius + 1):
            for x in range(center[0] - radius, center[0] + radius + 1):
                if not self._in_bounds((x, y)):
                    continue
                if manhattan(center, (x, y)) <= radius + 1:
                    self.visible_mask[y][x] = True

    def _bfs_path(self, start: Coordinate, goal: Coordinate) -> Optional[List[Coordinate]]:
        if start == goal:
            return [start]
        queue = deque([start])
        parent: Dict[Coordinate, Optional[Coordinate]] = {start: None}
        while queue:
            current = queue.popleft()
            if current == goal:
                break
            for dx, dy in self.DIRECTIONS.values():
                nxt = (current[0] + dx, current[1] + dy)
                if nxt in parent or not self._in_bounds(nxt):
                    continue
                if self.city_grid_true[nxt[1]][nxt[0]] == self.WALL:
                    continue
                if nxt in self.fire_intensity:
                    continue
                parent[nxt] = current
                queue.append(nxt)
        if goal not in parent:
            return None
        path: List[Coordinate] = []
        cursor: Optional[Coordinate] = goal
        while cursor is not None:
            path.append(cursor)
            cursor = parent[cursor]
        path.reverse()
        return path

    def _is_fire_adjacent_to_civilian(self, fire_pos: Coordinate) -> bool:
        return manhattan(fire_pos, self.civilian_pos) <= 1

    def _can_enter(self, pos: Coordinate) -> bool:
        if not self._in_bounds(pos):
            return False
        return self.city_grid_true[pos[1]][pos[0]] != self.WALL

    def _in_bounds(self, pos: Coordinate) -> bool:
        return 0 <= pos[0] < self.width and 0 <= pos[1] < self.height

    def _is_done(self) -> bool:
        if self.step_count >= self.max_steps:
            return True
        if self.civilian_reached_exit and not self.fire_intensity:
            return True
        if all(drone["battery_level"] <= 0 for drone in self.drones.values()):
            return True
        return False
