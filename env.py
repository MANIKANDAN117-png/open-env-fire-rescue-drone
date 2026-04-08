from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Any, Dict, List, Optional, Set, Tuple

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
        reward = -float(self.profile["step_penalty"])
        parsed = self._normalize_action(action)
        commands_by_drone = self._parse_commands(parsed.commands)
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

        info["graders"] = {
            "task1_scout_map": self.grade_task1_scout_and_map(),
            "task2_containment": self.grade_task2_containment(),
            "task3_coordinated_rescue": self.grade_task3_coordinated_rescue(),
        }
        info["fire_tiles_remaining"] = len(self.fire_intensity)
        info["civilian_position"] = self.civilian_pos
        info["civilian_reached_exit"] = self.civilian_reached_exit
        info["coordination"] = {
            "blocking_fire_cleared": self.fire_blocking_civilian_cleared,
            "suppression_drones": sorted(self.fire_suppression_drone_ids),
            "path_drones": sorted(self.path_projection_drone_ids),
        }

        done = self._is_done()
        return self.state(), float(round(reward, 3)), done, info

    def grade_task1_scout_and_map(self) -> float:
        return 1.0 if self.civilian_discovered_with_power else 0.0

    def grade_task2_containment(self) -> float:
        if self.initial_fire_tiles == 0:
            return 1.0
        cleared = self.initial_fire_tiles - len(self.fire_intensity)
        return max(0.0, min(1.0, cleared / self.initial_fire_tiles))

    def grade_task3_coordinated_rescue(self) -> float:
        coordinated = bool(
            self.civilian_reached_exit
            and not self.fire_intensity
            and self.fire_suppression_drone_ids
            and self.path_projection_drone_ids
        )
        return 1.0 if coordinated else 0.0

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
                continue
            name, remainder = raw.split("(", 1)
            args_text = remainder[:-1]
            args = [item.strip() for item in args_text.split(",") if item.strip()]
            if not args:
                continue
            drone_id = args[0]
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
            return -10.0, [f"{drone_id} is out of battery."]

        if command_name == "noop":
            drone["battery_level"] = max(0.0, drone["battery_level"] - float(self.profile["idle_battery_cost"]))
            return reward, events

        drone["battery_level"] = max(0.0, drone["battery_level"] - float(self.profile["action_battery_cost"]))

        if command_name == "move":
            if not args:
                return reward, events
            direction = args[0]
            delta = self.DIRECTIONS.get(direction)
            if not delta:
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
                return reward, events
            direction = args[0]
            delta = self.DIRECTIONS.get(direction)
            if not delta:
                return reward, events
            target = (drone["x"] + delta[0], drone["y"] + delta[1])
            drone["payload_capacity"] = max(0.0, drone["payload_capacity"] - 1.0)
            reward += self._reduce_fire(target, amount=int(self.profile["spray_power"]), drone_id=drone_id)
            if target not in self.fire_intensity:
                events.append(f"{drone_id} sprayed {direction}.")
            return reward, events

        if command_name == "drop_ball":
            if drone["payload_capacity"] <= 0:
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
                return reward, events
            direction = args[0]
            delta = self.DIRECTIONS.get(direction)
            if not delta:
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
            and abs(drone["x"] - self.civilian_pos[0]) + abs(drone["y"] - self.civilian_pos[1]) <= 1
            for drone in self.drones.values()
        )
        if not escort_nearby:
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
                if abs(center[0] - x) + abs(center[1] - y) <= radius + 1:
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
        return abs(fire_pos[0] - self.civilian_pos[0]) + abs(fire_pos[1] - self.civilian_pos[1]) <= 1

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
