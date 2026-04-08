from __future__ import annotations

import copy
import mimetypes
import os
import random
import time
from collections import deque
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from env import FireDroneSwarmEnv

ROOT = Path(__file__).resolve().parent
HTML_PATH = ROOT / "dashboard.html"
CSS_PATH = ROOT / "dashboard.css"
JS_PATH = ROOT / "dashboard.js"
MAP_UPLOADED_ASSET = ROOT / "map.png"
MAP_DEFAULT_ASSET = ROOT / "dashboard_map_uploaded.jpeg"
BUILDING_UPLOADED_ASSET = ROOT / "building.png"
BUILDING_TARGET_ASSET = ROOT / "dashboard_building_target.jpeg"

Coordinate = Tuple[int, int]

BUILDINGS: List[Dict[str, Any]] = [
    {
        "id": "oakwood_residences",
        "name": "Oakwood Residences",
        "street": "Oakwood Ave",
        "type": "residential",
        "rect": {"x": 0.30, "y": 0.07, "w": 0.18, "h": 0.12},
        "entry": [5, 7],
    },
    {
        "id": "central_plaza",
        "name": "Central Plaza Apartments",
        "street": "Central Blvd",
        "type": "apartment",
        "rect": {"x": 0.49, "y": 0.21, "w": 0.18, "h": 0.14},
        "entry": [5, 7],
    },
    {
        "id": "market_street",
        "name": "Market Street Shops",
        "street": "Market Street",
        "type": "commercial",
        "rect": {"x": 0.72, "y": 0.37, "w": 0.16, "h": 0.12},
        "entry": [5, 7],
    },
    {
        "id": "city_library",
        "name": "City Library",
        "street": "Market Street",
        "type": "civic",
        "rect": {"x": 0.72, "y": 0.48, "w": 0.15, "h": 0.12},
        "entry": [5, 7],
    },
    {
        "id": "maple_midrise",
        "name": "Maple Midrise",
        "street": "Maple St",
        "type": "apartment",
        "rect": {"x": 0.37, "y": 0.60, "w": 0.18, "h": 0.11},
        "entry": [5, 7],
    },
    {
        "id": "south_school",
        "name": "South Neighborhood School",
        "street": "Cedar Dr",
        "type": "school",
        "rect": {"x": 0.72, "y": 0.67, "w": 0.16, "h": 0.12},
        "entry": [5, 7],
    },
]

SCENARIO_BUILDING_POOLS = {
    "Easy": ["oakwood_residences", "city_library", "maple_midrise"],
    "Medium": ["oakwood_residences", "central_plaza", "market_street", "city_library"],
    "Hard": [building["id"] for building in BUILDINGS],
}


def resolve_map_background_path() -> Optional[Path]:
    candidates: List[str] = []
    env_candidate = os.environ.get("GREXO_MAP_IMAGE", "").strip()
    if env_candidate:
        candidates.append(env_candidate)
    if MAP_UPLOADED_ASSET.exists():
        candidates.append(str(MAP_UPLOADED_ASSET))
    if MAP_DEFAULT_ASSET.exists():
        candidates.append(str(MAP_DEFAULT_ASSET))
    for raw in candidates:
        try:
            path = Path(os.path.expandvars(os.path.expanduser(raw))).resolve()
        except OSError:
            continue
        if path.is_file():
            return path
    return None


def resolve_building_target_path() -> Optional[Path]:
    if BUILDING_UPLOADED_ASSET.exists():
        return BUILDING_UPLOADED_ASSET
    return BUILDING_TARGET_ASSET if BUILDING_TARGET_ASSET.exists() else None


def guess_content_type(path: Path) -> str:
    return mimetypes.guess_type(str(path))[0] or "application/octet-stream"


def manhattan(a: Coordinate, b: Coordinate) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def direction_between(start: Coordinate, end: Coordinate) -> Optional[str]:
    mapping = {
        (1, 0): "right",
        (-1, 0): "left",
        (0, 1): "down",
        (0, -1): "up",
    }
    return mapping.get((end[0] - start[0], end[1] - start[1]))


def build_analysis_payload(env: FireDroneSwarmEnv) -> Dict[str, Any]:
    bx0, by0, bx1, by1 = env.building_bounds
    total_tiles = 0
    visible_tiles = 0
    revealed_cells: List[List[int]] = []
    revealed_walls: List[List[int]] = []
    for y in range(by0, by1 + 1):
        for x in range(bx0, bx1 + 1):
            total_tiles += 1
            if not env.visible_mask[y][x]:
                continue
            visible_tiles += 1
            revealed_cells.append([x, y])
            if env.city_grid_true[y][x] == env.WALL:
                revealed_walls.append([x, y])
    progress = visible_tiles / max(1, total_tiles)
    safe_path = env._bfs_path(env.civilian_pos, env.exit_pos) if not env.civilian_reached_exit else [env.exit_pos]
    return {
        "progress": round(progress, 3),
        "revealed_tiles": visible_tiles,
        "total_tiles": total_tiles,
        "revealed_cells": revealed_cells,
        "revealed_walls": revealed_walls,
        "safe_path": [[x, y] for (x, y) in safe_path] if safe_path else [],
        "status": (
            "Dispatch and scan in progress."
            if progress < 0.25
            else "Interior walls mapped, rescue lane stabilizing."
            if progress < 0.75
            else "Interior analysis complete."
        ),
    }


def build_severity_payload(env: FireDroneSwarmEnv) -> Dict[str, Any]:
    active_fires = len(env.fire_intensity)
    total_heat = sum(env.fire_intensity.values())
    blocked = any(manhattan(pos, env.civilian_pos) <= 1 for pos in env.fire_intensity)
    score = min(100, active_fires * 10 + total_heat * 7 + (18 if blocked else 0))
    if score < 25:
        level = "LOW"
        note = "Contained incident. Normal response posture is enough."
    elif score < 50:
        level = "SERIOUS"
        note = "Balanced fire-rescue response required."
    elif score < 75:
        level = "SEVERE"
        note = "Reinforced rescue and suppression drones recommended."
    else:
        level = "CRITICAL"
        note = "Maximum response. Keep exit corridor clear at all costs."
    return {
        "score": score,
        "level": level,
        "team_note": note,
        "corridor_blocked": blocked,
        "active_fires": active_fires,
        "total_heat": total_heat,
    }


def build_structure_payload(env: FireDroneSwarmEnv) -> Dict[str, Any]:
    walls: List[List[int]] = []
    for y in range(env.height):
        for x in range(env.width):
            if env.city_grid_true[y][x] == env.WALL:
                walls.append([x, y])
    return {
        "bounds": list(env.building_bounds),
        "exit": [env.exit_pos[0], env.exit_pos[1]],
        "base_station": [env.base_station[0], env.base_station[1]],
        "sensor_alert": [env.sensor_alert[0], env.sensor_alert[1]],
        "walls": walls,
        "interior_walls": [[x, 6] for x in range(7, 11)] + [[9, y] for y in range(7, 10)],
    }


def choose_target_building(scenario: str) -> Dict[str, Any]:
    pool = SCENARIO_BUILDING_POOLS.get(scenario, SCENARIO_BUILDING_POOLS["Medium"])
    target_id = random.choice(pool)
    return copy.deepcopy(next(building for building in BUILDINGS if building["id"] == target_id))


class MissionCommander:
    def _path(
        self,
        env: FireDroneSwarmEnv,
        start: Coordinate,
        goal: Coordinate,
        avoid_fire: bool = True,
    ) -> Optional[List[Coordinate]]:
        if start == goal:
            return [start]
        queue = deque([start])
        parent: Dict[Coordinate, Optional[Coordinate]] = {start: None}
        while queue:
            current = queue.popleft()
            if current == goal:
                break
            for dx, dy in env.DIRECTIONS.values():
                nxt = (current[0] + dx, current[1] + dy)
                if nxt in parent or not env._in_bounds(nxt):
                    continue
                if env.city_grid_true[nxt[1]][nxt[0]] == env.WALL:
                    continue
                if avoid_fire and nxt in env.fire_intensity:
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

    def _next_move(
        self,
        env: FireDroneSwarmEnv,
        start: Coordinate,
        goal: Coordinate,
        avoid_fire: bool = True,
    ) -> Optional[str]:
        path = self._path(env, start, goal, avoid_fire=avoid_fire)
        if not path or len(path) < 2:
            return None
        return direction_between(path[0], path[1])

    def _adjacent_fire_direction(self, env: FireDroneSwarmEnv, position: Coordinate) -> Optional[str]:
        for direction, delta in env.DIRECTIONS.items():
            target = (position[0] + delta[0], position[1] + delta[1])
            if target in env.fire_intensity:
                return direction
        return None

    def _nearest_fire(self, env: FireDroneSwarmEnv, position: Coordinate) -> Optional[Coordinate]:
        if not env.fire_intensity:
            return None
        return min(env.fire_intensity, key=lambda pos: (manhattan(position, pos), -env.fire_intensity[pos]))

    def _fire_attack_position(
        self,
        env: FireDroneSwarmEnv,
        start: Coordinate,
        fire_pos: Coordinate,
    ) -> Optional[Coordinate]:
        candidates: List[Coordinate] = []
        for dx, dy in env.DIRECTIONS.values():
            nxt = (fire_pos[0] + dx, fire_pos[1] + dy)
            if not env._in_bounds(nxt):
                continue
            if env.city_grid_true[nxt[1]][nxt[0]] == env.WALL:
                continue
            candidates.append(nxt)
        if not candidates:
            return None
        ranked: List[Tuple[int, Coordinate]] = []
        for candidate in candidates:
            path = self._path(env, start, candidate, avoid_fire=False)
            if not path:
                continue
            ranked.append((len(path), candidate))
        if not ranked:
            return None
        ranked.sort(key=lambda item: item[0])
        return ranked[0][1]

    def _safe_exit_path(self, env: FireDroneSwarmEnv) -> Optional[List[Coordinate]]:
        return env._bfs_path(env.civilian_pos, env.exit_pos)

    def _projected_route(self, safe_path: Optional[List[Coordinate]], projected: set[Coordinate]) -> List[List[int]]:
        if not safe_path:
            return []
        return [[x, y] for (x, y) in safe_path if (x, y) in projected or (x, y) == safe_path[-1]]

    def act(self, env: FireDroneSwarmEnv, observation: Dict[str, Any]) -> Dict[str, Any]:
        drones = {item["id"]: item for item in observation["drones"]}
        commands: Dict[str, str] = {}
        objectives: Dict[str, str] = {drone_id: "Holding pattern." for drone_id in drones}
        comms: List[str] = []
        safe_path = self._safe_exit_path(env)
        analyzed_route = [[x, y] for (x, y) in safe_path] if safe_path else []
        projected_route = self._projected_route(safe_path, env.projected_paths)

        guide_ids = [drone_id for drone_id, drone in drones.items() if drone["role"] == "guide"]
        rescue_ids = [drone_id for drone_id, drone in drones.items() if drone["role"] == "rescue"]
        fire_ids = [drone_id for drone_id, drone in drones.items() if drone["role"] == "extinguish"]

        if not env.civilian_discovered:
            mission_phase = "THERMAL SCOUT / WALL ANALYSIS"
            camera_status = "Drones are leaving the station and scanning the target building."
            comms.append("SWARM: dispatching scout and rescue units to the alert sector.")
            for drone_id in guide_ids + rescue_ids:
                drone = drones[drone_id]
                pos = (drone["x"], drone["y"])
                if manhattan(pos, env.civilian_pos) <= 1:
                    commands[drone_id] = f"scan({drone_id}, 2)"
                    objectives[drone_id] = "Scan rooms and confirm civilian location."
                else:
                    move = self._next_move(env, pos, env.civilian_pos, avoid_fire=False)
                    if move:
                        commands[drone_id] = f"move({drone_id}, {move})"
                    objectives[drone_id] = "Advance to the interior civilian search area."
        elif env.fire_intensity:
            mission_phase = "FIRE SUPPRESSION / CORRIDOR CLEAR"
            camera_status = "Suppression drones are extinguishing interior fire before extraction."
            comms.append("SWARM: extinguish teams clearing fire, rescue teams holding near civilian.")
            for drone_id in guide_ids + rescue_ids:
                drone = drones[drone_id]
                pos = (drone["x"], drone["y"])
                if manhattan(pos, env.civilian_pos) <= 1:
                    commands.setdefault(drone_id, f"scan({drone_id}, 1)")
                    objectives[drone_id] = "Hold beside the civilian and scan the corridor."
                else:
                    move = self._next_move(env, pos, env.civilian_pos, avoid_fire=False)
                    if move:
                        commands[drone_id] = f"move({drone_id}, {move})"
                    objectives[drone_id] = "Stage beside the civilian for escort once fire is clear."
        elif not env.civilian_reached_exit:
            mission_phase = "ESCORT / EXIT CORRIDOR"
            camera_status = "Guide drones are projecting the exit lane and rescue drones are escorting the civilian."
            comms.append("SWARM: rescue escort active, all civilians follow the guide path to the exit.")
            if safe_path and len(safe_path) > 1:
                lead = safe_path[1]
                direction = direction_between(safe_path[0], lead)
                if direction and guide_ids:
                    guide_id = guide_ids[0]
                    guide_pos = (drones[guide_id]["x"], drones[guide_id]["y"])
                    if manhattan(guide_pos, env.civilian_pos) <= 1:
                        commands[guide_id] = f"project_path({guide_id}, {direction})"
                    else:
                        move = self._next_move(env, guide_pos, env.civilian_pos)
                        if move:
                            commands[guide_id] = f"move({guide_id}, {move})"
                    objectives[guide_id] = "Project the shortest safe path to the exit."
            for drone_id in rescue_ids:
                drone = drones[drone_id]
                pos = (drone["x"], drone["y"])
                if manhattan(pos, env.civilian_pos) > 1:
                    move = self._next_move(env, pos, env.civilian_pos)
                    if move:
                        commands[drone_id] = f"move({drone_id}, {move})"
                else:
                    commands.setdefault(drone_id, f"scan({drone_id}, 1)")
                objectives[drone_id] = "Escort civilian along the projected exit corridor."
        else:
            mission_phase = "RETURN / BASE STATION"
            camera_status = "Mission complete. Drones are returning to the station."
            comms.append("SWARM: hazard cleared, all units return to base.")
            for drone_id, drone in drones.items():
                pos = (drone["x"], drone["y"])
                move = self._next_move(env, pos, env.base_station)
                if move:
                    commands[drone_id] = f"move({drone_id}, {move})"
                objectives[drone_id] = "Return to base station and hold ready."

        for drone_id in fire_ids:
            drone = drones[drone_id]
            pos = (drone["x"], drone["y"])
            if env.fire_intensity:
                if pos in env.fire_intensity and drone["payload_capacity"] > 0:
                    commands[drone_id] = f"drop_ball({drone_id})"
                    objectives[drone_id] = "Drop suppression payload on the active fire cluster."
                    continue
                direction = self._adjacent_fire_direction(env, pos)
                if direction:
                    commands[drone_id] = f"spray({drone_id}, {direction})"
                    objectives[drone_id] = "Extinguish adjacent fire completely."
                    continue
                nearest = self._nearest_fire(env, pos)
                if nearest is not None:
                    attack_position = self._fire_attack_position(env, pos, nearest)
                    move = self._next_move(env, pos, attack_position, avoid_fire=False) if attack_position else None
                    if move:
                        commands[drone_id] = f"move({drone_id}, {move})"
                    objectives[drone_id] = "Reach the nearest fire node and suppress it."
            else:
                move = self._next_move(env, pos, env.base_station)
                if move:
                    commands.setdefault(drone_id, f"move({drone_id}, {move})")
                objectives[drone_id] = "Return to base after suppression complete."

        ordered = [commands.get(drone_id, f"noop({drone_id})") for drone_id in sorted(drones)]
        return {
            "commands": ordered,
            "mission_phase": mission_phase,
            "camera_status": camera_status,
            "comms": comms,
            "objectives": objectives,
            "analyzed_route": analyzed_route,
            "projected_route": projected_route,
        }


class DashboardSession:
    def __init__(self) -> None:
        self.lock = Lock()
        self.commander = MissionCommander()
        self.reset_runtime("Medium")

    def reset_runtime(self, scenario: str = "Medium") -> None:
        self.scenario = scenario if scenario in FireDroneSwarmEnv.SCENARIO_PROFILES else "Medium"
        self.target_building = choose_target_building(self.scenario)
        self.env = FireDroneSwarmEnv(num_drones=6)
        self.observation = self.env.reset(self.scenario)
        self.auto = False
        self.live_armed = False
        self.done = False
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.last_info = {
            "events": [],
            "graders": {
                "task1_scout_map": 0.0,
                "task2_containment": 0.0,
                "task3_coordinated_rescue": 0.0,
            },
            "fire_tiles_remaining": len(self.env.fire_intensity),
            "civilian_position": self.env.civilian_pos,
            "civilian_reached_exit": False,
            "coordination": {
                "blocking_fire_cleared": False,
                "suppression_drones": [],
                "path_drones": [],
            },
        }
        self.last_commands: List[str] = []
        self.last_objectives = {drone["id"]: "Holding pattern." for drone in self.observation["drones"]}
        self.events = [
            "GREXO dashboard ready.",
            f"Mission set to {self.scenario} difficulty.",
        ]
        self.mission_phase = "STANDBY / WAITING FOR LAUNCH"
        self.camera_status = "Mission idle at base station. Select difficulty and run the simulator."
        self.swarm_comms = ["SWARM NET: standing by for operator launch command."]
        self.analyzed_route: List[List[int]] = []
        self.projected_route: List[List[int]] = []
        self.last_auto_at = 0.0
        self.response_boost = 0

    def _score(self) -> float:
        graders = self.last_info["graders"]
        return round(
            (graders["task1_scout_map"] + graders["task2_containment"] + graders["task3_coordinated_rescue"]) / 3.0,
            3,
        )

    def append_event(self, message: str) -> None:
        self.events.append(message)
        if len(self.events) > 240:
            self.events = self.events[-240:]

    def arm_live(self, auto_enabled: Optional[bool] = None, message: Optional[str] = None) -> None:
        self.live_armed = True
        if auto_enabled is not None:
            self.auto = auto_enabled
        self.done = False
        self.last_auto_at = 0.0
        if message:
            self.append_event(message)

    def _translate_manual_command(self, drone_id: str, command: str) -> Optional[str]:
        mapping = {
            "move_up": f"move({drone_id}, up)",
            "move_down": f"move({drone_id}, down)",
            "move_left": f"move({drone_id}, left)",
            "move_right": f"move({drone_id}, right)",
        }
        return mapping.get(command)

    def _command_map(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for command in self.last_commands:
            if not command.endswith(")") or "(" not in command:
                continue
            name, rest = command.split("(", 1)
            args = [arg.strip() for arg in rest[:-1].split(",") if arg.strip()]
            if args:
                mapping[args[0]] = name.strip()
        return mapping

    def _build_drone_view(self) -> List[Dict[str, Any]]:
        command_map = self._command_map()
        drones: List[Dict[str, Any]] = []
        for drone in self.observation["drones"]:
            drone_id = drone["id"]
            status = command_map.get(drone_id, "idle")
            objective = self.last_objectives.get(drone_id, "Holding pattern.")
            drones.append(
                {
                    "id": drone_id,
                    "role": drone["role"],
                    "x": drone["x"],
                    "y": drone["y"],
                    "battery": int(round(drone["battery_level"])),
                    "payload": int(round(drone["payload_capacity"])),
                    "status": status,
                    "objective": objective,
                }
            )
        return drones

    def _snapshot_payload(self) -> Dict[str, Any]:
        severity = build_severity_payload(self.env)
        analysis = build_analysis_payload(self.env)
        map_background = resolve_map_background_path()
        building_target = resolve_building_target_path()
        return {
            "scenario": self.scenario,
            "difficulty_profile": self.env._difficulty_profile_payload(),
            "tick": self.env.step_count,
            "auto": self.auto,
            "live_armed": self.live_armed,
            "done": self.done,
            "active_fires": len(self.env.fire_intensity),
            "trapped_humans": 0 if self.env.civilian_reached_exit else 1,
            "rescued_humans": 1 if self.env.civilian_reached_exit else 0,
            "mission_phase": self.mission_phase,
            "camera_status": self.camera_status,
            "swarm_comms": self.swarm_comms,
            "events": self.events,
            "score": self._score(),
            "last_reward": round(self.last_reward, 3),
            "total_reward": round(self.total_reward, 3),
            "graders": copy.deepcopy(self.last_info["graders"]),
            "analyzed_route": self.analyzed_route,
            "projected_route": self.projected_route,
            "base_station": [self.env.base_station[0], self.env.base_station[1]],
            "sensor_alert": [self.env.sensor_alert[0], self.env.sensor_alert[1]],
            "civilian": {
                "x": self.env.civilian_pos[0],
                "y": self.env.civilian_pos[1],
                "active": not self.env.civilian_reached_exit,
            },
            "exit": {"x": self.env.exit_pos[0], "y": self.env.exit_pos[1]},
            "fire_nodes": [
                {"x": x, "y": y, "intensity": intensity}
                for (x, y), intensity in sorted(self.env.fire_intensity.items())
            ],
            "drones": self._build_drone_view(),
            "structure": build_structure_payload(self.env),
            "analysis": analysis,
            "severity": severity,
            "target_building": copy.deepcopy(self.target_building),
            "buildings": copy.deepcopy(BUILDINGS),
            "map_context": {
                "station_name": "North Response Station",
                "incident_label": self.target_building["name"],
                "dispatch_route": [
                    [x, y] for (x, y) in (self.commander._path(self.env, self.env.base_station, self.env.sensor_alert, avoid_fire=False) or [self.env.base_station])
                ],
                "eta_steps": max(0, len(self.commander._path(self.env, self.env.base_station, self.env.sensor_alert, avoid_fire=False) or []) - 1),
                "judge_caption": "Station launch -> shortest route -> building scan -> suppression -> rescue -> return",
            },
            "map_background_url": "/map-background" if map_background else None,
            "building_target_url": "/building-target" if building_target else None,
            "response_boost": self.response_boost,
        }

    def _apply_step(self, commands: List[str], source: str) -> None:
        self.last_commands = commands
        self.observation, self.last_reward, self.done, self.last_info = self.env.step({"commands": commands})
        if self.env.fire_intensity:
            self.done = False
        self.total_reward += self.last_reward
        for event in self.last_info["events"]:
            self.append_event(event)
        self.append_event(f"{source.upper()} -> {', '.join(commands)}")
        if self.env.civilian_reached_exit and not self.env.fire_intensity:
            self.mission_phase = "MISSION COMPLETE / EXTRACTION SECURED"
            self.camera_status = "All civilians safe and all fire nodes extinguished."
            self.swarm_comms = ["SWARM NET: mission complete, all units returning to base."]
        elif self.env.civilian_reached_exit and self.env.fire_intensity:
            self.mission_phase = "FINAL FIRE SWEEP / HOTSPOT CLEANUP"
            self.camera_status = "Civilian safe. Remaining drones are clearing the last hotspots."
            self.swarm_comms = ["SWARM NET: extraction complete, suppression sweep active."]

    def auto_tick(self) -> None:
        if not self.auto or self.done:
            return
        interval = 0.35 if self.response_boost > 0 else 0.60
        now = time.monotonic()
        if now - self.last_auto_at < interval:
            return
        plan = self.commander.act(self.env, self.observation)
        self.mission_phase = plan["mission_phase"]
        self.camera_status = plan["camera_status"]
        self.swarm_comms = plan["comms"]
        self.last_objectives = plan["objectives"]
        self.analyzed_route = plan["analyzed_route"]
        self.projected_route = plan["projected_route"]
        self._apply_step(plan["commands"], source="auto")
        self.last_auto_at = now
        if self.response_boost > 0:
            self.response_boost -= 1

    def snapshot(self) -> Dict[str, Any]:
        if self.live_armed:
            self.auto_tick()
        return self._snapshot_payload()

    def reset(self, scenario: Optional[str] = None) -> Dict[str, Any]:
        self.reset_runtime(scenario or self.scenario)
        return {"ok": True, "message": f"Environment reset for {self.scenario} difficulty.", "scenario": self.scenario}

    def trigger_alert(self, scenario: Optional[str] = None) -> Dict[str, Any]:
        self.reset_runtime(scenario or self.scenario)
        self.arm_live(auto_enabled=True, message="Live mission armed from dashboard.")
        self.mission_phase = "THERMAL SCOUT / WALL ANALYSIS"
        self.camera_status = "Drones launched from the fire station toward the target building."
        self.swarm_comms = ["SWARM NET: fastest route locked, launch sequence active."]
        return {"ok": True, "scenario": self.scenario, "auto": self.auto, "live_armed": self.live_armed}

    def toggle_auto(self, scenario: Optional[str] = None) -> Dict[str, Any]:
        requested = scenario if scenario in FireDroneSwarmEnv.SCENARIO_PROFILES else self.scenario
        if requested != self.scenario and not self.live_armed:
            self.reset_runtime(requested)
        self.auto = not self.auto
        if self.auto:
            self.arm_live(auto_enabled=True, message="Auto AI enabled.")
        else:
            self.append_event("Auto AI disabled.")
        return {"ok": True, "auto": self.auto, "live_armed": self.live_armed, "scenario": self.scenario}

    def manual_step(self, drone_id: str, command: str) -> Dict[str, Any]:
        translated = self._translate_manual_command(drone_id, command)
        if translated is None:
            return {"ok": False, "reason": "unsupported command"}
        self.arm_live(auto_enabled=self.auto, message=f"Manual override for {drone_id}.")
        commands = [f"noop({item['id']})" for item in self.observation["drones"]]
        for index, item in enumerate(self.observation["drones"]):
            if item["id"] == drone_id:
                commands[index] = translated
                break
        self._apply_step(commands, source="manual")
        return {"ok": True, "drone_id": drone_id, "command": command}

    def spawn_fire(self) -> Dict[str, Any]:
        self.arm_live(auto_enabled=True, message="Additional fire injected from dashboard.")
        candidates = [(11, 8), (8, 9), (10, 9), (7, 8), (11, 7), (10, 6)]
        for pos in candidates:
            if pos in self.env.fire_intensity or self.env.city_grid_true[pos[1]][pos[0]] == self.env.WALL:
                continue
            if pos == self.env.civilian_pos or pos == self.env.exit_pos:
                continue
            self.env.fire_intensity[pos] = 2
            self.env.city_grid_true[pos[1]][pos[0]] = self.env.FIRE
            self.response_boost = min(4, self.response_boost + 2)
            self.done = False
            self.append_event(f"Operator injected fire at {pos[0]},{pos[1]}.")
            return {"ok": True, "fire_at": {"x": pos[0], "y": pos[1]}, "response_boost": self.response_boost}
        return {"ok": False, "reason": "no valid fire location"}

    def spawn_human(self) -> Dict[str, Any]:
        self.arm_live(auto_enabled=True, message="Additional rescue request injected from dashboard.")
        if not self.env.civilian_reached_exit:
            self.response_boost = min(4, self.response_boost + 2)
            self.append_event("Extra rescue drones assigned to the active civilian corridor.")
            return {
                "ok": True,
                "human_at": {"x": self.env.civilian_pos[0], "y": self.env.civilian_pos[1]},
                "existing": True,
                "response_boost": self.response_boost,
            }
        self.env.civilian_reached_exit = False
        self.env.civilian_discovered = False
        self.env.civilian_discovered_with_power = False
        self.env.civilian_pos = (10, 8)
        self.env.city_grid_true[8][10] = self.env.CIVILIAN
        self.done = False
        self.response_boost = min(4, self.response_boost + 2)
        self.append_event("New civilian injected into the target building.")
        return {"ok": True, "human_at": {"x": 10, "y": 8}, "response_boost": self.response_boost}

    def refill(self) -> Dict[str, Any]:
        for drone in self.observation["drones"]:
            drone["battery_level"] = 100
            drone["payload_capacity"] = self.env.profile["payload_start"]
            live = self.env.drones[drone["id"]]
            live["battery_level"] = 100
            live["payload_capacity"] = self.env.profile["payload_start"]
        self.append_event("All drones refilled and recharged.")
        return {"ok": True, "refilled": [drone["id"] for drone in self.observation["drones"]]}


SESSION = DashboardSession()


def build_replay_payload(scenario: str, max_steps: Optional[int] = None) -> Dict[str, Any]:
    limits = {"Easy": 36, "Medium": 50, "Hard": 64}
    replay_session = DashboardSession()
    replay_session.reset_runtime(scenario)
    frames: List[Dict[str, Any]] = [replay_session.snapshot()]
    for _ in range(max_steps or limits.get(scenario, 50)):
        plan = replay_session.commander.act(replay_session.env, replay_session.observation)
        replay_session.mission_phase = plan["mission_phase"]
        replay_session.camera_status = plan["camera_status"]
        replay_session.swarm_comms = plan["comms"]
        replay_session.last_objectives = plan["objectives"]
        replay_session.analyzed_route = plan["analyzed_route"]
        replay_session.projected_route = plan["projected_route"]
        replay_session._apply_step(plan["commands"], source="replay")
        frames.append(replay_session.snapshot())
        if replay_session.done:
            break
    return {"ok": True, "scenario": scenario, "frames": frames}
