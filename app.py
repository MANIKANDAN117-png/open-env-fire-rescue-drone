from __future__ import annotations

from threading import Lock
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field, model_validator

from dashboard_backend import (
    CSS_PATH,
    HTML_PATH,
    JS_PATH,
    SESSION as DASHBOARD_SESSION,
    build_replay_payload,
    guess_content_type,
    resolve_building_target_path,
    resolve_map_background_path,
)
from env import FireDroneSwarmEnv
from models import (
    CoordinationSummary,
    GraderSummary,
    HealthResponse,
    ObservationResponse,
    ResetRequest,
    ResetResponse,
    RootResponse,
    StepInfoResponse,
    StepRequest,
    StepResponse,
    ScenarioName,
)

APP_TITLE = "GREXO Fire Rescue API"
APP_VERSION = "1.2.0"
APP_DESCRIPTION = """
FastAPI service for the GREXO fire rescue drone swarm environment.

Core OpenEnv endpoints:
- `POST /reset`
- `GET /state`
- `POST /step`

Dashboard endpoints are also available under `/dashboard/...`.
Swagger UI is available at `/docs` and the OpenAPI schema is available at `/openapi.json`.
"""

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
    docs_url="/docs",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ENV_LOCK = Lock()
ENV = FireDroneSwarmEnv(num_drones=6)
CURRENT_SCENARIO: ScenarioName = "Medium"


class DashboardDifficultyRequest(BaseModel):
    scenario: Optional[ScenarioName] = Field(default=None)
    difficulty: Optional[ScenarioName] = Field(default=None)

    @model_validator(mode="after")
    def default_selector(self) -> "DashboardDifficultyRequest":
        if self.scenario is None and self.difficulty is None:
            self.scenario = "Medium"
        return self

    @property
    def resolved(self) -> ScenarioName:
        return self.difficulty or self.scenario or "Medium"


class DashboardManualStepRequest(BaseModel):
    drone_id: str = Field(..., description="Dashboard drone id such as drone_0.")
    command: str = Field(..., description="Manual command: move_up, move_down, move_left, or move_right.")


class DashboardSimpleResponse(BaseModel):
    ok: bool
    message: Optional[str] = None
    scenario: Optional[ScenarioName] = None


class DashboardReplayResponse(BaseModel):
    ok: bool
    scenario: ScenarioName
    frames: list[dict[str, Any]]


class DashboardActionResponse(BaseModel):
    ok: bool
    auto: Optional[bool] = None
    live_armed: Optional[bool] = None
    scenario: Optional[ScenarioName] = None
    reason: Optional[str] = None
    response_boost: Optional[int] = None


class DashboardRefillResponse(BaseModel):
    ok: bool
    refilled: list[str] = Field(default_factory=list)


class DashboardStepResponse(BaseModel):
    ok: bool
    drone_id: Optional[str] = None
    command: Optional[str] = None
    reason: Optional[str] = None


class DashboardSpawnResponse(BaseModel):
    ok: bool
    fire_at: Optional[dict[str, int]] = None
    human_at: Optional[dict[str, int]] = None
    existing: Optional[bool] = None
    response_boost: Optional[int] = None
    reason: Optional[str] = None


class DashboardStateResponse(BaseModel):
    scenario: ScenarioName
    difficulty_profile: dict[str, Any]
    tick: int
    auto: bool
    live_armed: bool
    done: bool
    active_fires: int
    trapped_humans: int
    rescued_humans: int
    mission_phase: str
    camera_status: str
    swarm_comms: list[str]
    events: list[str]
    score: float
    last_reward: float
    total_reward: float
    graders: dict[str, float]
    analyzed_route: list[list[int]]
    projected_route: list[list[int]]
    base_station: list[int]
    sensor_alert: list[int]
    civilian: dict[str, Any]
    exit: dict[str, int]
    fire_nodes: list[dict[str, Any]]
    drones: list[dict[str, Any]]
    structure: dict[str, Any]
    analysis: dict[str, Any]
    severity: dict[str, Any]
    target_building: dict[str, Any]
    buildings: list[dict[str, Any]]
    map_context: dict[str, Any]
    map_background_url: Optional[str] = None
    building_target_url: Optional[str] = None
    response_boost: int


def _build_observation(payload: Dict[str, Any]) -> ObservationResponse:
    return ObservationResponse.model_validate(payload)


def _build_info(payload: Dict[str, Any]) -> StepInfoResponse:
    return StepInfoResponse(
        events=list(payload.get("events", [])),
        graders=GraderSummary(**payload.get("graders", {})),
        fire_tiles_remaining=int(payload.get("fire_tiles_remaining", 0)),
        civilian_position=tuple(payload.get("civilian_position", (0, 0))),
        civilian_reached_exit=bool(payload.get("civilian_reached_exit", False)),
        coordination=CoordinationSummary(**payload.get("coordination", {})),
    )


def dashboard_html() -> FileResponse:
    if not HTML_PATH.exists():
        raise HTTPException(status_code=404, detail="dashboard.html not found")
    return FileResponse(HTML_PATH, media_type="text/html; charset=utf-8")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root() -> FileResponse:
    return dashboard_html()


@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
def dashboard_route() -> FileResponse:
    return dashboard_html()


@app.get("/dashboard.css", include_in_schema=False)
def dashboard_css() -> FileResponse:
    return FileResponse(CSS_PATH, media_type="text/css; charset=utf-8")


@app.get("/dashboard.js", include_in_schema=False)
def dashboard_js() -> FileResponse:
    return FileResponse(JS_PATH, media_type="application/javascript; charset=utf-8")


@app.get("/map-background", include_in_schema=False)
def map_background() -> FileResponse:
    path = resolve_map_background_path()
    if path is None:
        raise HTTPException(status_code=404, detail="Map background not configured")
    return FileResponse(path, media_type=guess_content_type(path))


@app.get("/building-target", include_in_schema=False)
def building_target() -> FileResponse:
    path = resolve_building_target_path()
    if path is None:
        raise HTTPException(status_code=404, detail="Building target not configured")
    return FileResponse(path, media_type=guess_content_type(path))


@app.get("/meta", response_model=RootResponse, tags=["meta"])
def meta() -> RootResponse:
    return RootResponse(
        service=APP_TITLE,
        version=APP_VERSION,
        docs_url="/docs",
        openapi_url="/openapi.json",
        endpoints={
            "reset": "POST /reset",
            "state": "GET /state",
            "step": "POST /step",
            "dashboard": "GET /dashboard",
            "dashboard_state": "GET /dashboard/state",
            "health": "GET /health",
        },
    )


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    with ENV_LOCK:
        scenario = ENV.scenario
        step_count = ENV.step_count
    return HealthResponse(status="ok", scenario=scenario, step_count=step_count)


@app.post("/reset", response_model=ResetResponse, tags=["core"])
def reset_environment(request: ResetRequest) -> ResetResponse:
    global CURRENT_SCENARIO
    resolved = request.resolved_difficulty
    with ENV_LOCK:
        observation = ENV.reset(resolved)
        CURRENT_SCENARIO = ENV.scenario
    return ResetResponse(
        scenario=CURRENT_SCENARIO,
        difficulty=CURRENT_SCENARIO,
        observation=_build_observation(observation),
    )


@app.get("/state", response_model=ObservationResponse, tags=["core"])
def get_state() -> ObservationResponse:
    with ENV_LOCK:
        observation = ENV.state()
    return _build_observation(observation)


@app.post("/step", response_model=StepResponse, tags=["core"])
def step_environment(request: StepRequest) -> StepResponse:
    with ENV_LOCK:
        observation, reward, done, info = ENV.step({"commands": request.commands})
    return StepResponse(
        observation=_build_observation(observation),
        reward=reward,
        done=done,
        info=_build_info(info),
    )


@app.get("/dashboard/state", response_model=DashboardStateResponse, tags=["dashboard"])
def dashboard_state() -> DashboardStateResponse:
    with DASHBOARD_SESSION.lock:
        payload = DASHBOARD_SESSION.snapshot()
    return DashboardStateResponse.model_validate(payload)


@app.post("/dashboard/reset", response_model=DashboardSimpleResponse, tags=["dashboard"])
def dashboard_reset(request: DashboardDifficultyRequest) -> DashboardSimpleResponse:
    with DASHBOARD_SESSION.lock:
        payload = DASHBOARD_SESSION.reset(request.resolved)
    return DashboardSimpleResponse.model_validate(payload)


@app.post("/dashboard/step", response_model=DashboardStepResponse, tags=["dashboard"])
def dashboard_step(request: DashboardManualStepRequest) -> DashboardStepResponse:
    with DASHBOARD_SESSION.lock:
        payload = DASHBOARD_SESSION.manual_step(request.drone_id, request.command)
    return DashboardStepResponse.model_validate(payload)


@app.post("/dashboard/api/trigger_alert", response_model=DashboardActionResponse, tags=["dashboard"])
def dashboard_trigger_alert(request: DashboardDifficultyRequest) -> DashboardActionResponse:
    with DASHBOARD_SESSION.lock:
        payload = DASHBOARD_SESSION.trigger_alert(request.resolved)
    return DashboardActionResponse.model_validate(payload)


@app.post("/dashboard/api/toggle_auto", response_model=DashboardActionResponse, tags=["dashboard"])
def dashboard_toggle_auto(request: DashboardDifficultyRequest) -> DashboardActionResponse:
    with DASHBOARD_SESSION.lock:
        payload = DASHBOARD_SESSION.toggle_auto(request.resolved)
    return DashboardActionResponse.model_validate(payload)


@app.post("/dashboard/api/spawn_fire", response_model=DashboardSpawnResponse, tags=["dashboard"])
def dashboard_spawn_fire() -> DashboardSpawnResponse:
    with DASHBOARD_SESSION.lock:
        payload = DASHBOARD_SESSION.spawn_fire()
    return DashboardSpawnResponse.model_validate(payload)


@app.post("/dashboard/api/spawn_human", response_model=DashboardSpawnResponse, tags=["dashboard"])
def dashboard_spawn_human() -> DashboardSpawnResponse:
    with DASHBOARD_SESSION.lock:
        payload = DASHBOARD_SESSION.spawn_human()
    return DashboardSpawnResponse.model_validate(payload)


@app.post("/dashboard/api/refill", response_model=DashboardRefillResponse, tags=["dashboard"])
def dashboard_refill() -> DashboardRefillResponse:
    with DASHBOARD_SESSION.lock:
        payload = DASHBOARD_SESSION.refill()
    return DashboardRefillResponse.model_validate(payload)


@app.post("/dashboard/api/run_ai_simulation", response_model=DashboardReplayResponse, tags=["dashboard"])
def dashboard_run_ai_simulation(request: DashboardDifficultyRequest) -> DashboardReplayResponse:
    payload = build_replay_payload(request.resolved)
    return DashboardReplayResponse.model_validate(payload)
