from __future__ import annotations

from threading import Lock
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field, model_validator

from dashboard_backend import (
    CSS_PATH,
    HTML_PATH,
    JS_PATH,
    SESSION as DASHBOARD_SESSION,
    build_replay_payload,
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
    ScenarioName,
    StateResponse,
    StepInfoResponse,
    StepRequest,
    StepResponse,
    TaskDefinition,
    TaskListResponse,
    TaskScoreRange,
    ValidateResponse,
)

APP_TITLE = "GREXO Fire Rescue API"
APP_VERSION = "1.3.0"
APP_DESCRIPTION = """
FastAPI service for the GREXO fire rescue drone swarm environment.

Core OpenEnv endpoints:
- `POST /reset`
- `GET /state`
- `POST /step`
- `GET /tasks`
- `GET /validate`

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
TASKS = [
    TaskDefinition(
        id="scout_and_map",
        name="Scout And Map",
        description="Reveal the civilian location by scanning the target structure before resources are exhausted.",
        grader="grade_task1_scout_and_map",
        score_range=TaskScoreRange(min=0.01, max=0.99),
        success_criteria="A powered scan reveals the civilian and task1_scout_map advances toward 0.99.",
    ),
    TaskDefinition(
        id="fire_containment",
        name="Fire Containment",
        description="Suppress the active fire tiles and keep the rescue corridor from collapsing under the selected difficulty.",
        grader="grade_task2_containment",
        score_range=TaskScoreRange(min=0.01, max=0.99),
        success_criteria="All initial fire tiles are progressively cleared and task2_containment approaches 0.99.",
    ),
    TaskDefinition(
        id="coordinated_rescue",
        name="Coordinated Rescue",
        description="Clear the route, project a safe corridor, and escort the civilian to the exit with role coordination.",
        grader="grade_task3_coordinated_rescue",
        score_range=TaskScoreRange(min=0.01, max=0.99),
        success_criteria="The civilian reaches the exit after suppression and task3_coordinated_rescue approaches 0.99.",
    ),
]


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
            "tasks": "GET /tasks",
            "validate": "GET /validate",
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


@app.get("/tasks", response_model=TaskListResponse, tags=["meta"])
def list_tasks() -> TaskListResponse:
    return TaskListResponse(count=len(TASKS), tasks=TASKS)


@app.get("/validate", response_model=ValidateResponse, tags=["meta"])
def validate_environment() -> ValidateResponse:
    checks = {
        "has_reset": True,
        "has_state": True,
        "has_step": True,
        "has_tasks": True,
        "has_three_tasks": len(TASKS) >= 3,
        "task_scores_bounded": all(
            0.0 < task.score_range.min < task.score_range.max < 1.0
            for task in TASKS
        ),
    }
    return ValidateResponse(
        valid=all(checks.values()),
        service=APP_TITLE,
        version=APP_VERSION,
        checks=checks,
    )


@app.post("/reset", response_model=ResetResponse, tags=["core"])
def reset_environment(request: Optional[ResetRequest] = Body(default=None)) -> ResetResponse:
    global CURRENT_SCENARIO
    resolved = (request or ResetRequest()).resolved_difficulty
    with ENV_LOCK:
        observation = ENV.reset(resolved)
        CURRENT_SCENARIO = ENV.scenario
    initial_state = _build_observation(observation)
    return ResetResponse(
        state=initial_state,
        reward=0.0,
        done=False,
        info={},
    )


@app.get("/state", response_model=StateResponse, tags=["core"])
def get_state() -> StateResponse:
    with ENV_LOCK:
        observation = ENV.state()
    return StateResponse(state=_build_observation(observation))


@app.post("/step", response_model=StepResponse, tags=["core"])
def step_environment(request: Optional[StepRequest] = Body(default=None)) -> StepResponse:
    resolved_request = request or StepRequest()
    with ENV_LOCK:
        observation, reward, done, info = ENV.step({"commands": resolved_request.commands})
    return StepResponse(
        state=_build_observation(observation),
        reward=float(reward),
        done=bool(done),
        info=_build_info(info),
    )


@app.get("/dashboard/state", response_model=DashboardStateResponse, tags=["dashboard"])
def dashboard_state() -> DashboardStateResponse:
    with DASHBOARD_SESSION.lock:
        payload = DASHBOARD_SESSION.snapshot()
    return DashboardStateResponse.model_validate(payload)


@app.post("/dashboard/reset", response_model=DashboardSimpleResponse, tags=["dashboard"])
def dashboard_reset(request: Optional[DashboardDifficultyRequest] = Body(default=None)) -> DashboardSimpleResponse:
    resolved_request = request or DashboardDifficultyRequest()
    with DASHBOARD_SESSION.lock:
        payload = DASHBOARD_SESSION.reset(resolved_request.resolved)
    return DashboardSimpleResponse.model_validate(payload)


@app.post("/dashboard/step", response_model=DashboardStepResponse, tags=["dashboard"])
def dashboard_step(request: DashboardManualStepRequest) -> DashboardStepResponse:
    with DASHBOARD_SESSION.lock:
        payload = DASHBOARD_SESSION.manual_step(request.drone_id, request.command)
    return DashboardStepResponse.model_validate(payload)


@app.post("/dashboard/api/trigger_alert", response_model=DashboardActionResponse, tags=["dashboard"])
def dashboard_trigger_alert(request: Optional[DashboardDifficultyRequest] = Body(default=None)) -> DashboardActionResponse:
    resolved_request = request or DashboardDifficultyRequest()
    with DASHBOARD_SESSION.lock:
        payload = DASHBOARD_SESSION.trigger_alert(resolved_request.resolved)
    return DashboardActionResponse.model_validate(payload)


@app.post("/dashboard/api/toggle_auto", response_model=DashboardActionResponse, tags=["dashboard"])
def dashboard_toggle_auto(request: Optional[DashboardDifficultyRequest] = Body(default=None)) -> DashboardActionResponse:
    resolved_request = request or DashboardDifficultyRequest()
    with DASHBOARD_SESSION.lock:
        payload = DASHBOARD_SESSION.toggle_auto(resolved_request.resolved)
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
def dashboard_run_ai_simulation(request: Optional[DashboardDifficultyRequest] = Body(default=None)) -> DashboardReplayResponse:
    resolved_request = request or DashboardDifficultyRequest()
    payload = build_replay_payload(resolved_request.resolved)
    return DashboardReplayResponse.model_validate(payload)


if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run(
        "app:app",
        host=os.getenv("GREXO_HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", os.getenv("GREXO_PORT", "7860"))),
        reload=False,
    )
