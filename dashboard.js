const API = `${window.location.origin}/dashboard`;
const POLL_MS = 500;
const REPLAY_MS = 150;
const MAX_LOG_LINES = 40;
const MAP_BLUEPRINT = { width: 1600, height: 1000 };
const INTERIOR_BLUEPRINT = { width: 920, height: 920 };
const FALLBACK_BUILDINGS = [
  { id: "oakwood_residences", name: "Oakwood Residences", street: "Oakwood Ave", rect: { x: 0.3, y: 0.07, w: 0.18, h: 0.12 } },
  { id: "central_plaza", name: "Central Plaza Apartments", street: "Central Blvd", rect: { x: 0.49, y: 0.21, w: 0.18, h: 0.14 } },
  { id: "market_street", name: "Market Street Shops", street: "Market Street", rect: { x: 0.72, y: 0.37, w: 0.16, h: 0.12 } },
  { id: "city_library", name: "City Library", street: "Market Street", rect: { x: 0.72, y: 0.48, w: 0.15, h: 0.12 } },
  { id: "maple_midrise", name: "Maple Apartments", street: "Maple St", rect: { x: 0.37, y: 0.6, w: 0.18, h: 0.11 } },
  { id: "south_school", name: "Neighborhood School", street: "Cedar Dr", rect: { x: 0.72, y: 0.67, w: 0.16, h: 0.12 } },
];
const FALLBACK_STRUCTURE = {
  bounds: [5, 3, 12, 11],
  interior_walls: [[7, 6], [8, 6], [9, 6], [10, 6], [9, 7], [9, 8], [9, 9]],
};

const DIFFICULTY_COPY = {
  Easy: "Easy difficulty uses fewer initial fire nodes, more battery tolerance, and higher rescue reward forgiveness.",
  Medium: "Medium difficulty balances fire spread, rescue pressure, and reward penalties.",
  Hard: "Hard difficulty increases fire spread, lowers operating tolerance, and applies stricter reward pressure.",
};

const dom = {
  connectStatus: document.getElementById("connectStatus"),
  phaseText: document.getElementById("phaseText"),
  cameraText: document.getElementById("cameraText"),
  difficultyBadge: document.getElementById("difficultyBadge"),
  activeFires: document.getElementById("activeFires"),
  trappedHumans: document.getElementById("trappedHumans"),
  rescuedHumans: document.getElementById("rescuedHumans"),
  tickValue: document.getElementById("tickValue"),
  scoreValue: document.getElementById("scoreValue"),
  difficultyHelp: document.getElementById("difficultyHelp"),
  runSimulationBtn: document.getElementById("runSimulationBtn"),
  toggleAutoBtn: document.getElementById("toggleAutoBtn"),
  followLiveBtn: document.getElementById("followLiveBtn"),
  spawnFireBtn: document.getElementById("spawnFireBtn"),
  spawnHumanBtn: document.getElementById("spawnHumanBtn"),
  refillBtn: document.getElementById("refillBtn"),
  resetBtn: document.getElementById("resetBtn"),
  selectedDroneText: document.getElementById("selectedDroneText"),
  lastRewardValue: document.getElementById("lastRewardValue"),
  totalRewardValue: document.getElementById("totalRewardValue"),
  severityValue: document.getElementById("severityValue"),
  responseBoostValue: document.getElementById("responseBoostValue"),
  analysisPercent: document.getElementById("analysisPercent"),
  analysisBar: document.getElementById("analysisBar"),
  severityNote: document.getElementById("severityNote"),
  targetBanner: document.getElementById("targetBanner"),
  buildingStatus: document.getElementById("buildingStatus"),
  mapMissionNote: document.getElementById("mapMissionNote"),
  difficultyButtons: [...document.querySelectorAll("[data-scenario]")],
  dronePills: document.getElementById("dronePills"),
  droneList: document.getElementById("droneList"),
  logBox: document.getElementById("logBox"),
  mapBackdropCanvas: document.getElementById("mapBackdropCanvas"),
  mapCanvas: document.getElementById("mapCanvas"),
  interiorBackdropCanvas: document.getElementById("interiorBackdropCanvas"),
  interiorCanvas: document.getElementById("interiorCanvas"),
};

const mapBgCtx = dom.mapBackdropCanvas.getContext("2d");
const mapCtx = dom.mapCanvas.getContext("2d");
const interiorBgCtx = dom.interiorBackdropCanvas.getContext("2d");
const interiorCtx = dom.interiorCanvas.getContext("2d");

let selectedScenario = "Medium";
let selectedDrone = null;
let mode = "live";
let liveScene = null;
let replayFrames = [];
let replayIndex = 0;
let replayTimer = null;
let lastSeenEvent = "";
let returnAnimation = { key: null, startedAt: 0 };
let renderStarted = false;
let backgroundDirty = true;

function log(message, kind = "info") {
  const line = document.createElement("div");
  line.className = `logLine ${kind === "err" ? "err" : kind === "warn" ? "warn" : ""}`;
  line.textContent = message;
  dom.logBox.prepend(line);
  while (dom.logBox.children.length > MAX_LOG_LINES) {
    dom.logBox.removeChild(dom.logBox.lastChild);
  }
}

function resizeCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.floor(rect.width * ratio);
  canvas.height = Math.floor(rect.height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function resizeAll() {
  [dom.mapBackdropCanvas, dom.mapCanvas, dom.interiorBackdropCanvas, dom.interiorCanvas].forEach(resizeCanvas);
  backgroundDirty = true;
  render();
}

function setDifficulty(level) {
  selectedScenario = level;
  dom.difficultyButtons.forEach((btn) => btn.classList.toggle("active", btn.dataset.scenario === level));
  dom.difficultyBadge.textContent = level.toUpperCase();
  dom.difficultyHelp.textContent = DIFFICULTY_COPY[level];
  log(`Difficulty set to ${level}.`, "info");
}

async function jsonPost(path, body = {}) {
  const response = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }
  return response.json();
}

async function pollState() {
  if (mode !== "live") return;
  try {
    const response = await fetch(`${API}/state`, { signal: AbortSignal.timeout(2500) });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const scene = await response.json();
    liveScene = scene;
    dom.connectStatus.textContent = "ONLINE";
    dom.connectStatus.classList.add("online");
    dom.connectStatus.classList.remove("offline");
    updateScene(scene);
  } catch (error) {
    dom.connectStatus.textContent = "NO SIGNAL";
    dom.connectStatus.classList.add("offline");
    dom.connectStatus.classList.remove("online");
    if (lastSeenEvent !== "no-signal") {
      log(`Telemetry link lost: ${error.message}`, "err");
      lastSeenEvent = "no-signal";
    }
  }
}

function syncReturnAnimation(scene) {
  const key = `${scene.scenario}:${scene.tick}:${scene.done}`;
  if (scene.done && returnAnimation.key !== key) {
    returnAnimation = { key, startedAt: performance.now() };
  }
  if (!scene.done) {
    returnAnimation = { key: null, startedAt: 0 };
  }
}

function currentScene() {
  if (mode === "replay" && replayFrames.length) {
    return replayFrames[Math.min(replayIndex, replayFrames.length - 1)];
  }
  return liveScene;
}

function getBuildings(scene) {
  return scene?.buildings?.length ? scene.buildings : FALLBACK_BUILDINGS;
}

function getStructure(scene) {
  return scene?.structure || FALLBACK_STRUCTURE;
}

function updateScene(scene) {
  syncReturnAnimation(scene);
  backgroundDirty = true;

  dom.phaseText.textContent = scene.mission_phase;
  dom.cameraText.textContent = scene.camera_status;
  dom.difficultyBadge.textContent = scene.scenario.toUpperCase();
  dom.activeFires.textContent = String(scene.active_fires);
  dom.trappedHumans.textContent = String(scene.trapped_humans);
  dom.rescuedHumans.textContent = String(scene.rescued_humans);
  dom.tickValue.textContent = String(scene.tick);
  dom.scoreValue.textContent = Number(scene.score || 0).toFixed(3);
  dom.lastRewardValue.textContent = Number(scene.last_reward || 0).toFixed(3);
  dom.totalRewardValue.textContent = Number(scene.total_reward || 0).toFixed(3);
  dom.severityValue.textContent = scene.severity.level;
  dom.responseBoostValue.textContent = String(scene.response_boost || 0);
  dom.analysisPercent.textContent = `${Math.round((scene.analysis.progress || 0) * 100)}%`;
  dom.analysisBar.style.width = `${Math.round((scene.analysis.progress || 0) * 100)}%`;
  dom.severityNote.textContent = scene.severity.team_note;
  dom.targetBanner.textContent = `${scene.map_context.station_name} -> ${scene.target_building.name} (${scene.target_building.street})`;
  dom.buildingStatus.textContent = `${scene.scenario} difficulty. ${scene.analysis.status} ${scene.camera_status}`;
  dom.toggleAutoBtn.textContent = `Auto AI: ${scene.auto ? "Active" : "Paused"}`;
  dom.mapMissionNote.textContent = `${scene.map_context.judge_caption}. Dispatch ETA ${scene.map_context.eta_steps} steps.`;

  renderDronePills(scene);
  renderDroneList(scene);
  updateLogs(scene);
  render();
}

function renderDronePills(scene) {
  dom.dronePills.innerHTML = "";
  scene.drones.forEach((drone) => {
    const btn = document.createElement("button");
    btn.className = `dronePill ${selectedDrone === drone.id ? "active" : ""}`;
    btn.textContent = drone.id.toUpperCase();
    btn.addEventListener("click", () => {
      selectedDrone = drone.id;
      dom.selectedDroneText.textContent = `${drone.id.toUpperCase()} selected. ${drone.objective}`;
      renderDronePills(scene);
    });
    dom.dronePills.appendChild(btn);
  });

  if (!selectedDrone && scene.drones.length) {
    selectedDrone = scene.drones[0].id;
    dom.selectedDroneText.textContent = `${selectedDrone.toUpperCase()} selected. ${scene.drones[0].objective}`;
    renderDronePills(scene);
  }
}

function renderDroneList(scene) {
  dom.droneList.innerHTML = "";
  scene.drones.forEach((drone) => {
    const card = document.createElement("div");
    card.className = "droneItem";
    card.innerHTML = `
      <div class="droneItemTop">
        <div>
          <div class="droneName">${drone.id.toUpperCase()}</div>
          <div class="droneMeta">${drone.role} - ${drone.status}</div>
        </div>
        <div class="droneMeta">(${drone.x}, ${drone.y})</div>
      </div>
      <div class="droneMeta">Battery ${drone.battery}% - Payload ${drone.payload}</div>
      <div class="droneMeta">${drone.objective}</div>
    `;
    dom.droneList.appendChild(card);
  });
}

function updateLogs(scene) {
  const latest = scene.events[scene.events.length - 1] || "";
  if (latest && latest !== lastSeenEvent) {
    log(latest, latest.toLowerCase().includes("failed") ? "err" : "info");
    lastSeenEvent = latest;
  }
}

function replayStep() {
  if (mode !== "replay" || !replayFrames.length) return;
  replayIndex += 1;
  if (replayIndex >= replayFrames.length) {
    clearInterval(replayTimer);
    replayTimer = null;
    replayIndex = replayFrames.length - 1;
  }
  updateScene(replayFrames[replayIndex]);
}

function startReplay(frames) {
  if (!frames.length) {
    log("Replay returned no frames.", "warn");
    return;
  }
  mode = "replay";
  replayFrames = frames;
  replayIndex = 0;
  if (replayTimer) clearInterval(replayTimer);
  updateScene(replayFrames[0]);
  replayTimer = setInterval(replayStep, REPLAY_MS);
}

function followLive() {
  mode = "live";
  replayFrames = [];
  replayIndex = 0;
  if (replayTimer) {
    clearInterval(replayTimer);
    replayTimer = null;
  }
  log("Following live mission telemetry.", "info");
  pollState();
}

async function runSimulation() {
  try {
    const payload = await jsonPost("/api/run_ai_simulation", {
      difficulty: selectedScenario,
      scenario: selectedScenario,
    });
    log(`Replay generated for ${payload.scenario} difficulty.`, "info");
    startReplay(payload.frames || []);
  } catch (error) {
    log(`Run simulator failed: ${error.message}`, "err");
  }
}

async function toggleAuto() {
  try {
    const payload = await jsonPost("/api/toggle_auto", {
      difficulty: selectedScenario,
      scenario: selectedScenario,
    });
    log(`Auto AI ${payload.auto ? "enabled" : "paused"}.`, "info");
    mode = "live";
    pollState();
  } catch (error) {
    log(`Auto toggle failed: ${error.message}`, "err");
  }
}

async function triggerAction(path, label) {
  try {
    await jsonPost(path, { difficulty: selectedScenario, scenario: selectedScenario });
    log(label, "info");
    mode = "live";
    pollState();
  } catch (error) {
    log(`${label} failed: ${error.message}`, "err");
  }
}

async function resetDashboard() {
  try {
    await jsonPost("/reset", { difficulty: selectedScenario, scenario: selectedScenario });
    mode = "live";
    replayFrames = [];
    replayIndex = 0;
    backgroundDirty = true;
    log(`Dashboard reset for ${selectedScenario} difficulty.`, "info");
    pollState();
  } catch (error) {
    log(`Reset failed: ${error.message}`, "err");
  }
}

async function manualStep(command) {
  if (!selectedDrone) {
    log("Select a drone before using manual controls.", "warn");
    return;
  }
  try {
    await jsonPost("/step", { drone_id: selectedDrone, command });
    log(`${selectedDrone.toUpperCase()} -> ${command}`, "info");
    mode = "live";
    pollState();
  } catch (error) {
    log(`Manual step failed: ${error.message}`, "err");
  }
}

function computeSceneFrame(width, height, sourceWidth, sourceHeight, padding = 0) {
  const availWidth = Math.max(1, width - padding * 2);
  const availHeight = Math.max(1, height - padding * 2);
  const scale = Math.min(availWidth / sourceWidth, availHeight / sourceHeight);
  const drawWidth = sourceWidth * scale;
  const drawHeight = sourceHeight * scale;
  return {
    x: (width - drawWidth) / 2,
    y: (height - drawHeight) / 2,
    width: drawWidth,
    height: drawHeight,
  };
}

function drawBlueprintGrid(ctx, frame, minor, major, minorAlpha, majorAlpha) {
  ctx.save();
  ctx.beginPath();
  ctx.rect(frame.x, frame.y, frame.width, frame.height);
  ctx.clip();
  ctx.strokeStyle = `rgba(104, 197, 255, ${minorAlpha})`;
  ctx.lineWidth = 1;
  for (let x = frame.x + 0.5; x <= frame.x + frame.width; x += minor) {
    ctx.beginPath();
    ctx.moveTo(x, frame.y);
    ctx.lineTo(x, frame.y + frame.height);
    ctx.stroke();
  }
  for (let y = frame.y + 0.5; y <= frame.y + frame.height; y += minor) {
    ctx.beginPath();
    ctx.moveTo(frame.x, y);
    ctx.lineTo(frame.x + frame.width, y);
    ctx.stroke();
  }

  ctx.strokeStyle = `rgba(104, 214, 255, ${majorAlpha})`;
  for (let x = frame.x + 0.5; x <= frame.x + frame.width; x += major) {
    ctx.beginPath();
    ctx.moveTo(x, frame.y);
    ctx.lineTo(x, frame.y + frame.height);
    ctx.stroke();
  }
  for (let y = frame.y + 0.5; y <= frame.y + frame.height; y += major) {
    ctx.beginPath();
    ctx.moveTo(frame.x, y);
    ctx.lineTo(frame.x + frame.width, y);
    ctx.stroke();
  }
  ctx.restore();
}

function projectPoint(frame, point) {
  return {
    x: frame.x + point.x * frame.width,
    y: frame.y + point.y * frame.height,
  };
}

function stationPointNormalized() {
  return { x: 0.11, y: 0.18 };
}

function targetRect(scene, frame) {
  const rect = scene.target_building.rect;
  return {
    x: frame.x + rect.x * frame.width,
    y: frame.y + rect.y * frame.height,
    w: rect.w * frame.width,
    h: rect.h * frame.height,
  };
}

function routePoint(route, progress) {
  if (!route.length) return [0, 0];
  if (route.length === 1) return route[0];
  const totalSegments = route.length - 1;
  const scaled = Math.min(totalSegments, Math.max(0, progress * totalSegments));
  const index = Math.floor(scaled);
  const t = scaled - index;
  const a = route[index];
  const b = route[Math.min(route.length - 1, index + 1)];
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
}

function dispatchProgress(scene, index) {
  if (!scene.live_armed && scene.tick === 0) return 0;
  if (scene.done) {
    const elapsed = Math.min(1, (performance.now() - returnAnimation.startedAt) / 2200);
    return Math.max(0, 1 - elapsed + index * 0.03);
  }
  if (scene.mission_phase.includes("THERMAL")) return Math.min(1, (scene.tick + 1 + index * 0.35) / 7);
  if (scene.mission_phase.includes("FIRE")) return Math.min(1, 0.78 + index * 0.02);
  return 1;
}

function droneWorldPointNormalized(scene, drone, index) {
  const target = scene.target_building.rect;
  const route = [
    [stationPointNormalized().x, stationPointNormalized().y],
    [0.22, 0.2],
    [0.36, 0.24],
    [target.x + target.w * 0.5, target.y + target.h * 0.5],
  ];
  const progress = dispatchProgress(scene, index);
  if (progress < 1) {
    const [x, y] = routePoint(route, progress);
    return { x, y };
  }
  return {
    x: target.x + target.w * (0.2 + (drone.x % 5) / 6),
    y: target.y + target.h * (0.2 + (drone.y % 5) / 6),
  };
}

function drawRoad(ctx, frame, points, width, label, labelOffset = { x: 0, y: 0 }) {
  const scaled = points.map(([x, y]) => ({ x: frame.x + x * frame.width, y: frame.y + y * frame.height }));
  ctx.save();
  ctx.strokeStyle = "rgba(16, 36, 56, 0.96)";
  ctx.lineWidth = width + 10;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(scaled[0].x, scaled[0].y);
  for (let index = 1; index < scaled.length; index += 1) {
    ctx.lineTo(scaled[index].x, scaled[index].y);
  }
  ctx.stroke();

  ctx.strokeStyle = "rgba(124, 222, 255, 0.26)";
  ctx.lineWidth = width;
  ctx.setLineDash([14, 10]);
  ctx.stroke();
  ctx.restore();

  if (label) {
    const midpoint = scaled[Math.floor(scaled.length / 2)];
    ctx.fillStyle = "rgba(199, 242, 255, 0.82)";
    ctx.font = "600 14px 'Share Tech Mono'";
    ctx.fillText(label, midpoint.x + labelOffset.x, midpoint.y + labelOffset.y);
  }
}

function drawLandmarkBlock(ctx, frame, building, color = "rgba(196, 239, 255, 0.66)") {
  const rect = {
    x: frame.x + building.rect.x * frame.width,
    y: frame.y + building.rect.y * frame.height,
    w: building.rect.w * frame.width,
    h: building.rect.h * frame.height,
  };
  ctx.save();
  ctx.fillStyle = "rgba(11, 30, 47, 0.9)";
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.setLineDash([8, 5]);
  ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
  ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
  ctx.restore();

  ctx.fillStyle = "rgba(229, 248, 255, 0.9)";
  ctx.font = "700 16px Rajdhani";
  ctx.fillText(building.name.toUpperCase(), rect.x + 10, rect.y + rect.h / 2);
}

function drawMapBackdrop(scene) {
  const width = dom.mapBackdropCanvas.clientWidth;
  const height = dom.mapBackdropCanvas.clientHeight;
  if (!width || !height) return;

  const frame = computeSceneFrame(width, height, MAP_BLUEPRINT.width, MAP_BLUEPRINT.height, 18);
  const buildings = getBuildings(scene);
  mapBgCtx.clearRect(0, 0, width, height);

  const stageGradient = mapBgCtx.createLinearGradient(0, 0, 0, height);
  stageGradient.addColorStop(0, "#05111f");
  stageGradient.addColorStop(1, "#071423");
  mapBgCtx.fillStyle = stageGradient;
  mapBgCtx.fillRect(0, 0, width, height);

  const mapGradient = mapBgCtx.createLinearGradient(frame.x, frame.y, frame.x + frame.width, frame.y + frame.height);
  mapGradient.addColorStop(0, "#0a1f35");
  mapGradient.addColorStop(1, "#091729");
  mapBgCtx.fillStyle = mapGradient;
  mapBgCtx.fillRect(frame.x, frame.y, frame.width, frame.height);
  drawBlueprintGrid(mapBgCtx, frame, frame.width / 34, frame.width / 8.5, 0.08, 0.16);

  drawRoad(mapBgCtx, frame, [[0.06, 0.15], [0.94, 0.15]], 18, "OAKWOOD AVE", { x: -38, y: -10 });
  drawRoad(mapBgCtx, frame, [[0.08, 0.43], [0.92, 0.43]], 20, "CENTRAL BLVD", { x: -40, y: -10 });
  drawRoad(mapBgCtx, frame, [[0.11, 0.64], [0.9, 0.64]], 18, "MARKET ST", { x: -38, y: -10 });
  drawRoad(mapBgCtx, frame, [[0.24, 0.07], [0.24, 0.93]], 16, "ALPHA GRID", { x: 12, y: 6 });
  drawRoad(mapBgCtx, frame, [[0.53, 0.07], [0.53, 0.93]], 16, "BRAVO GRID", { x: 12, y: 6 });
  drawRoad(mapBgCtx, frame, [[0.8, 0.08], [0.8, 0.92]], 16, "CHARLIE GRID", { x: 12, y: 6 });

  mapBgCtx.save();
  mapBgCtx.strokeStyle = "rgba(111, 214, 255, 0.32)";
  mapBgCtx.fillStyle = "rgba(23, 51, 67, 0.76)";
  mapBgCtx.beginPath();
  mapBgCtx.ellipse(frame.x + frame.width * 0.48, frame.y + frame.height * 0.48, frame.width * 0.14, frame.height * 0.12, 0, 0, Math.PI * 2);
  mapBgCtx.fill();
  mapBgCtx.stroke();
  mapBgCtx.restore();
  mapBgCtx.fillStyle = "rgba(219, 245, 255, 0.86)";
  mapBgCtx.font = "700 22px Rajdhani";
  mapBgCtx.fillText("CENTRAL PARK", frame.x + frame.width * 0.41, frame.y + frame.height * 0.49);

  const station = projectPoint(frame, stationPointNormalized());
  mapBgCtx.save();
  mapBgCtx.fillStyle = "rgba(10, 40, 53, 0.98)";
  mapBgCtx.strokeStyle = "rgba(115, 255, 214, 0.78)";
  mapBgCtx.lineWidth = 2;
  mapBgCtx.beginPath();
  mapBgCtx.moveTo(station.x, station.y - 22);
  mapBgCtx.lineTo(station.x + 24, station.y - 6);
  mapBgCtx.lineTo(station.x + 24, station.y + 20);
  mapBgCtx.lineTo(station.x - 24, station.y + 20);
  mapBgCtx.lineTo(station.x - 24, station.y - 6);
  mapBgCtx.closePath();
  mapBgCtx.fill();
  mapBgCtx.stroke();
  mapBgCtx.restore();
  mapBgCtx.fillStyle = "rgba(230, 252, 240, 0.96)";
  mapBgCtx.font = "700 16px Rajdhani";
  mapBgCtx.fillText("FIRE STATION", station.x - 38, station.y + 42);

  buildings.forEach((building) => drawLandmarkBlock(mapBgCtx, frame, building));

  const extraBlocks = [
    { x: 0.08, y: 0.28, w: 0.11, h: 0.11, label: "CIVIC PLAZA" },
    { x: 0.31, y: 0.74, w: 0.14, h: 0.1, label: "RIVERSIDE APARTMENTS" },
    { x: 0.57, y: 0.74, w: 0.14, h: 0.1, label: "SUPPLY YARD" },
  ];
  extraBlocks.forEach((block) => {
    const rect = {
      x: frame.x + block.x * frame.width,
      y: frame.y + block.y * frame.height,
      w: block.w * frame.width,
      h: block.h * frame.height,
    };
    mapBgCtx.strokeStyle = "rgba(134, 210, 245, 0.28)";
    mapBgCtx.lineWidth = 2;
    mapBgCtx.strokeRect(rect.x, rect.y, rect.w, rect.h);
    mapBgCtx.fillStyle = "rgba(9, 22, 36, 0.7)";
    mapBgCtx.fillRect(rect.x, rect.y, rect.w, rect.h);
    mapBgCtx.fillStyle = "rgba(216, 243, 255, 0.72)";
    mapBgCtx.font = "600 13px 'Share Tech Mono'";
    mapBgCtx.fillText(block.label, rect.x + 8, rect.y + rect.h / 2);
  });

  mapBgCtx.strokeStyle = "rgba(132, 221, 255, 0.28)";
  mapBgCtx.lineWidth = 2;
  mapBgCtx.strokeRect(frame.x + 1, frame.y + 1, frame.width - 2, frame.height - 2);

  mapBgCtx.fillStyle = "rgba(208, 243, 255, 0.7)";
  mapBgCtx.font = "600 12px 'Share Tech Mono'";
  for (let index = 0; index <= 10; index += 1) {
    mapBgCtx.fillText(`X${index}`, frame.x + 24 + (frame.width - 48) * (index / 10), frame.y + 18);
    mapBgCtx.fillText(`Y${index}`, frame.x + 8, frame.y + 34 + (frame.height - 68) * (index / 10));
  }

  mapBgCtx.save();
  mapBgCtx.translate(frame.x + 38, frame.y + frame.height - 72);
  mapBgCtx.strokeStyle = "rgba(215, 247, 255, 0.88)";
  mapBgCtx.fillStyle = "rgba(215, 247, 255, 0.88)";
  mapBgCtx.lineWidth = 2;
  mapBgCtx.beginPath();
  mapBgCtx.moveTo(0, 44);
  mapBgCtx.lineTo(12, 12);
  mapBgCtx.lineTo(24, 44);
  mapBgCtx.closePath();
  mapBgCtx.fill();
  mapBgCtx.beginPath();
  mapBgCtx.moveTo(12, 14);
  mapBgCtx.lineTo(12, 62);
  mapBgCtx.stroke();
  mapBgCtx.font = "700 18px Rajdhani";
  mapBgCtx.fillText("N", 6, 84);
  mapBgCtx.restore();
}

function drawTargetGlow(ctx, rect, pulse) {
  const glow = ctx.createRadialGradient(
    rect.x + rect.w / 2,
    rect.y + rect.h / 2,
    Math.min(rect.w, rect.h) * 0.1,
    rect.x + rect.w / 2,
    rect.y + rect.h / 2,
    Math.max(rect.w, rect.h) * 0.8,
  );
  glow.addColorStop(0, "rgba(255, 214, 110, 0.18)");
  glow.addColorStop(1, "rgba(255, 214, 110, 0)");
  ctx.fillStyle = glow;
  ctx.fillRect(rect.x - 24, rect.y - 24, rect.w + 48, rect.h + 48);

  ctx.save();
  ctx.strokeStyle = `rgba(255, 214, 110, ${0.62 + pulse * 0.22})`;
  ctx.lineWidth = 2.5;
  ctx.setLineDash([10, 7]);
  ctx.lineDashOffset = -pulse * 24;
  ctx.strokeRect(rect.x - 2.5, rect.y - 2.5, rect.w + 5, rect.h + 5);
  ctx.restore();

  ctx.fillStyle = "rgba(255, 214, 110, 0.08)";
  ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
}

function drawFireMarkers(ctx, rect, fireCount, pulse) {
  if (!fireCount) return;
  const hotspots = [
    [0.28, 0.32],
    [0.58, 0.38],
    [0.42, 0.68],
    [0.74, 0.22],
    [0.78, 0.6],
  ];
  const visible = Math.min(hotspots.length, Math.max(1, fireCount));
  for (let index = 0; index < visible; index += 1) {
    const [dx, dy] = hotspots[index];
    const cx = rect.x + rect.w * dx;
    const cy = rect.y + rect.h * dy;
    const radius = 16 + index * 2 + pulse * 8;
    const gradient = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);
    gradient.addColorStop(0, `rgba(255, 92, 92, ${0.44 + pulse * 0.12})`);
    gradient.addColorStop(0.45, `rgba(255, 92, 92, ${0.18 + pulse * 0.08})`);
    gradient.addColorStop(1, "rgba(255, 92, 92, 0)");
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "#ff6f6f";
    ctx.beginPath();
    ctx.arc(cx, cy, 5 + pulse * 2, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawRouteLine(ctx, points, color, lineWidth, dash, alpha = 1) {
  if (points.length < 2) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.globalAlpha = alpha;
  ctx.lineWidth = lineWidth;
  ctx.setLineDash(dash);
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let index = 1; index < points.length; index += 1) {
    ctx.lineTo(points[index].x, points[index].y);
  }
  ctx.stroke();
  ctx.restore();
}

function drawDroneMarker(ctx, point, emphasis, label, pulse) {
  const halo = emphasis ? 18 + pulse * 5 : 11 + pulse * 3;
  const gradient = ctx.createRadialGradient(point.x, point.y, 0, point.x, point.y, halo);
  gradient.addColorStop(0, emphasis ? "rgba(84, 215, 255, 0.52)" : "rgba(84, 215, 255, 0.3)");
  gradient.addColorStop(1, "rgba(84, 215, 255, 0)");
  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(point.x, point.y, halo, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "#54d7ff";
  ctx.beginPath();
  ctx.arc(point.x, point.y, emphasis ? 6 : 4.5, 0, Math.PI * 2);
  ctx.fill();

  ctx.strokeStyle = emphasis ? "rgba(255,255,255,0.95)" : "rgba(196, 242, 255, 0.72)";
  ctx.lineWidth = emphasis ? 2 : 1;
  ctx.beginPath();
  ctx.arc(point.x, point.y, emphasis ? 9 : 7, 0, Math.PI * 2);
  ctx.stroke();

  ctx.fillStyle = "rgba(232, 247, 255, 0.92)";
  ctx.font = emphasis ? "700 11px Rajdhani" : "600 10px Rajdhani";
  ctx.fillText(label, point.x + 10, point.y - 10);
}

function drawMap(scene) {
  const width = dom.mapCanvas.clientWidth;
  const height = dom.mapCanvas.clientHeight;
  if (!width || !height || !scene) return;

  const frame = computeSceneFrame(width, height, MAP_BLUEPRINT.width, MAP_BLUEPRINT.height, 18);
  const pulse = 0.5 + 0.5 * Math.sin(performance.now() / 260);

  mapCtx.clearRect(0, 0, width, height);

  const station = projectPoint(frame, stationPointNormalized());
  const target = targetRect(scene, frame);
  const targetCenter = { x: target.x + target.w * 0.5, y: target.y + target.h * 0.5 };

  drawTargetGlow(mapCtx, target, pulse);
  drawFireMarkers(mapCtx, target, scene.active_fires, pulse);

  mapCtx.fillStyle = "#63f1b8";
  mapCtx.beginPath();
  mapCtx.arc(station.x, station.y, 8, 0, Math.PI * 2);
  mapCtx.fill();
  mapCtx.fillStyle = "#dbfff0";
  mapCtx.font = "700 12px Rajdhani";
  mapCtx.fillText("Station", station.x + 12, station.y + 4);

  drawRouteLine(
    mapCtx,
    [station, { x: target.x + target.w * 0.18, y: target.y + target.h * 0.22 }, targetCenter],
    "rgba(159, 216, 255, 0.8)",
    2,
    [10, 8],
    0.85,
  );

  mapCtx.fillStyle = "#fff0bf";
  mapCtx.font = "700 14px Rajdhani";
  mapCtx.fillText(scene.target_building.name, target.x + 10, Math.max(18, target.y - 8));

  scene.drones.forEach((drone, index) => {
    const logical = droneWorldPointNormalized(scene, drone, index);
    const point = projectPoint(frame, logical);
    const emphasis = selectedDrone ? drone.id === selectedDrone : index === 0;
    drawRouteLine(
      mapCtx,
      [point, targetCenter],
      emphasis ? "rgba(84, 215, 255, 0.78)" : "rgba(84, 215, 255, 0.24)",
      emphasis ? 2.2 : 1.2,
      emphasis ? [8, 6] : [4, 8],
      emphasis ? 1 : 0.9,
    );
    drawDroneMarker(mapCtx, point, emphasis, drone.id.toUpperCase(), pulse);
  });
}

function scenarioPalette(scenario) {
  if (scenario === "Easy") {
    return { route: "#7cffc9", reveal: "rgba(84, 215, 255, 0.08)", border: "rgba(118, 225, 255, 0.28)" };
  }
  if (scenario === "Hard") {
    return { route: "#93ffc2", reveal: "rgba(255, 112, 112, 0.07)", border: "rgba(255, 112, 112, 0.2)" };
  }
  return { route: "#71ffd5", reveal: "rgba(84, 215, 255, 0.09)", border: "rgba(84, 215, 255, 0.22)" };
}

function buildInteriorGeometry(scene, width, height) {
  const structure = getStructure(scene);
  const [x0, y0, x1, y1] = structure.bounds;
  const cols = x1 - x0 + 1;
  const rows = y1 - y0 + 1;
  const paddingX = 22;
  const paddingY = 22;
  const cell = Math.min((width - paddingX * 2) / cols, (height - paddingY * 2) / rows);
  const drawWidth = cols * cell;
  const drawHeight = rows * cell;
  const originX = (width - drawWidth) / 2;
  const originY = (height - drawHeight) / 2;
  return { x0, y0, x1, y1, cols, rows, cell, originX, originY, drawWidth, drawHeight };
}

function cellPoint(geometry, logicalX, logicalY) {
  return {
    x: geometry.originX + (logicalX - geometry.x0) * geometry.cell,
    y: geometry.originY + (logicalY - geometry.y0) * geometry.cell,
  };
}

function cellCenter(geometry, logicalX, logicalY) {
  const point = cellPoint(geometry, logicalX, logicalY);
  return { x: point.x + geometry.cell / 2, y: point.y + geometry.cell / 2 };
}

function buildWallSet(scene) {
  const structure = getStructure(scene);
  const walls = new Set((structure.interior_walls || []).map(([x, y]) => `${x},${y}`));
  const [x0, y0, x1, y1] = structure.bounds;
  const exit = scene?.exit || { x: 5, y: 7 };
  for (let x = x0; x <= x1; x += 1) {
    walls.add(`${x},${y0}`);
    walls.add(`${x},${y1}`);
  }
  for (let y = y0; y <= y1; y += 1) {
    walls.add(`${x0},${y}`);
    walls.add(`${x1},${y}`);
  }
  walls.delete(`${exit.x},${exit.y}`);
  return walls;
}

function buildCorridorSet(scene, geometry) {
  const cells = new Set();
  const exit = scene?.exit || { x: 5, y: 7 };
  for (let x = exit.x; x <= geometry.x1 - 1; x += 1) {
    cells.add(`${x},${exit.y}`);
  }
  for (let y = geometry.y0 + 1; y <= geometry.y1 - 1; y += 1) {
    cells.add(`${geometry.x0 + 3},${y}`);
  }
  for (let x = geometry.x0 + 2; x <= geometry.x1 - 2; x += 1) {
    cells.add(`${x},${geometry.y0 + 3}`);
  }
  return cells;
}

function drawRoomLabel(ctx, geometry, logicalX, logicalY, text) {
  const center = cellCenter(geometry, logicalX, logicalY);
  ctx.fillStyle = "rgba(212, 244, 255, 0.78)";
  ctx.font = "700 12px 'Share Tech Mono'";
  ctx.textAlign = "center";
  ctx.fillText(text, center.x, center.y);
  ctx.textAlign = "left";
}

function drawInteriorBackdrop(scene) {
  const width = dom.interiorBackdropCanvas.clientWidth;
  const height = dom.interiorBackdropCanvas.clientHeight;
  if (!width || !height) return;

  const palette = scenarioPalette(scene?.scenario || selectedScenario);
  const geometry = buildInteriorGeometry(scene, width, height);
  const wallSet = buildWallSet(scene);
  const corridorSet = buildCorridorSet(scene, geometry);
  const exit = scene?.exit || { x: 5, y: 7 };

  interiorBgCtx.clearRect(0, 0, width, height);
  const bg = interiorBgCtx.createLinearGradient(0, 0, 0, height);
  bg.addColorStop(0, "#061120");
  bg.addColorStop(1, "#081521");
  interiorBgCtx.fillStyle = bg;
  interiorBgCtx.fillRect(0, 0, width, height);

  const frame = computeSceneFrame(width, height, INTERIOR_BLUEPRINT.width, INTERIOR_BLUEPRINT.height, 12);
  drawBlueprintGrid(interiorBgCtx, frame, frame.width / 24, frame.width / 8, 0.06, 0.12);

  for (let y = geometry.y0; y <= geometry.y1; y += 1) {
    for (let x = geometry.x0; x <= geometry.x1; x += 1) {
      const point = cellPoint(geometry, x, y);
      const key = `${x},${y}`;
      const isWall = wallSet.has(key);
      const isCorridor = corridorSet.has(key);

      interiorBgCtx.fillStyle = isWall
        ? "rgba(214, 240, 255, 0.1)"
        : isCorridor
        ? "rgba(11, 49, 66, 0.72)"
        : "rgba(8, 27, 44, 0.92)";
      interiorBgCtx.fillRect(point.x, point.y, geometry.cell - 1, geometry.cell - 1);

      interiorBgCtx.strokeStyle = isWall ? "rgba(218, 243, 255, 0.5)" : palette.border;
      interiorBgCtx.lineWidth = isWall ? 2 : 1;
      interiorBgCtx.strokeRect(point.x + 0.5, point.y + 0.5, geometry.cell - 2, geometry.cell - 2);
    }
  }

  const exitPoint = cellPoint(geometry, exit.x, exit.y);
  interiorBgCtx.fillStyle = "rgba(99, 241, 184, 0.14)";
  interiorBgCtx.fillRect(exitPoint.x, exitPoint.y, geometry.cell - 1, geometry.cell - 1);
  interiorBgCtx.strokeStyle = "rgba(99, 241, 184, 0.88)";
  interiorBgCtx.lineWidth = 2;
  interiorBgCtx.strokeRect(exitPoint.x + 1.5, exitPoint.y + 1.5, geometry.cell - 4, geometry.cell - 4);

  drawRoomLabel(interiorBgCtx, geometry, geometry.x0 + 1.5, geometry.y0 + 1.4, "LOBBY");
  drawRoomLabel(interiorBgCtx, geometry, geometry.x0 + 5.6, geometry.y0 + 1.6, "CONTROL");
  drawRoomLabel(interiorBgCtx, geometry, geometry.x0 + 1.8, geometry.y0 + 4.6, "STAIRS");
  drawRoomLabel(interiorBgCtx, geometry, geometry.x0 + 5.8, geometry.y0 + 4.8, "ARCHIVE");
  drawRoomLabel(interiorBgCtx, geometry, geometry.x0 + 1.7, geometry.y0 + 7.3, "OFFICE A");
  drawRoomLabel(interiorBgCtx, geometry, geometry.x0 + 5.7, geometry.y0 + 7.2, "MED BAY");

  interiorBgCtx.fillStyle = "rgba(210, 247, 255, 0.85)";
  interiorBgCtx.font = "700 14px Rajdhani";
  interiorBgCtx.fillText("TACTICAL FLOOR PLAN", geometry.originX, geometry.originY - 10);
}

function droneInteriorPosition(scene, geometry, drone, index) {
  const inside =
    drone.x >= geometry.x0 &&
    drone.x <= geometry.x1 &&
    drone.y >= geometry.y0 &&
    drone.y <= geometry.y1;
  if (inside) {
    return { x: drone.x, y: drone.y, inside: true };
  }
  return {
    x: scene.exit.x - 0.75 - (index % 2) * 0.35,
    y: Math.min(geometry.y1 - 0.4, Math.max(geometry.y0 + 0.4, scene.exit.y - 0.7 + Math.floor(index / 2) * 0.42)),
    inside: false,
  };
}

function drawInterior(scene) {
  const width = dom.interiorCanvas.clientWidth;
  const height = dom.interiorCanvas.clientHeight;
  if (!width || !height || !scene) return;

  const pulse = 0.5 + 0.5 * Math.sin(performance.now() / 240);
  const palette = scenarioPalette(scene.scenario);
  const geometry = buildInteriorGeometry(scene, width, height);
  const revealedSet = new Set((scene.analysis.revealed_cells || []).map(([x, y]) => `${x},${y}`));

  interiorCtx.clearRect(0, 0, width, height);

  for (let y = geometry.y0; y <= geometry.y1; y += 1) {
    for (let x = geometry.x0; x <= geometry.x1; x += 1) {
      const key = `${x},${y}`;
      if (!revealedSet.has(key)) continue;
      const point = cellPoint(geometry, x, y);
      interiorCtx.fillStyle = palette.reveal;
      interiorCtx.fillRect(point.x + 2, point.y + 2, geometry.cell - 5, geometry.cell - 5);
    }
  }

  const safePath = scene.analysis.safe_path || [];
  if (safePath.length > 1) {
    interiorCtx.save();
    interiorCtx.strokeStyle = palette.route;
    interiorCtx.lineWidth = Math.max(3, geometry.cell * 0.16);
    interiorCtx.beginPath();
    safePath.forEach(([x, y], index) => {
      const center = cellCenter(geometry, x, y);
      if (index === 0) {
        interiorCtx.moveTo(center.x, center.y);
      } else {
        interiorCtx.lineTo(center.x, center.y);
      }
    });
    interiorCtx.stroke();
    interiorCtx.restore();
  }

  (scene.fire_nodes || []).forEach((node) => {
    const center = cellCenter(geometry, node.x, node.y);
    const radius = Math.max(geometry.cell * 0.24, 7) + node.intensity * 1.7 + pulse * 3;
    const gradient = interiorCtx.createRadialGradient(center.x, center.y, 0, center.x, center.y, radius);
    gradient.addColorStop(0, `rgba(255, 90, 90, ${0.44 + pulse * 0.14})`);
    gradient.addColorStop(0.45, `rgba(255, 90, 90, ${0.18 + pulse * 0.08})`);
    gradient.addColorStop(1, "rgba(255, 90, 90, 0)");
    interiorCtx.fillStyle = gradient;
    interiorCtx.beginPath();
    interiorCtx.arc(center.x, center.y, radius, 0, Math.PI * 2);
    interiorCtx.fill();

    interiorCtx.fillStyle = "#ff6f6f";
    interiorCtx.beginPath();
    interiorCtx.arc(center.x, center.y, Math.max(geometry.cell * 0.12, 4) + pulse * 1.3, 0, Math.PI * 2);
    interiorCtx.fill();
  });

  if (scene.civilian.active) {
    const center = cellCenter(geometry, scene.civilian.x, scene.civilian.y);
    const glow = interiorCtx.createRadialGradient(center.x, center.y, 0, center.x, center.y, geometry.cell * 0.58 + pulse * 6);
    glow.addColorStop(0, "rgba(255, 230, 125, 0.56)");
    glow.addColorStop(1, "rgba(255, 230, 125, 0)");
    interiorCtx.fillStyle = glow;
    interiorCtx.beginPath();
    interiorCtx.arc(center.x, center.y, geometry.cell * 0.58 + pulse * 6, 0, Math.PI * 2);
    interiorCtx.fill();

    interiorCtx.fillStyle = "#ffe68b";
    interiorCtx.beginPath();
    interiorCtx.arc(center.x, center.y, Math.max(geometry.cell * 0.16, 6), 0, Math.PI * 2);
    interiorCtx.fill();
  }

  const exitPoint = cellPoint(geometry, scene.exit.x, scene.exit.y);
  interiorCtx.fillStyle = "rgba(99, 241, 184, 0.16)";
  interiorCtx.fillRect(exitPoint.x + 2, exitPoint.y + 2, geometry.cell - 5, geometry.cell - 5);
  interiorCtx.strokeStyle = "rgba(99, 241, 184, 0.86)";
  interiorCtx.lineWidth = 2;
  interiorCtx.strokeRect(exitPoint.x + 2.5, exitPoint.y + 2.5, geometry.cell - 6, geometry.cell - 6);
  interiorCtx.fillStyle = "#9dffd1";
  interiorCtx.font = `700 ${Math.max(10, geometry.cell * 0.22)}px Rajdhani`;
  interiorCtx.fillText("EXIT", exitPoint.x + geometry.cell * 0.12, exitPoint.y + geometry.cell * 0.62);

  scene.drones.forEach((drone, index) => {
    const placement = droneInteriorPosition(scene, geometry, drone, index);
    const center = cellCenter(geometry, placement.x, placement.y);
    const emphasis = selectedDrone ? drone.id === selectedDrone : index === 0;
    const ring = emphasis ? geometry.cell * 0.34 : geometry.cell * 0.28;
    const gradient = interiorCtx.createRadialGradient(center.x, center.y, 0, center.x, center.y, ring + pulse * 4);
    gradient.addColorStop(0, emphasis ? "rgba(84, 215, 255, 0.48)" : "rgba(84, 215, 255, 0.28)");
    gradient.addColorStop(1, "rgba(84, 215, 255, 0)");
    interiorCtx.fillStyle = gradient;
    interiorCtx.beginPath();
    interiorCtx.arc(center.x, center.y, ring + pulse * 4, 0, Math.PI * 2);
    interiorCtx.fill();

    interiorCtx.fillStyle = "#54d7ff";
    interiorCtx.beginPath();
    interiorCtx.arc(center.x, center.y, Math.max(geometry.cell * 0.13, 5), 0, Math.PI * 2);
    interiorCtx.fill();

    interiorCtx.strokeStyle = emphasis ? "rgba(255,255,255,0.92)" : "rgba(176, 237, 255, 0.66)";
    interiorCtx.lineWidth = emphasis ? 2 : 1.2;
    interiorCtx.beginPath();
    interiorCtx.arc(center.x, center.y, Math.max(geometry.cell * 0.2, 8), 0, Math.PI * 2);
    interiorCtx.stroke();
  });
}

function render() {
  const scene = currentScene();
  if (backgroundDirty) {
    drawMapBackdrop(scene);
    drawInteriorBackdrop(scene);
    backgroundDirty = false;
  }

  mapCtx.clearRect(0, 0, dom.mapCanvas.clientWidth, dom.mapCanvas.clientHeight);
  interiorCtx.clearRect(0, 0, dom.interiorCanvas.clientWidth, dom.interiorCanvas.clientHeight);

  if (!scene) return;
  drawMap(scene);
  drawInterior(scene);
}

function renderLoop() {
  render();
  window.requestAnimationFrame(renderLoop);
}

dom.difficultyButtons.forEach((button) => {
  button.addEventListener("click", () => {
    setDifficulty(button.dataset.scenario);
    backgroundDirty = true;
    render();
  });
});

dom.runSimulationBtn.addEventListener("click", runSimulation);
dom.toggleAutoBtn.addEventListener("click", toggleAuto);
dom.followLiveBtn.addEventListener("click", followLive);
dom.spawnFireBtn.addEventListener("click", () =>
  triggerAction("/api/spawn_fire", "Extra fire dispatched into target area."),
);
dom.spawnHumanBtn.addEventListener("click", () =>
  triggerAction("/api/spawn_human", "Extra rescue request dispatched into target area."),
);
dom.refillBtn.addEventListener("click", () => triggerAction("/api/refill", "All drones refilled."));
dom.resetBtn.addEventListener("click", resetDashboard);

document.querySelectorAll(".dpadBtn").forEach((button) => {
  button.addEventListener("click", () => manualStep(button.dataset.command));
});

window.addEventListener("resize", resizeAll);

(async function bootstrap() {
  setDifficulty("Medium");
  resizeAll();
  render();
  await pollState();
  setInterval(pollState, POLL_MS);
  if (!renderStarted) {
    renderStarted = true;
    window.requestAnimationFrame(renderLoop);
  }
})();
