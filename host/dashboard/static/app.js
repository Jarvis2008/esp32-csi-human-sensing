const MAX_POINTS = 120;
const motionSeries = [];
const confidenceSeries = [];

const elements = {
  badge: document.getElementById("healthBadge"),
  presence: document.getElementById("presenceValue"),
  activity: document.getElementById("activityValue"),
  confidence: document.getElementById("confidenceValue"),
  received: document.getElementById("packetsReceived"),
  parsed: document.getElementById("packetsParsed"),
  invalid: document.getElementById("packetsInvalid"),
  dropped: document.getElementById("droppedEstimate"),
  errorRate: document.getElementById("parserErrorRate"),
  nodeList: document.getElementById("nodeList"),
  motionCanvas: document.getElementById("motionChart"),
  confidenceCanvas: document.getElementById("confidenceChart"),
  activeLabel: document.getElementById("activeLabel"),
  labelButtons: Array.from(document.querySelectorAll("#labelButtons .label-btn")),
};

function setBadge(kind, text) {
  elements.badge.classList.remove("badge-ok", "badge-warn", "badge-bad");
  elements.badge.classList.add(kind);
  elements.badge.textContent = text;
}

function renderNodes(nodes) {
  const keys = Object.keys(nodes || {});
  if (keys.length === 0) {
    elements.nodeList.textContent = "No node packets yet.";
    return;
  }

  elements.nodeList.innerHTML = keys
    .sort((a, b) => Number(a) - Number(b))
    .map((id) => {
      const n = nodes[id];
      return `node ${id}: packets=${n.packets} last_seq=${n.last_seq} drop=${n.dropped_estimate}`;
    })
    .join("<br />");
}

function pushSeries(series, value) {
  series.push(Number.isFinite(value) ? value : 0);
  if (series.length > MAX_POINTS) {
    series.shift();
  }
}

function drawSeries(canvas, series, color, maxFallback = 1) {
  const ctx = canvas.getContext("2d");
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);

  ctx.strokeStyle = "rgba(135, 161, 181, 0.25)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, height - 20);
  ctx.lineTo(width, height - 20);
  ctx.stroke();

  if (series.length < 2) return;

  const max = Math.max(maxFallback, ...series);
  const min = Math.min(0, ...series);
  const span = Math.max(1e-6, max - min);

  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  series.forEach((v, i) => {
    const x = (i / (MAX_POINTS - 1)) * width;
    const y = height - 20 - ((v - min) / span) * (height - 40);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function setActiveLabelDisplay(label) {
  const normalized = label || "unlabeled";
  elements.activeLabel.textContent = normalized;
  elements.labelButtons.forEach((btn) => {
    if (btn.dataset.label === normalized) {
      btn.classList.add("is-active");
    } else {
      btn.classList.remove("is-active");
    }
  });
}

async function setLabel(label) {
  try {
    const res = await fetch("/api/labels/current", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label }),
    });
    if (!res.ok) return;
    const payload = await res.json();
    setActiveLabelDisplay(payload.active_label || label);
  } catch (_err) {
    // ignore and keep old label
  }
}

async function loadLabelState() {
  try {
    const res = await fetch("/api/labels");
    if (!res.ok) return;
    const payload = await res.json();
    setActiveLabelDisplay(payload.active_label || "unlabeled");
  } catch (_err) {
    // ignore and keep defaults
  }
}

function updateFromPayload(payload) {
  const state = payload.state || {};
  const metrics = payload.metrics || {};
  const features = payload.features || {};
  const labeling = payload.labeling || {};

  elements.presence.textContent = state.presence || "unknown";
  elements.activity.textContent = state.activity || "unknown";
  elements.confidence.textContent = Number(state.confidence || 0).toFixed(3);

  elements.received.textContent = String(metrics.packets_received ?? 0);
  elements.parsed.textContent = String(metrics.packets_parsed ?? 0);
  elements.invalid.textContent = String(metrics.packets_invalid ?? 0);
  elements.dropped.textContent = String(metrics.dropped_estimate ?? 0);
  elements.errorRate.textContent = Number(metrics.error_rate ?? 0).toFixed(3);

  renderNodes(payload.nodes || {});

  pushSeries(motionSeries, Number(features.motion_index ?? 0));
  pushSeries(confidenceSeries, Number(state.confidence ?? 0));

  drawSeries(elements.motionCanvas, motionSeries, "#4fd1c5", 5);
  drawSeries(elements.confidenceCanvas, confidenceSeries, "#ffb84c", 1);

  if ((metrics.packets_invalid ?? 0) > 0 && (metrics.error_rate ?? 0) > 0.1) {
    setBadge("badge-bad", "degraded");
  } else if ((metrics.packets_parsed ?? 0) > 0) {
    setBadge("badge-ok", "healthy");
  } else {
    setBadge("badge-warn", "waiting-data");
  }

  setActiveLabelDisplay(labeling.active_label || elements.activeLabel.textContent);
}

async function pollStatus() {
  try {
    const res = await fetch("/api/status");
    if (!res.ok) return;
    const status = await res.json();
    if (status.packets_parsed > 0) {
      setBadge("badge-ok", "healthy");
    }
  } catch (_err) {
    setBadge("badge-bad", "api-down");
  }
}

function connectWs() {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${scheme}://${window.location.host}/ws/live`);

  ws.onopen = () => {
    setBadge("badge-warn", "connected");
    ws.send("hello");
  };

  ws.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    updateFromPayload(payload);
  };

  ws.onclose = () => {
    setBadge("badge-warn", "reconnecting");
    setTimeout(connectWs, 1200);
  };

  ws.onerror = () => {
    ws.close();
  };
}

setInterval(pollStatus, 3000);
pollStatus();
loadLabelState();
elements.labelButtons.forEach((btn) => {
  btn.addEventListener("click", () => setLabel(btn.dataset.label || "unlabeled"));
});
connectWs();
