"use strict";

const MODEL_IDS = ["logistic_regression", "random_forest", "xgboost", "cnn_1d"];
const MODEL_NAMES = {
  logistic_regression: "Logistic Regression",
  random_forest: "Random Forest",
  xgboost: "XGBoost",
  cnn_1d: "1D CNN",
};
const MODEL_COLORS = {
  logistic_regression: "#087c72",
  random_forest: "#3d7085",
  xgboost: "#ba7827",
  cnn_1d: "#a9473e",
};
const CHARTS = [
  ["model_curves.png", "ROC and precision-recall"],
  ["model_forest.png", "Discrimination intervals"],
  ["calibration.png", "Calibration"],
  ["decision_curve.png", "Decision curve"],
];
const jobStartedAt = new Map();
let currentJob = { state: "idle", active_model: null };

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[char]);
}

function formatMetric(value, digits = 3) {
  return value === null || value === undefined || !Number.isFinite(Number(value))
    ? "—"
    : Number(value).toFixed(digits);
}

function formatElapsed(seconds) {
  const value = Math.max(0, Math.floor(seconds));
  return `${Math.floor(value / 60)}:${String(value % 60).padStart(2, "0")}`;
}

function renderSnapshot(snapshot) {
  const root = document.getElementById("study-summary");
  const cohort = snapshot.cohort ?? {};
  const config = snapshot.config ?? {};
  const design = config.design ?? {};
  const values = [
    ["Patients", cohort.eligible_patients ?? config.n_patients],
    ["Positive outcomes", cohort.positive_outcomes],
    ["Horizon", design.horizon_hours ? `${design.horizon_hours} h` : null],
    ["Fit / validation / test", config.n_fit !== undefined
      ? `${config.n_fit} / ${config.n_validation} / ${config.n_test}` : null],
    ["Seed", config.random_state],
  ];
  root.innerHTML = values.map(([label, value]) =>
    `<div class="summary-item"><strong>${escapeHtml(value ?? "—")}</strong><span>${escapeHtml(label)}</span></div>`,
  ).join("");
}

function reviewText(model, result) {
  const evaluation = result.evaluation ?? {};
  const metrics = result.metrics ?? {};
  const features = result.leading_features ?? [];
  const score = `AUPRC ${formatMetric(evaluation.auprc ?? metrics.auprc_average_precision)}, AUROC ${formatMetric(evaluation.auroc ?? metrics.auroc)}.`;
  const operation = evaluation.operating_point ?? {};
  const thresholdSensitivity = Number(result.target_sensitivity ?? 0.8);
  const observedSensitivity = operation.sensitivity ?? metrics.sensitivity_recall;
  let text;
  if (model === "logistic_regression") {
    text = "Its additive decision function tests whether engineered levels and trends are sufficient to rank patients.";
  } else if (model === "random_forest") {
    text = "Its bagged trees can represent nonlinear cut-points and interactions; compare its ranking with the linear baseline to see whether those splits add signal.";
  } else if (model === "xgboost") {
    text = "Boosted trees iteratively fit residual errors and can exploit nonlinear combinations; compare its held-out ranking and calibration with the other tabular model.";
  } else {
    text = `The CNN learns from ordered values and missingness masks. It trained for ${result.config?.cnn_epochs ?? "the configured number of"} epoch(s); representation and architecture both differ from tabular models.`;
  }
  if (features.length) {
    text += ` Leading fit-set associations: ${features.map((feature) => feature.feature).join(", ")}. These are not causal explanations.`;
  } else if (model === "cnn_1d") {
    text += " No sequence-level feature attribution was computed.";
  }
  text += ` ${score} Test sensitivity was ${formatMetric(observedSensitivity)} versus a validation target of ${Math.round(thresholdSensitivity * 100)}%.`;
  if (evaluation.calibration?.calibration_slope !== undefined) {
    text += ` Calibration slope ${formatMetric(evaluation.calibration.calibration_slope, 2)}; Brier ${formatMetric(evaluation.calibration.brier)}.`;
  }
  if (result.synthetic_demo) {
    text += " Synthetic data have constructed label-linked trends; these metrics are workflow smoke-test results, not clinical evidence.";
  }
  return text;
}

function modelPanel(model, result = {}) {
  const metrics = result.metrics ?? {};
  const evaluation = result.evaluation ?? {};
  const operation = evaluation.operating_point ?? {};
  const running = currentJob.state === "running";
  const started = jobStartedAt.get(model);
  const completedElapsed = result.status?.elapsed_seconds;
  const elapsed = started
    ? formatElapsed((Date.now() - started) / 1000)
    : formatElapsed(result.status?.elapsed_seconds ?? 0);
  const state = result.status?.state ?? (result.available ? "complete" : "idle");
  const stateLabel = state === "running" ? `Running · ${elapsed}`
    : state === "failed" ? "Run failed"
      : state === "complete" ? `Complete${completedElapsed === null || completedElapsed === undefined ? "" : ` · ${formatElapsed(completedElapsed)}`}`
        : "Not run yet";
  const fields = [
    ["AUPRC", evaluation.auprc ?? metrics.auprc_average_precision],
    ["AUROC", evaluation.auroc ?? metrics.auroc],
    ["Sensitivity", operation.sensitivity ?? metrics.sensitivity_recall],
    ["Specificity", operation.specificity ?? metrics.specificity],
    ["Brier", evaluation.calibration?.brier ?? metrics.brier_score],
    ["Threshold", operation.threshold ?? metrics.threshold],
  ];
  const charts = result.isolated_run
    ? CHARTS.map(([file, label]) => `<figure class="chart"><img loading="lazy" src="/assets/${model}/${file}?v=${encodeURIComponent(result.updated_at ?? "0")}" alt="${escapeHtml(label)} for ${escapeHtml(MODEL_NAMES[model])}"><figcaption>${escapeHtml(label)}</figcaption></figure>`).join("")
    : `<p class="empty">${result.available ? "Run this model to generate its dedicated charts." : "Metrics and charts appear here after this model completes a run."}</p>`;
  const error = result.status?.state === "failed"
    ? `<p class="notice">${escapeHtml(result.status.logs ?? "Pipeline command failed")}</p>` : "";
  return `<article class="model-panel" style="--model-color:${MODEL_COLORS[model]}" data-model="${model}">
    <h3>${escapeHtml(MODEL_NAMES[model])}</h3>
    <div class="actions"><button type="button" data-run-model="${model}" ${running ? "disabled" : ""}>Run ${escapeHtml(MODEL_NAMES[model])}</button><span class="job-status" aria-live="polite">${escapeHtml(stateLabel)}</span></div>
    ${error}
    <div class="metrics">${fields.map(([label, value]) => `<div class="metric"><strong>${formatMetric(value)}</strong><span>${escapeHtml(label)}</span></div>`).join("")}</div>
    <p class="review">${escapeHtml(reviewText(model, result))}</p>
    <div class="chart-grid">${charts}</div>
  </article>`;
}

async function refreshResults() {
  const response = await fetch("/api/results", { cache: "no-store" });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error ?? "Could not load pipeline results");
  if (payload.datasource_description) {
    document.getElementById("datasource-description").textContent = payload.datasource_description;
  }
  currentJob = payload.job_status ?? { state: "idle", active_model: null };
  renderSnapshot(payload.snapshot ?? {});
  document.getElementById("model-grid").innerHTML = MODEL_IDS
    .map((model) => modelPanel(model, payload.models?.[model] ?? {})).join("");
  document.querySelectorAll("[data-run-model]").forEach((button) => {
    button.addEventListener("click", () => runModel(button.dataset.runModel));
  });
}

async function runModel(model) {
  const response = await fetch(`/api/run/${model}`, { method: "POST" });
  const payload = await response.json();
  if (!response.ok) {
    window.alert(payload.error ?? "Another model is already running.");
    return;
  }
  jobStartedAt.set(model, Date.now());
  currentJob = { state: "running", active_model: model };
  await refreshResults();
  await pollJob(model);
}

async function pollJob(model) {
  const response = await fetch("/api/status", { cache: "no-store" });
  const status = await response.json();
  if (status.active_model === model && status.state === "running") {
    await refreshResults();
    window.setTimeout(() => pollJob(model), 1000);
    return;
  }
  jobStartedAt.delete(model);
  await refreshResults();
}

refreshResults().catch((error) => {
  document.getElementById("model-grid").innerHTML = `<p class="empty">${escapeHtml(error.message)}</p>`;
});