const shell = document.querySelector(".shell");
const API_BASE = shell.dataset.apiBase;

const apiDot = document.getElementById("apiDot");
const apiStatus = document.getElementById("apiStatus");
const updatedAt = document.getElementById("updatedAt");
const metrics = document.getElementById("metrics");
const newsGrid = document.getElementById("newsGrid");
const predictionGrid = document.getElementById("predictionGrid");
const refreshNews = document.getElementById("refreshNews");
let tomorrowGrid = document.getElementById("tomorrowGrid");

if (!tomorrowGrid && newsGrid) {
  const head = document.createElement("section");
  head.className = "section-head";
  head.innerHTML = `
    <div>
      <p class="eyebrow">Tomorrow Forecast</p>
      <h2>최신뉴스 기반 내일 주가 예측</h2>
    </div>
  `;
  tomorrowGrid = document.createElement("section");
  tomorrowGrid.className = "tomorrow-grid";
  tomorrowGrid.id = "tomorrowGrid";
  newsGrid.parentNode.insertBefore(tomorrowGrid, newsGrid);
  newsGrid.parentNode.insertBefore(head, tomorrowGrid);
}

function pct(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function labelDirection(value) {
  return Number(value) === 1 ? "상승" : "하락";
}

function setApiState(ok, message) {
  apiDot.classList.toggle("ok", ok);
  apiDot.classList.toggle("bad", !ok);
  apiStatus.textContent = message;
}

async function getJson(path) {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

function renderMetrics(summary) {
  const cards = [];
  for (const [key, payload] of Object.entries(summary)) {
    const report = payload.report || {};
    const label = key === "koreanair" ? "대한항공" : "LG화학";
    cards.push(`
      <article class="metric">
        <span>${label} Accuracy</span>
        <strong>${pct(report.test_accuracy)}</strong>
      </article>
      <article class="metric">
        <span>${label} F1 Score</span>
        <strong>${pct(report.test_f1)}</strong>
      </article>
    `);
  }
  metrics.innerHTML = cards.join("");
}

function renderNews(data) {
  updatedAt.textContent = `뉴스 업데이트: ${data.updated_at}`;
  newsGrid.innerHTML = Object.entries(data.groups).map(([key, group]) => {
    const items = group.items || [];
    const body = group.error
      ? `<div class="error">뉴스 수집 실패: ${group.error}</div>`
      : items.length
        ? `<div class="news-list">${items.map(item => `
            <a class="news-item" href="${item.url}" target="_blank" rel="noreferrer">
              <strong>${item.title}</strong>
              <p>${item.summary || "요약 정보가 없습니다."}</p>
              <span class="meta"><span>${item.source}</span><span>${item.published_at}</span></span>
            </a>
          `).join("")}</div>`
        : `<div class="empty">표시할 뉴스가 없습니다.</div>`;
    return `
      <article class="news-column">
        <div class="column-title">
          <h3>${group.label}</h3>
          <p>${group.query}</p>
        </div>
        ${body}
      </article>
    `;
  }).join("");
}

function renderTomorrow(payload) {
  const predictions = payload.predictions || {};
  tomorrowGrid.innerHTML = Object.entries(predictions).map(([key, item]) => {
    const isUp = Number(item.predicted) === 1;
    const confidence = item.confidence === null || item.confidence === undefined
      ? "확률 정보 없음"
      : `모델 확신도 ${pct(item.confidence)}`;
    if (item.error) {
      return `
        <article class="tomorrow-card">
          <div>
            <h3>${item.display_name}</h3>
            <p class="error">${item.error}</p>
          </div>
          <div class="direction down">오류</div>
        </article>
      `;
    }
    return `
      <article class="tomorrow-card">
        <div>
          <h3>${item.display_name}</h3>
          <p>최신 ${item.input_news_chars}자 뉴스 신호와 최근 종가 ${Number(item.latest_close).toLocaleString()} 기준 · ${confidence}</p>
        </div>
        <div class="direction ${isUp ? "" : "down"}">${item.direction}</div>
      </article>
    `;
  }).join("");
}

function renderPrediction(payload) {
  const rows = payload.recent.map((row, index) => `
    <tr>
      <td>${index + 1}</td>
      <td><span class="badge ${row.actual ? "" : "down"}">${labelDirection(row.actual)}</span></td>
      <td><span class="badge ${row.predicted ? "" : "down"}">${labelDirection(row.predicted)}</span></td>
    </tr>
  `).join("");

  return `
    <article class="prediction-card">
      <h3>${payload.display_name}</h3>
      <p class="meta">총 ${payload.total_rows}개 샘플 · 최근 ${payload.recent.length}개 표시</p>
      <table class="prediction-table">
        <thead>
          <tr><th>#</th><th>실제</th><th>예측</th></tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </article>
  `;
}

async function loadDashboard() {
  try {
    await getJson("/api/health");
    setApiState(true, "FastAPI 연결됨");
  } catch (error) {
    setApiState(false, "FastAPI 연결 실패");
    updatedAt.textContent = "백엔드 서버를 먼저 실행하세요.";
    return;
  }

  const [summary, tomorrow, koreanair, lgchem] = await Promise.all([
    getJson("/api/predictions/summary"),
    getJson("/api/predictions/tomorrow?limit=5"),
    getJson("/api/predictions/koreanair?limit=10"),
    getJson("/api/predictions/lgchem?limit=10"),
  ]);

  renderMetrics(summary);
  renderTomorrow(tomorrow);
  renderNews(tomorrow.news);
  predictionGrid.innerHTML = [renderPrediction(koreanair), renderPrediction(lgchem)].join("");
}

refreshNews.addEventListener("click", async () => {
  refreshNews.disabled = true;
  refreshNews.textContent = "수집 중";
  try {
    const tomorrow = await getJson("/api/predictions/tomorrow?limit=5");
    renderTomorrow(tomorrow);
    renderNews(tomorrow.news);
  } finally {
    refreshNews.disabled = false;
    refreshNews.textContent = "뉴스 업데이트";
  }
});

loadDashboard().catch(error => {
  setApiState(false, "데이터 로드 실패");
  updatedAt.textContent = error.message;
});
