/* Churn Copilot SPA: чаты + разборы клиентов с нижней панелью. Без сборки и зависимостей. */
"use strict";

/* ---------- утилиты ---------- */

const $ = (sel, root = document) => root.querySelector(sel);

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
));

const ICONS = {
  plus: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" aria-hidden="true"><path d="M8 3v10M3 8h10"/></svg>',
  up: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M8 13V3M4 7l4-4 4 4"/></svg>',
  db: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" aria-hidden="true"><ellipse cx="8" cy="4.2" rx="5" ry="2.2"/><path d="M3 4.2v7.6c0 1.2 2.2 2.2 5 2.2s5-1 5-2.2V4.2"/><path d="M3 8c0 1.2 2.2 2.2 5 2.2S13 9.2 13 8"/></svg>',
  term: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 5l3.5 3L3 11"/><path d="M9 11h4"/></svg>',
  right: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M6 3l5 5-5 5"/></svg>',
  x: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" aria-hidden="true"><path d="M4 4l8 8M12 4l-8 8"/></svg>',
};

const icon = (name) => ICONS[name] || "";

const fmtRisk = (p) => (typeof p === "number" ? p.toFixed(2) : "—");

const fmtMoney = (v) => (typeof v === "number" ? v.toLocaleString("ru-RU", { maximumFractionDigits: 0 }) : "—");

function dateGroup(iso) {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "Ранее";
  const now = new Date();
  const day = (x) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const diff = (day(now) - day(d)) / 86400000;
  if (diff === 0) return "Сегодня";
  if (diff === 1) return "Вчера";
  return d.toLocaleDateString("ru-RU", { day: "2-digit", month: "long" });
}

/* ---------- состояние ---------- */

const state = {
  view: "chats",            // chats | clients
  threads: [],
  threadId: null,           // открытый чат (null = новый)
  messages: [],             // история открытого чата
  sending: false,
  clients: [],
  clientsTotal: 0,
  clientsLoaded: false,
  clientsLoading: false,
  clientsFilter: "all",     // all | high | mid | low
  activeRow: null,
  analyses: new Map(),      // client_id (string) -> analysis
  drawerOpen: false,
  drawerActive: null,
};

function newAnalysis(client) {
  return {
    client,
    clientId: String(client.client_id),
    surname: client.surname,
    data: null,             // {churn_probability, reasons:[...]}
    text: "",
    status: "loading",      // loading | done | error
    controller: new AbortController(),
  };
}

/* ---------- API ---------- */

async function api(path, opts = {}) {
  const resp = await fetch(path, opts);
  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`;
    try {
      const data = await resp.json();
      if (data && data.detail) detail = data.detail;
    } catch { /* не JSON — оставляем статус */ }
    throw new Error(detail);
  }
  return resp;
}

/** POST c SSE-ответом: разбирает "data: {...}\n\n" и вызывает onEvent(event). */
async function sse(path, body, onEvent, signal) {
  const resp = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!resp.ok || !resp.body) {
    let detail = `HTTP ${resp.status}`;
    try {
      const data = await resp.json();
      if (data && data.detail) detail = data.detail;
    } catch { /* не JSON */ }
    throw new Error(detail);
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let sep;
    while ((sep = buffer.indexOf("\n\n")) >= 0) {
      const raw = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      for (const line of raw.split("\n")) {
        if (!line.startsWith("data: ")) continue;
        try { onEvent(JSON.parse(line.slice(6))); } catch { /* битый кусок — пропускаем */ }
      }
    }
  }
}

/* ---------- боковая панель ---------- */

function renderSeg() {
  document.body.dataset.view = state.view;
  $("#seg").querySelectorAll("button").forEach((b) => {
    b.classList.toggle("active", b.dataset.view === state.view);
  });
}

function renderSidebar() {
  const list = $("#side-list");
  if (state.view === "chats") {
    list.innerHTML = state.threads.length
      ? groupThreads(state.threads)
      : '<div class="group">Пока нет диалогов</div>';
  } else {
    list.innerHTML = clientFiltersHtml();
  }
}

function groupThreads(threads) {
  let html = "";
  let lastGroup = null;
  for (const t of threads) {
    const group = dateGroup(t.updated_at);
    if (group !== lastGroup) {
      html += `<div class="group">${esc(group)}</div>`;
      lastGroup = group;
    }
    const time = new Date(t.updated_at);
    const timeText = Number.isNaN(time.getTime()) ? "" : time.toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit" });
    html += `
      <button type="button" class="item ${t.thread_id === state.threadId ? "active" : ""}" data-thread="${esc(t.thread_id)}">
        <span class="item-title">${esc(t.title)}</span>
        <span class="item-time">${esc(timeText)}</span>
      </button>`;
  }
  return html;
}

const FILTERS = [
  { key: "all", label: "Все клиенты" },
  { key: "high", label: "Высокий риск", match: (p) => p >= 0.6 },
  { key: "mid", label: "Средний риск", match: (p) => p >= 0.4 && p < 0.6 },
  { key: "low", label: "Низкий риск", match: (p) => p < 0.4 },
];

function clientFiltersHtml() {
  const counts = { all: state.clients.length };
  for (const f of FILTERS.slice(1)) {
    counts[f.key] = state.clients.filter((c) => f.match(c.churn_probability)).length;
  }
  return `
    <div class="group">Списки</div>
    ${FILTERS.map((f) => `
      <button type="button" class="item ${state.clientsFilter === f.key ? "active" : ""}" data-filter="${f.key}">
        <span class="item-title">${f.label}</span>
        <span class="item-time">${counts[f.key] ?? 0}</span>
      </button>`).join("")}`;
}

/* ---------- маршрутизация ---------- */

function route() {
  const hash = location.hash || "#/chats";
  // Во время стрима не перерисовываем: ссылки на DOM-узлы ответа потеряются
  if (state.sending) return;
  if (hash.startsWith("#/c/")) {
    openThread(decodeURIComponent(hash.slice(4)));
  } else if (hash === "#/clients") {
    showView("clients");
  } else {
    showView("chats");
  }
}

function showView(view) {
  state.view = view;
  renderSeg();
  if (view === "clients") {
    renderClientsView();
    renderSidebar();
  } else {
    $("#card").hidden = true;
    state.drawerOpen = false;
    renderChatView();
    renderSidebar();
  }
}

function openThread(threadId) {
  if (state.view !== "chats" || state.threadId !== threadId) {
    state.view = "chats";
    state.drawerOpen = false;
    $("#card").hidden = true;
  }
  state.threadId = threadId;
  renderSeg();
  renderChatView(true);
  renderSidebar();
}

/* ---------- чат ---------- */

function threadTitle() {
  const t = state.threads.find((x) => x.thread_id === state.threadId);
  return t ? t.title : "Новый чат";
}

async function loadThreads() {
  try {
    const resp = await api("/api/threads");
    const data = await resp.json();
    state.threads = data.threads || [];
    if (state.view === "chats") renderSidebar();
  } catch (e) {
    console.warn("Не удалось загрузить диалоги:", e.message);
  }
}

function renderChatView(withHistory = false) {
  const main = $("#main");
  main.innerHTML = `
    <header class="main-head">
      <div class="crumb">${esc(threadTitle())}</div>
      <div class="head-meta">локальная модель · vLLM</div>
    </header>
    <div class="stream" id="stream"></div>
    <div class="composer">
      <div class="input-pill">
        <textarea id="msg-input" rows="1" placeholder="Спросите про отток клиентов…"></textarea>
        <button class="send" id="send-btn" type="button" aria-label="Отправить">${icon("up")}</button>
      </div>
      <div class="composer-note">Enter — отправить · Shift+Enter — перенос строки</div>
    </div>`;

  const stream = $("#stream");
  if (withHistory) {
    loadHistory().then(renderMessages);
  } else {
    renderMessages();
  }

  const input = $("#msg-input");
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 120)}px`;
  });
  $("#send-btn").addEventListener("click", sendMessage);
  input.focus();
}

async function loadHistory() {
  if (!state.threadId) { state.messages = []; return; }
  try {
    const resp = await api(`/api/threads/${encodeURIComponent(state.threadId)}`);
    const data = await resp.json();
    state.messages = data.messages || [];
  } catch (e) {
    state.messages = [];
    console.warn("Не удалось загрузить историю:", e.message);
  }
}

function renderMessages() {
  const stream = $("#stream");
  if (!stream) return;
  stream.innerHTML = "";
  if (!state.messages.length && !state.sending) {
    stream.innerHTML = `
      <div class="welcome">
        <h2>Чем помочь?</h2>
        <p>Спросите про риск оттока конкретного клиента или откройте раздел «Клиенты» — там можно разобрать клиента одной кнопкой.</p>
        <div class="hint">
          <button type="button" data-goto="#/clients">Открыть список клиентов</button>
        </div>
      </div>`;
    stream.querySelector("[data-goto]").addEventListener("click", () => { location.hash = "#/clients"; });
    return;
  }
  for (const m of state.messages) {
    addMessageNode(m.role, m.content);
  }
  scrollStream(true);
}

function addMessageNode(role, text) {
  const stream = $("#stream");
  const welcome = stream.querySelector(".welcome");
  if (welcome) welcome.remove();
  const node = document.createElement("div");
  node.className = `msg ${role === "user" ? "user" : "ai"}`;
  if (role === "user") {
    node.innerHTML = `<div class="bubble"></div>`;
    node.firstElementChild.textContent = text;
  } else {
    node.innerHTML = `<div class="answer"></div>`;
    node.firstElementChild.textContent = text;
  }
  stream.appendChild(node);
  scrollStream();
  return node;
}

function scrollStream(force = false) {
  const stream = $("#stream");
  if (!stream) return;
  stream.scrollTop = stream.scrollHeight;
  void force;
}

async function sendMessage() {
  if (state.sending) return;
  const input = $("#msg-input");
  const text = (input.value || "").trim();
  if (!text) return;

  if (!state.threadId) state.threadId = crypto.randomUUID();
  state.sending = true;

  addMessageNode("user", text);
  input.value = "";
  input.style.height = "auto";

  const stream = $("#stream");
  const typing = document.createElement("div");
  typing.className = "typing";
  typing.innerHTML = '<span class="dot"></span>Агент думает…';
  stream.appendChild(typing);

  const aiNode = document.createElement("div");
  aiNode.className = "msg ai";
  stream.appendChild(aiNode);
  let currentAnswer = null;

  const ensureAnswer = () => {
    if (!currentAnswer) {
      currentAnswer = document.createElement("div");
      currentAnswer.className = "answer";
      aiNode.appendChild(currentAnswer);
    }
    return currentAnswer;
  };

  try {
    await sse("/api/chat", { thread_id: state.threadId, message: text }, (ev) => {
      if (ev.type === "tool" && ev.phase === "start") {
        currentAnswer = null;
        typing.remove();
        const chip = document.createElement("div");
        chip.className = "toolchip";
        chip.innerHTML = `${icon("db")}<code>${esc(ev.name)}(…)</code><span class="ms">…</span>`;
        aiNode.appendChild(chip);
        scrollStream();
      } else if (ev.type === "tool" && ev.phase === "end") {
        const chip = aiNode.querySelector(".toolchip:last-of-type .ms");
        if (chip) {
          chip.textContent = ev.ok ? "ok" : "ошибка";
          chip.className = `ms ${ev.ok ? "ok" : "fail"}`;
        }
      } else if (ev.type === "token") {
        typing.remove();
        const answer = ensureAnswer();
        answer.textContent += ev.text;
        scrollStream();
      } else if (ev.type === "error") {
        typing.remove();
        const answer = ensureAnswer();
        answer.classList.add("error");
        answer.textContent += (answer.textContent ? "\n\n" : "") + `Ошибка: ${ev.message}`;
      }
    });
    await loadThreads();
    renderSidebar();
  } catch (e) {
    const answer = ensureAnswer();
    answer.classList.add("error");
    answer.textContent += (answer.textContent ? "\n\n" : "") + `Ошибка: ${e.message}`;
  } finally {
    typing.remove();
    state.sending = false;
    scrollStream();
  }
}

/* ---------- клиенты ---------- */

async function loadClients() {
  if (state.clientsLoading) return;
  state.clientsLoading = true;
  renderClientsMeta();
  try {
    const resp = await api("/api/clients?limit=50");
    const data = await resp.json();
    state.clients = data.clients || [];
    state.clientsTotal = data.total || 0;
    state.clientsLoaded = true;
  } catch (e) {
    console.warn("Не удалось загрузить клиентов:", e.message);
    state.clients = [];
    state.clientsLoaded = true;
  } finally {
    state.clientsLoading = false;
    renderClientsMeta();
    renderClientsTable();
    renderSidebar();
  }
}

function renderClientsMeta() {
  const meta = $("#cl-meta");
  if (meta) {
    meta.textContent = state.clientsLoading
      ? "загрузка…"
      : state.clientsLoaded
        ? `${state.clients.length} из ${state.clientsTotal} · риски из churn_pred_service`
        : "";
  }
}

function renderClientsView() {
  $("#main").innerHTML = `
    <header class="main-head">
      <div class="crumb">Клиенты · выгрузка</div>
      <div class="head-meta" id="cl-meta"></div>
    </header>
    <div class="table-area">
      <table class="cc-table">
        <thead>
          <tr><th>Клиент</th><th>Страна</th><th>Возраст</th><th>Баланс</th><th>Риск</th><th></th></tr>
        </thead>
        <tbody id="cl-body"></tbody>
      </table>
    </div>
    <div class="drawer hidden" id="drawer">
      <div class="drawer-handle"></div>
      <div class="drawer-tabs" id="drawer-tabs"></div>
      <div class="drawer-body" id="drawer-body"></div>
      <div class="drawer-status" id="drawer-status">copilot ›</div>
    </div>`;
  renderClientsMeta();
  renderClientsTable();
  if (!state.clientsLoaded && !state.clientsLoading) loadClients();
  renderDrawer();
}

function riskClass(p) {
  if (typeof p !== "number") return "";
  if (p >= 0.6) return "high";
  if (p >= 0.4) return "mid";
  return "low";
}

function visibleClients() {
  const f = FILTERS.find((x) => x.key === state.clientsFilter);
  if (!f || !f.match) return state.clients;
  return state.clients.filter((c) => f.match(c.churn_probability));
}

function renderClientsTable() {
  const body = $("#cl-body");
  if (!body) return;
  const rows = visibleClients();
  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="6" style="color:var(--ink-3)">${state.clientsLoading ? "Загрузка…" : "Нет клиентов в этом списке"}</td></tr>`;
    return;
  }
  body.innerHTML = rows.map((c) => `
    <tr class="${state.activeRow === c.client_id ? "active" : ""}" data-row="${c.client_id}">
      <td><span class="cid">#${c.client_id}</span>${esc(c.surname)}</td>
      <td>${esc(c.geography)}</td>
      <td class="tabnum">${c.age ?? "—"}</td>
      <td class="tabnum">${fmtMoney(c.balance)}</td>
      <td><span class="risk-badge ${riskClass(c.churn_probability)} tabnum">${fmtRisk(c.churn_probability)}</span></td>
      <td class="td-act"><button class="rowbtn" type="button" data-analyze="${c.client_id}">Разобрать</button></td>
    </tr>`).join("");
}

/* ---------- разбор клиента (нижняя панель) ---------- */

function openAnalysis(client, clientRow) {
  const key = String(client);
  let analysis = state.analyses.get(key);
  if (!analysis) {
    analysis = newAnalysis(clientRow || { client_id: client, surname: `№${client}` });
    state.analyses.set(key, analysis);
    streamAnalysis(analysis);
  }
  state.drawerOpen = true;
  state.drawerActive = key;
  state.activeRow = Number(client);
  $("#drawer").classList.remove("hidden");
  renderClientsTable();
  renderDrawer();
}

function streamAnalysis(analysis) {
  sse("/api/analyze", { client_id: Number(analysis.clientId) }, (ev) => {
    if (ev.type === "data") {
      analysis.data = ev;
    } else if (ev.type === "token") {
      analysis.text += ev.text;
    } else if (ev.type === "done") {
      analysis.status = "done";
    } else if (ev.type === "error") {
      analysis.status = "error";
      analysis.text += (analysis.text ? "\n\n" : "") + `Ошибка: ${ev.message}`;
    }
    if (state.drawerActive === analysis.clientId) renderDrawer();
  }, analysis.controller.signal).catch((e) => {
    if (e.name === "AbortError") return;
    analysis.status = "error";
    analysis.text += (analysis.text ? "\n\n" : "") + `Ошибка: ${e.message}`;
    if (state.drawerActive === analysis.clientId) renderDrawer();
  });
}

function drawerStatusText(a) {
  if (!a) return "copilot ›";
  if (a.status === "done") return `copilot › №${a.clientId} · готово`;
  if (a.status === "error") return `copilot › №${a.clientId} · ошибка`;
  if (a.data && a.data.reasons && a.data.reasons.length) return `copilot › №${a.clientId} · риски получены, генерирую разбор…`;
  return `copilot › №${a.clientId} · вызываю инструменты…`;
}

function miniFactorsHtml(a) {
  const reasons = (a.data && a.data.reasons) || [];
  return reasons.slice(0, 3).map((r) => `
    <div class="mini-f">
      <div class="mf-top"><span>${esc(r.label)}${r.value ? ` · ${esc(r.value)}` : ""}</span><b class="tabnum">${r.pct}%</b></div>
      <div class="mini-bar"><div class="mini-fill" style="width:${Math.min(100, r.pct)}%"></div></div>
    </div>`).join("");
}

function renderDrawer() {
  const drawer = $("#drawer");
  if (!drawer) return;
  drawer.classList.toggle("hidden", !state.drawerOpen);
  const tabs = $("#drawer-tabs");
  const body = $("#drawer-body");
  const status = $("#drawer-status");
  if (!state.drawerOpen) return;

  tabs.innerHTML = `${icon("term")}${[...state.analyses.values()].map((a) => `
    <div class="dtab ${state.drawerActive === a.clientId ? "active" : ""}" data-tab="${a.clientId}" role="tab" aria-selected="${state.drawerActive === a.clientId}">
      №${a.clientId} · ${a.data && typeof a.data.churn_probability === "number" ? a.data.churn_probability.toFixed(2) : "…"}
      <button type="button" class="closer" data-close="${a.clientId}" aria-label="Закрыть разбор">${icon("x")}</button>
    </div>`).join("")}
    <span class="dtab-note">результат одного клика — чат не обязателен</span>`;

  const a = state.analyses.get(state.drawerActive);
  if (!a) {
    body.innerHTML = '<div class="placeholder" style="color:var(--ink-3);font-size:12.5px">Выберите клиента и нажмите «Разобрать»</div>';
    status.textContent = "copilot ›";
    return;
  }

  const riskLabel = a.data && typeof a.data.churn_probability === "number"
    ? (a.data.churn_probability >= 0.6 ? "высокий риск" : a.data.churn_probability >= 0.4 ? "средний риск" : "низкий риск")
    : "…";
  body.innerHTML = `
    <div class="drawer-pane">
      <div class="dp-risk">
        <div class="dp-big tabnum">${fmtRisk(a.data && a.data.churn_probability)}</div>
        <div class="dp-lab">${riskLabel}</div>
        ${miniFactorsHtml(a)}
      </div>
      <div class="dp-text">
        <div class="dp-lab">Разбор агента</div>
        <div class="text">${a.text ? esc(a.text) : '<span style="color:var(--ink-3)">Запускаю агента…</span>'}</div>
      </div>
      <div class="dp-actions">
        <button class="abtn primary" type="button" data-act="continue" ${a.status === "done" && a.text ? "" : "disabled"}>Продолжить в чате${icon("right")}</button>
        <button class="abtn" type="button" data-act="card" ${a.data ? "" : "disabled"}>Открыть карточку</button>
        <button class="abtn" type="button" data-act="close">Закрыть</button>
      </div>
    </div>`;
  status.textContent = drawerStatusText(a);
  const continueBtn = body.querySelector('[data-act="continue"]');
  if (continueBtn) continueBtn.addEventListener("click", () => continueInChat(a));
}

function closeAnalysis(key) {
  const a = state.analyses.get(key);
  if (a) a.controller.abort();
  state.analyses.delete(key);
  if (state.drawerActive === key) {
    state.drawerActive = state.analyses.keys().next().value || null;
  }
  if (!state.analyses.size) state.drawerOpen = false;
  renderClientsTable();
  renderDrawer();
}

async function continueInChat(a) {
  try {
    const resp = await api("/api/threads/seed", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title: `Разбор №${a.clientId}`,
        analysis_text: a.text,
        client_id: a.clientId,
      }),
    });
    const data = await resp.json();
    await loadThreads();
    $("#card").hidden = true;
    location.hash = `#/c/${encodeURIComponent(data.thread_id)}`;
  } catch (e) {
    console.warn("Не удалось создать чат:", e.message);
    const status = $("#drawer-status");
    if (status) status.textContent = `copilot › ошибка: ${e.message}`;
  }
}

/* ---------- карточка клиента ---------- */

function renderCard(a) {
  const card = $("#card");
  const c = a.client || {};
  const prob = a.data ? a.data.churn_probability : null;
  card.innerHTML = `
    <div class="detail-head">
      <span class="cid-head">Карточка клиента <code>#${a.clientId}</code></span>
      <button class="detail-close" type="button" aria-label="Закрыть карточку">${icon("x")}</button>
    </div>
    <div class="risk-block">
      <div class="risk-top"><span class="risk-label">Вероятность оттока</span><span class="risk-value tabnum">${fmtRisk(prob)}</span></div>
      <div class="risk-track"><div class="risk-fill" style="width:${typeof prob === "number" ? Math.round(prob * 100) : 0}%"></div></div>
      <div class="risk-note">${prob == null ? "прогноз недоступен" : `${(prob * 100).toFixed(1)}% · порог модели 0.50`}</div>
    </div>
    <dl class="facts">
      <div><dt>Фамилия</dt><dd>${esc(c.surname ?? "—")}</dd></div>
      <div><dt>Страна</dt><dd>${esc(c.geography ?? "—")}</dd></div>
      <div><dt>Возраст</dt><dd class="tabnum">${c.age ?? "—"}</dd></div>
      <div><dt>Стаж, лет</dt><dd class="tabnum">${c.tenure ?? "—"}</dd></div>
      <div><dt>Баланс</dt><dd class="tabnum">${fmtMoney(c.balance)}</dd></div>
      <div><dt>Зарплата (оценка)</dt><dd class="tabnum">${fmtMoney(c.estimated_salary)}</dd></div>
      <div><dt>Продукты</dt><dd class="tabnum">${c.num_of_products ?? "—"}</dd></div>
      <div><dt>Активный участник</dt><dd>${c.is_active_member ? "Да" : "Нет"}</dd></div>
    </dl>
    <div class="detail-sub">Факторы риска (SHAP)</div>
    ${(a.data && a.data.reasons || []).map((r) => `
      <div class="factor">
        <div class="f-top"><span>${esc(r.label)}${r.value ? ` · ${esc(r.value)}` : ""}</span><span class="f-value tabnum">${r.pct}%</span></div>
        <div class="f-track"><div class="f-fill" style="width:${Math.min(100, r.pct)}%"></div></div>
      </div>`).join("") || '<div style="font-size:12px;color:var(--ink-3)">Нет данных от prediction-сервиса</div>'}
    <div class="dc-actions">
      <button class="abtn primary" type="button" data-act="continue" ${a.status === "done" && a.text ? "" : "disabled"}>Продолжить в чате${icon("right")}</button>
    </div>`;
  card.hidden = false;
  card.querySelector(".detail-close").addEventListener("click", () => { card.hidden = true; });
  card.querySelector('[data-act="continue"]').addEventListener("click", () => continueInChat(a));
}

/* ---------- обработчики ---------- */

document.addEventListener("click", (e) => {
  // Кнопка-крестик на вкладке панели — раньше переключения вкладок
  const closeBtn = e.target.closest("[data-close]");
  if (closeBtn) {
    closeAnalysis(closeBtn.dataset.close);
    return;
  }

  const target = e.target.closest("button");
  if (!target) return;

  if (target.closest("#new-chat")) {
    if (state.sending) return;
    state.threadId = null;
    state.messages = [];
    if (location.hash === "#/chats" || location.hash === "") {
      showView("chats"); // hash не меняется — перерисовываем вручную
    } else {
      location.hash = "#/chats";
    }
    return;
  }
  if (target.closest("#seg")) {
    location.hash = target.dataset.view === "clients" ? "#/clients" : "#/chats";
    return;
  }
  if (target.dataset.thread) {
    if (state.sending) return;
    location.hash = `#/c/${encodeURIComponent(target.dataset.thread)}`;
    return;
  }
  if (target.dataset.filter) {
    state.clientsFilter = target.dataset.filter;
    renderSidebar();
    renderClientsTable();
    return;
  }
  if (target.dataset.analyze !== undefined) {
    const clientId = Number(target.dataset.analyze);
    const row = state.clients.find((c) => c.client_id === clientId);
    openAnalysis(clientId, row);
    return;
  }
  const tab = target.closest("[data-tab]");
  if (tab) {
    state.drawerActive = tab.dataset.tab;
    renderDrawer();
    return;
  }
  if (target.dataset.act === "close") {
    state.drawerOpen = false;
    renderDrawer();
    return;
  }
  if (target.dataset.act === "card") {
    const a = state.analyses.get(state.drawerActive);
    if (a) renderCard(a);
  }
  /* «Продолжить в чате» слушается точечно в renderDrawer/renderCard */
});

window.addEventListener("hashchange", route);

/* ---------- запуск ---------- */

async function boot() {
  renderSeg();
  route();
  try {
    const resp = await api("/api/health");
    const data = await resp.json();
    const el = $("#conn-status");
    el.textContent = data.agent_ready ? "агент готов" : "агент инициализируется…";
    el.classList.toggle("offline", !data.agent_ready);
  } catch {
    const el = $("#conn-status");
    el.textContent = "агент недоступен";
    el.classList.add("offline");
  }
  await loadThreads();
  if (state.view === "chats") renderSidebar();
}

boot();
