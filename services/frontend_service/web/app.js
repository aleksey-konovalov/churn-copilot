/* Churn Copilot SPA: чаты + клиенты с правой панелью (анализ оттока / карточка). Без сборки и зависимостей. */
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

/* ---------- мини-markdown (без зависимостей) ----------
   Источник — ответы LLM: заголовки, **жирный**, *курсив*, `код`,
   таблицы, списки, цитаты, fenced-код, ---. Весь текст экранируется
   до разметки, теги добавляет только рендерер — XSS исключён. */

function mdInline(s) {
  return s
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|\s)\*([^*\n]+)\*(?=\s|$|[.,!?:;)»])/g, "$1<em>$2</em>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
}

function mdToHtml(src) {
  // LLM иногда клеит заголовок к предыдущему предложению («…3 месяцев.## Разбор»):
  // переносим такую разметку на новую строку после знака препинания (компромисс
  // в пользу снисходительности; в коде-блоках такой паттерн практически не встречается)
  const lines = esc(src)
    .replace(/\r\n?/g, "\n")
    .replace(/([.!?:;)»%])(#{1,6}\s)/g, "$1\n$2")
    .split("\n");
  const out = [];
  let para = [];
  const flushPara = () => {
    if (para.length) {
      out.push(`<p>${para.map(mdInline).join("<br>")}</p>`);
      para = [];
    }
  };
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];

    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      flushPara();
      const lvl = Math.min(heading[1].length, 4);
      out.push(`<h${lvl}>${mdInline(heading[2])}</h${lvl}>`);
      i++;
      continue;
    }
    if (/^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(line)) {
      flushPara();
      out.push("<hr>");
      i++;
      continue;
    }
    // Таблица: строка с |, за которой следует разделитель |---|
    if (/^\s*\|/.test(line) && i + 1 < lines.length && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1])) {
      flushPara();
      const rows = [];
      while (i < lines.length && /^\s*\|/.test(lines[i])) {
        rows.push(lines[i]);
        i++;
      }
      const cells = (row) => row.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => mdInline(c.trim()));
      let table = `<table><thead><tr>${cells(rows[0]).map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>`;
      for (const row of rows.slice(2)) {
        table += `<tr>${cells(row).map((c) => `<td>${c}</td>`).join("")}</tr>`;
      }
      out.push(`${table}</tbody></table>`);
      continue;
    }
    const ordered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    const bulleted = line.match(/^\s*[-*+]\s+(.*)$/);
    if (ordered || bulleted) {
      flushPara();
      const re = ordered ? /^\s*\d+[.)]\s+(.*)$/ : /^\s*[-*+]\s+(.*)$/;
      const items = [];
      while (i < lines.length) {
        const m = lines[i].match(re);
        if (!m) break;
        items.push(mdInline(m[1].trim()));
        i++;
      }
      const tag = ordered ? "ol" : "ul";
      out.push(`<${tag}>${items.map((t) => `<li>${t}</li>`).join("")}</${tag}>`);
      continue;
    }
    if (/^\s*>\s?/.test(line)) {
      flushPara();
      const quote = [];
      while (i < lines.length && /^\s*>\s?/.test(lines[i])) {
        quote.push(mdInline(lines[i].replace(/^\s*>\s?/, "")));
        i++;
      }
      out.push(`<blockquote>${quote.join("<br>")}</blockquote>`);
      continue;
    }
    if (/^\s*```/.test(line)) {
      flushPara();
      i++;
      const code = [];
      while (i < lines.length && !/^\s*```/.test(lines[i])) {
        code.push(lines[i]);
        i++;
      }
      i++; // закрывающая ```
      out.push(`<pre><code>${code.join("\n")}</code></pre>`);
      continue;
    }
    if (!line.trim()) {
      flushPara();
      i++;
      continue;
    }
    para.push(line);
    i++;
  }
  flushPara();
  return out.join("");
}

/** Рендер markdown в узел с троттлингом через rAF: стрим токенов не пересобирает DOM на каждый чанк. */
function mdInto(node, raw) {
  node._mdRaw = raw;
  if (node._mdRaf) return;
  node._mdRaf = requestAnimationFrame(() => {
    node._mdRaf = 0;
    node.innerHTML = mdToHtml(node._mdRaw);
  });
}

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
  clientsError: null,       // сообщение последней ошибки /api/clients
  riskCounts: null,         // {all, high, mid, low, unknown} — глобальные счётчики с сервера
  clientsPage: 0,           // текущая страница пагинатора (смещение = page * PAGE_SIZE)
  clientsQuery: "",         // поиск по фамилии/ID (серверный, q в /api/clients)
  clientsFilter: "all",     // all | high | mid | low (серверный фильтр, risk в /api/clients)
  activeRow: null,
  analyses: new Map(),      // client_id (string) -> analysis
  panelOpen: false,         // правая панель: анализ оттока / карточка клиента
  panelClientId: null,      // какой клиент открыт в панели
  panelTab: "razbor",       // razbor | card
  confirmDelete: null,      // thread_id с нажатой «×» (второй клик удаляет)
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
    const confirming = state.confirmDelete === t.thread_id;
    html += `
      <div class="item ${t.thread_id === state.threadId ? "active" : ""} ${confirming ? "confirm" : ""}">
        <button type="button" class="item-main" data-thread="${esc(t.thread_id)}">
          <span class="item-title">${esc(t.title)}</span>
          ${confirming ? '<span class="item-time">удалить?</span>' : `<span class="item-time">${esc(timeText)}</span>`}
        </button>
        <button type="button" class="item-del" data-del="${esc(t.thread_id)}"
                aria-label="Удалить чат">${confirming ? "Да" : icon("x")}</button>
      </div>`;
  }
  return html;
}

// Пороги бакетов живут на сервере (RISK_BUCKETS в agent_service/src/api.py):
// фильтр и счётчики считает /api/clients глобально по всей выгрузке
const FILTERS = [
  { key: "all", label: "Все клиенты" },
  { key: "high", label: "Высокий риск" },
  { key: "mid", label: "Средний риск" },
  { key: "low", label: "Низкий риск" },
];

function clientFiltersHtml() {
  // Счётчики от сервера — по всем клиентам выборки; пока риски не посчитаны
  // (predict-сервис недоступен) — показываем без цифр вместо страничных
  const c = state.riskCounts;
  const counts = {
    all: c ? c.all : (state.clientsTotal || ""),
    high: c ? c.high : "",
    mid: c ? c.mid : "",
    low: c ? c.low : "",
  };
  // Структура как у чатов: .item-обёртка + .item-main (после перестройки списка чатов стили живут на item-main)
  return `
    <div class="group">Фильтры по рискам</div>
    ${FILTERS.map((f) => `
      <div class="item ${state.clientsFilter === f.key ? "active" : ""}">
        <button type="button" class="item-main" data-filter="${f.key}">
          <span class="item-title">${f.label}</span>
          <span class="item-time">${counts[f.key]}</span>
        </button>
      </div>`).join("")}`;
}

/* ---------- маршрутизация ---------- */

function route() {
  const hash = location.hash || "#/clients";
  // Во время стрима не перерисовываем: ссылки на DOM-узлы ответа потеряются
  if (state.sending) return;
  if (hash.startsWith("#/c/")) {
    openThread(decodeURIComponent(hash.slice(4)));
  } else if (hash === "#/chats") {
    showView("chats");
  } else {
    showView("clients");
  }
}

function showView(view) {
  state.view = view;
  renderSeg();
  if (view === "clients") {
    renderClientsView();
    renderSidebar();
  } else {
    closePanel();
    renderChatView();
    renderSidebar();
  }
}

async function openThread(threadId) {
  if (state.view !== "chats" || state.threadId !== threadId) {
    state.view = "chats";
    closePanel();
  }
  state.threadId = threadId;
  renderSeg();
  renderChatView(true);
  renderSidebar();
  // Тред может отсутствовать в загруженном списке (переход по ссылке на свежий seed) — подтягиваем
  if (!state.threads.some((t) => t.thread_id === threadId)) {
    await loadThreads();
  }
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
    if (state.view === "chats") {
      renderSidebar();
      updateCrumb();
    }
  } catch (e) {
    console.warn("Не удалось загрузить диалоги:", e.message);
  }
}

function updateCrumb() {
  const crumb = document.querySelector("#main .crumb");
  if (!crumb || !state.threadId) return;
  const thread = state.threads.find((x) => x.thread_id === state.threadId);
  if (thread) crumb.textContent = thread.title;
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
        <p>Спросите про риск оттока конкретного клиента или откройте раздел «Клиенты» — там можно запустить анализ оттока одной кнопкой.</p>
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
    node.innerHTML = `<div class="answer md"></div>`;
    node.firstElementChild.innerHTML = mdToHtml(text);
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
  let rawAnswer = "";

  const ensureAnswer = () => {
    if (!currentAnswer) {
      currentAnswer = document.createElement("div");
      currentAnswer.className = "answer md";
      aiNode.appendChild(currentAnswer);
    }
    return currentAnswer;
  };

  const showPlain = (message) => {
    // Ошибки — без markdown: сырой текст исключения не должен размечаться
    const answer = ensureAnswer();
    answer.classList.add("error");
    answer.textContent = (rawAnswer ? `${rawAnswer}\n\n` : "") + message;
  };

  try {
    await sse("/api/chat", { thread_id: state.threadId, message: text }, (ev) => {
      if (ev.type === "tool" && ev.phase === "start") {
        currentAnswer = null;
        rawAnswer = "";
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
        rawAnswer += ev.text;
        mdInto(ensureAnswer(), rawAnswer);
        scrollStream();
      } else if (ev.type === "error") {
        typing.remove();
        showPlain(`Ошибка: ${ev.message}`);
      }
    });
    await loadThreads();
    renderSidebar();
  } catch (e) {
    showPlain(`Ошибка: ${e.message}`);
  } finally {
    typing.remove();
    state.sending = false;
    scrollStream();
  }
}

/* ---------- клиенты ---------- */

const PAGE_SIZE = 50;

async function loadClients() {
  if (state.clientsLoading) return;
  state.clientsLoading = true;
  renderClientsMeta();
  try {
    const q = state.clientsQuery ? `&q=${encodeURIComponent(state.clientsQuery)}` : "";
    const risk = state.clientsFilter !== "all" ? `&risk=${state.clientsFilter}` : "";
    const resp = await api(`/api/clients?limit=${PAGE_SIZE}&offset=${state.clientsPage * PAGE_SIZE}${q}${risk}`);
    const data = await resp.json();
    state.clients = data.clients || [];
    state.clientsTotal = data.total || 0;
    state.riskCounts = data.risk_counts || null;
    state.clientsLoaded = true;
    state.clientsError = null;
  } catch (e) {
    console.warn("Не удалось загрузить клиентов:", e.message);
    state.clients = [];
    state.clientsTotal = 0;
    state.riskCounts = null;
    state.clientsLoaded = true;
    state.clientsError = e.message;
  } finally {
    state.clientsLoading = false;
    renderClientsMeta();
    renderClientsTable();
    renderClientsPager();
    renderSidebar();
  }
}

function renderClientsPager() {
  const box = $("#cl-pager");
  if (!box) return;
  const pages = Math.ceil(state.clientsTotal / PAGE_SIZE);
  if (!state.clientsLoaded || pages <= 1) {
    box.hidden = true;
    box.innerHTML = "";
    return;
  }
  const cur = state.clientsPage;
  // Окно номеров: первая, последняя и соседние с текущей; между разрывами — многоточие
  const nums = [...new Set([0, pages - 1, cur - 1, cur, cur + 1].filter((p) => p >= 0 && p < pages))]
    .sort((a, b) => a - b);
  let btns = "";
  nums.forEach((p, i) => {
    if (i && p - nums[i - 1] > 1) btns += '<span class="pg-dots">…</span>';
    btns += `<button type="button" class="${p === cur ? "active" : ""}" data-page="${p}" aria-label="Страница ${p + 1}">${p + 1}</button>`;
  });
  box.innerHTML = `
    <span class="pg-info">стр. ${cur + 1} из ${pages}</span>
    <div class="pg-btns">
      <button type="button" data-page="${cur - 1}" ${cur === 0 ? "disabled" : ""} aria-label="Назад">←</button>
      ${btns}
      <button type="button" data-page="${cur + 1}" ${cur >= pages - 1 ? "disabled" : ""} aria-label="Вперёд">→</button>
    </div>`;
}

async function goToClientsPage(page) {
  if (page < 0 || page === state.clientsPage) return;
  state.clientsPage = page;
  await loadClients();
  const area = $(".table-area");
  if (area) area.scrollTop = 0;
}

/** Клиент по id напрямую из API: deep-link вне загруженной страницы, свежий риск для карточки. */
async function fetchClient(clientId) {
  try {
    const resp = await api(`/api/clients/${clientId}`);
    return await resp.json();
  } catch (e) {
    console.warn(`Клиент ${clientId} недоступен:`, e.message);
    return null;
  }
}

function renderClientsMeta() {
  const meta = $("#cl-meta");
  if (meta) {
    meta.textContent = state.clientsLoading
      ? "загрузка…"
      : state.clientsError
        ? "ошибка загрузки"
        : state.clientsLoaded
          ? state.clientsQuery
            ? `найдено: ${state.clients.length} из ${state.clientsTotal}`
            : `${state.clients.length} из ${state.clientsTotal} · риски из churn_pred_service`
          : "";
  }
}

function renderClientsView() {
  $("#main").innerHTML = `
    <header class="main-head">
      <div class="crumb">Клиенты · выгрузка</div>
      <input class="head-search" id="cl-search" type="search" autocomplete="off"
             placeholder="Поиск по фамилии или ID…" value="${esc(state.clientsQuery)}">
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
    <div class="cl-pager" id="cl-pager" hidden></div>`;

  // Серверный поиск по фамилии/ID: дебаунс, чтобы не дёргать API на каждый символ
  const search = $("#cl-search");
  let debounce;
  search.addEventListener("input", () => {
    clearTimeout(debounce);
    debounce = setTimeout(() => {
      state.clientsQuery = search.value.trim();
      state.clientsPage = 0; // новый запрос — с первой страницы
      loadClients();
    }, 300);
  });

  $("#cl-pager").addEventListener("click", (e) => {
    const btn = e.target.closest("[data-page]");
    if (btn) goToClientsPage(Number(btn.dataset.page));
  });

  renderClientsMeta();
  renderClientsTable();
  renderClientsPager();
  if (!state.clientsLoaded && !state.clientsLoading) loadClients();
}

function riskClass(p) {
  // Пороги бакетов для раскраски бейджа — синхронизированы с RISK_BUCKETS в api.py
  if (typeof p !== "number") return "";
  if (p >= 0.6) return "high";
  if (p >= 0.4) return "mid";
  return "low";
}

function renderClientsTable() {
  const body = $("#cl-body");
  if (!body) return;
  // Фильтр по риску применяет сервер (параметр risk в /api/clients)
  const rows = state.clients;
  if (!rows.length) {
    let empty;
    if (state.clientsError) empty = `Не удалось загрузить клиентов: ${state.clientsError}`;
    else if (state.clientsLoading) empty = "Загрузка…";
    else if (state.clientsQuery || state.clientsFilter !== "all") empty = "Ничего не найдено";
    else empty = "Нет клиентов в этом списке";
    body.innerHTML = `<tr><td colspan="6" style="color:var(--ink-3)">${esc(empty)}</td></tr>`;
    return;
  }
  body.innerHTML = rows.map((c) => `
    <tr class="${state.activeRow === c.client_id ? "active" : ""}" data-row="${c.client_id}">
      <td><span class="cid">#${c.client_id}</span>${esc(c.surname)}</td>
      <td>${esc(c.geography)}</td>
      <td class="tabnum">${c.age ?? "—"}</td>
      <td class="tabnum">${fmtMoney(c.balance)}</td>
      <td><span class="risk-badge ${riskClass(c.churn_probability)} tabnum">${fmtRisk(c.churn_probability)}</span></td>
      <td class="td-act"><button class="rowbtn" type="button" data-analyze="${c.client_id}">Анализ оттока</button></td>
    </tr>`).join("");
}

/* ---------- анализ оттока (правая панель) ---------- */

function openAnalysis(client, clientRow) {
  const key = String(client);
  let analysis = state.analyses.get(key);
  if (!analysis) {
    analysis = newAnalysis(clientRow || { client_id: client, surname: `№${client}` });
    state.analyses.set(key, analysis);
    streamAnalysis(analysis);
  }
  state.panelOpen = true;
  state.panelClientId = key;
  state.panelTab = "razbor";
  state.activeRow = Number(client);
  renderClientsTable();
  renderPanel();
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
    if (state.panelClientId === analysis.clientId) renderPanel();
  }, analysis.controller.signal).catch((e) => {
    if (e.name === "AbortError") return;
    analysis.status = "error";
    analysis.text += (analysis.text ? "\n\n" : "") + `Ошибка: ${e.message}`;
    if (state.panelClientId === analysis.clientId) renderPanel();
  });
}

function panelStatusText(a) {
  if (!a) return "copilot ›";
  if (a.status === "done") return "copilot › анализ оттока · готово";
  if (a.status === "error") return "copilot › анализ оттока · ошибка";
  if (a.data && a.data.reasons && a.data.reasons.length) return "copilot › анализ оттока · готовлю рекомендации…";
  return "copilot › анализ оттока · провожу анализ…";
}

/* Каркас панели — статичный в index.html, сюда перерисовываются вкладки. */
function renderPanel() {
  const panel = $("#client-panel");
  if (!panel) return;
  panel.hidden = !state.panelOpen;
  if (!state.panelOpen) return;
  const a = state.analyses.get(state.panelClientId);

  $("#cp-id").textContent = `#${a ? a.clientId : state.panelClientId ?? "—"}`;
  $("#cp-name").textContent = a ? a.surname : "";

  const razborActive = state.panelTab === "razbor";
  $("#cp-tab-razbor").classList.toggle("active", razborActive);
  $("#cp-tab-razbor").setAttribute("aria-selected", String(razborActive));
  $("#cp-tab-card").classList.toggle("active", !razborActive);
  $("#cp-tab-card").setAttribute("aria-selected", String(!razborActive));
  $("#cp-pane-razbor").hidden = !razborActive;
  $("#cp-pane-card").hidden = razborActive;

  const cta = $("#panel-continue");
  cta.hidden = !razborActive;
  cta.disabled = !(a && a.status === "done" && a.text);

  if (!a) return;
  if (razborActive) renderPanelRazbor(a);
  else renderPanelCard(a);
}

function renderPanelRazbor(a) {
  const prob = a.data ? a.data.churn_probability : null;
  const note = prob == null
    ? (a.data ? "прогноз недоступен" : "получаю прогноз…")
    : `${(prob * 100).toFixed(1)}% · порог модели 0.50`;
  $("#cp-pane-razbor").innerHTML = `
    <div class="cp-risk">
      <div class="cp-risk-top"><span>Вероятность оттока</span><b class="tabnum">${fmtRisk(prob)}</b></div>
      <div class="cp-track"><div class="cp-fill" style="width:${typeof prob === "number" ? Math.round(prob * 100) : 0}%"></div></div>
      <small>${esc(note)}</small>
    </div>
    <div class="cp-sub">Факторы риска (SHAP)</div>
    ${(a.data && a.data.reasons || []).map((r) => `
      <div class="cp-factor">
        <div class="cp-f-top"><span>${esc(r.label)}${r.value ? ` · ${esc(r.value)}` : ""}</span><b class="tabnum">${r.pct}%</b></div>
        <div class="cp-f-track"><div class="cp-f-fill" style="width:${Math.min(100, r.pct)}%"></div></div>
      </div>`).join("") || '<div class="cp-empty">Нет данных от prediction-сервиса</div>'}
    <div class="cp-agent">
      <div class="cp-agent-label">${esc(panelStatusText(a))}</div>
      <div class="cp-agent-text md">${a.text ? mdToHtml(a.text) : '<span class="cp-dots"><span></span><span></span><span></span></span>'}</div>
    </div>`;
}

function renderPanelCard(a) {
  const c = a.client || {};
  const prob = a.data ? a.data.churn_probability : null;
  const riskText = typeof prob === "number"
    ? `${prob.toFixed(2)} · ${prob >= 0.6 ? "высокий" : prob >= 0.4 ? "средний" : "низкий"}`
    : "—";
  $("#cp-pane-card").innerHTML = `
    <div class="cp-riskline"><span>Вероятность оттока</span><b class="tabnum">${esc(riskText)}</b></div>
    <div class="cp-sub">Атрибуты</div>
    <dl class="cp-facts">
      <div><dt>Фамилия</dt><dd>${esc(c.surname ?? "—")}</dd></div>
      <div><dt>Страна</dt><dd>${esc(c.geography ?? "—")}</dd></div>
      <div><dt>Возраст</dt><dd class="tabnum">${c.age ?? "—"}</dd></div>
      <div><dt>Стаж, лет</dt><dd class="tabnum">${c.tenure ?? "—"}</dd></div>
      <div><dt>Баланс</dt><dd class="tabnum">${fmtMoney(c.balance)}</dd></div>
      <div><dt>Зарплата (оценка)</dt><dd class="tabnum">${fmtMoney(c.estimated_salary)}</dd></div>
      <div><dt>Продукты</dt><dd class="tabnum">${c.num_of_products ?? "—"}</dd></div>
      <div><dt>Кредитный рейтинг</dt><dd class="tabnum">${c.credit_score ?? "—"}</dd></div>
      <div><dt>Кредитная карта</dt><dd>${c.has_cr_card ? "Да" : "Нет"}</dd></div>
      <div><dt>Актив. участник</dt><dd>${c.is_active_member ? "Да" : "Нет"}</dd></div>
    </dl>`;
}

async function setPanelTab(tab) {
  state.panelTab = tab;
  if (tab === "card") {
    // Карточка показывает свежие данные клиента, а не снапшот на момент анализа
    const a = state.analyses.get(state.panelClientId);
    if (a) {
      const fresh = await fetchClient(Number(a.clientId));
      if (fresh) {
        a.client = fresh;
        a.surname = fresh.surname;
        const index = state.clients.findIndex((c) => c.client_id === fresh.client_id);
        if (index >= 0) state.clients[index] = fresh;
      }
    }
  }
  renderPanel();
}

function closePanel() {
  state.panelOpen = false;
  renderPanel();
}

async function continueInChat(a) {
  try {
    const resp = await api("/api/threads/seed", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title: `Анализ оттока №${a.clientId}`,
        analysis_text: a.text,
        client_id: a.clientId,
      }),
    });
    const data = await resp.json();
    await loadThreads();
    state.panelOpen = false;
    location.hash = `#/c/${encodeURIComponent(data.thread_id)}`;
  } catch (e) {
    console.warn("Не удалось создать чат:", e.message);
    const label = $("#cp-agent-label");
    if (label) label.textContent = `copilot › ошибка: ${e.message}`;
  }
}

async function deleteThread(threadId) {
  state.confirmDelete = null;
  try {
    await api(`/api/threads/${encodeURIComponent(threadId)}`, { method: "DELETE" });
  } catch (e) {
    console.warn("Не удалось удалить чат:", e.message);
  }
  // Из списка убираем независимо от ответа: сервер мог удалить, а ответ потеряться
  state.threads = state.threads.filter((t) => t.thread_id !== threadId);
  if (state.threadId === threadId) {
    state.threadId = null;
    state.messages = [];
    if (location.hash.startsWith("#/c/")) {
      location.hash = "#/chats"; // route() перерисует вид
    } else {
      renderChatView();
    }
  }
  renderSidebar();
}

/* ---------- обработчики ---------- */

document.addEventListener("click", async (e) => {
  const target = e.target.closest("button");
  if (!target) return;

  if (target.closest("#new-chat")) {
    if (state.sending) return;
    state.threadId = null;
    state.messages = [];
    if (location.hash === "#/chats") {
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
  if (target.closest("#panel-close")) {
    closePanel();
    return;
  }
  if (target.closest("#panel-continue")) {
    const a = state.analyses.get(state.panelClientId);
    if (a && !target.disabled) continueInChat(a);
    return;
  }
  if (target.dataset.panelTab) {
    setPanelTab(target.dataset.panelTab);
    return;
  }
  if (target.dataset.del !== undefined) {
    // Двухшаговое удаление: первый клик — подтверждение, второй — удаление
    if (state.sending) return;
    const id = target.dataset.del;
    if (state.confirmDelete === id) {
      deleteThread(id);
    } else {
      state.confirmDelete = id;
      renderSidebar();
      setTimeout(() => {
        if (state.confirmDelete === id) {
          state.confirmDelete = null;
          if (state.view === "chats") renderSidebar();
        }
      }, 3000);
    }
    return;
  }
  if (target.dataset.thread) {
    state.confirmDelete = null;
    if (state.sending) return;
    location.hash = `#/c/${encodeURIComponent(target.dataset.thread)}`;
    return;
  }
  if (target.dataset.filter) {
    state.clientsFilter = target.dataset.filter;
    state.clientsPage = 0; // фильтр меняет выборку — с первой страницы
    renderSidebar();
    if (state.view === "clients") loadClients();
    return;
  }
  if (target.dataset.analyze !== undefined) {
    const clientId = Number(target.dataset.analyze);
    // Рядом с таблицей строка обычно есть; если нет (deep-link) — подтягиваем по id
    const row = state.clients.find((c) => c.client_id === clientId) || await fetchClient(clientId);
    openAnalysis(clientId, row || undefined);
    return;
  }
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
