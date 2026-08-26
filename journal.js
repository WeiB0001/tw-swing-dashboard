/* =========================================================================
   journal.js — 我的交易記帳（首頁與 portfolio.html 共用同一份）

   資料只存在瀏覽器 LocalStorage，沒有伺服器、沒有登入。
   最新價完全重用 dashboard 已經算好的資料：
     - 首頁有 <script id="px-data"> → 解析後順手寫進 LocalStorage
     - portfolio.html 沒有那份資料 → 直接讀 LocalStorage
   所以切到交易頁不會多打任何一次股票 API。

   同一支檔案在兩個頁面跑，靠「元素存不存在」決定要畫哪些區塊：
     首頁           只有 #jn-modal 與卡片上的 [data-jbuy]
     portfolio.html 有 #jn-overview / #jn-open / #jn-history / #jn-chart / #jn-stats
   ========================================================================= */
(function () {
  "use strict";

  var KEY = "tw_swing_trade_journal_v1";        // 交易明細（兩頁共用，只有這一份）
  var SNAP = "tw_swing_trade_snapshots_v1";     // 每日累積報酬快照
  var PXKEY = "tw_swing_latest_prices_v1";      // 最新價快取（首頁寫、交易頁讀）

  /* ---------------------------------------------------------------
     最新價：首頁與這一頁共用同一份 data/latest_prices.json。

     取得順序（不各自保存舊價，也不多打任何 API）：
       1. data/latest_prices.json ── 每次 Actions 更新行情後產生，涵蓋全市場，
          所以就算某檔沒進首頁 Top10 也查得到價格
       2. 頁面內嵌的 #px-data ── 首頁本來就有，當作第一次載入前的墊底
       3. LocalStorage 快取 ── 行情抓不到時沿用最後有效價，並顯示它的資料日

     每一檔都帶自己的 market_date，非交易日或停牌就顯示那一檔最後有效的交易日，
     **不會用 0 或舊資料假裝成今天的價格**。
     --------------------------------------------------------------- */
  var PX = {}, PX_DATE = "", PX_UPDATED = "", PX_SOURCE = "";

  function adoptPrices(map, date, updated, source) {
    if (!map) return false;
    var n = 0;
    for (var k in map) { if (map.hasOwnProperty(k)) { n++; break; } }
    if (!n) return false;
    PX = map;
    PX_DATE = date || "";
    PX_UPDATED = updated || "";
    PX_SOURCE = source || "";
    try {
      localStorage.setItem(PXKEY, JSON.stringify({
        date: PX_DATE, updated_at: PX_UPDATED, source: PX_SOURCE, prices: PX
      }));
    } catch (e) {}
    return true;
  }

  (function initPrices() {
    // 先用頁面內嵌的（首頁才有），沒有就用上次快取，確保第一次 render 就有價格
    var el = document.getElementById("px-data");
    if (el) {
      try {
        var m = JSON.parse(el.textContent || "{}") || {};
        var conv = {};
        for (var k in m) {
          if (!m.hasOwnProperty(k)) continue;
          conv[k] = { name: m[k][0], price: m[k][1], prev_close: m[k][2] };
        }
        adoptPrices(conv, el.dataset.date || "", "", "頁面內嵌");
      } catch (e) {}
    }
    if (!PX_DATE) {
      try {
        var raw = JSON.parse(localStorage.getItem(PXKEY) || "{}");
        if (raw && raw.prices) {
          PX = raw.prices;
          PX_DATE = raw.date || "";
          PX_UPDATED = raw.updated_at || "";
          PX_SOURCE = raw.source || "本機快取";
        }
      } catch (e) {}
    }
  })();

  function refreshPrices() {
    // 抓共用的最新行情；失敗就維持現有的（上面已經墊底過），不清空
    return fetch("./data/latest_prices.json?t=" + Date.now(), { cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : Promise.reject(new Error("bad")); })
      .then(function (j) {
        if (!j || !j.prices) return false;
        return adoptPrices(j.prices, j.market_date, j.updated_at, j.source);
      })
      .catch(function () { return false; });
  }

  function px(code) {
    var v = PX[String(code || "").trim()];
    if (!v) return null;
    if (Array.isArray(v)) {                       // 舊格式（頁面內嵌）也吃
      return { name: v[0], last: num(v[1]), prev: v[2] == null ? null : num(v[2]),
               date: PX_DATE };
    }
    return { name: v.name, last: num(v.price),
             prev: v.prev_close == null ? null : num(v.prev_close),
             date: v.market_date || PX_DATE };
  }

  /* ---------------- 小工具 ---------------- */
  function num(x) { var v = parseFloat(x); return isFinite(v) ? v : null; }
  function today() {
    var d = new Date();
    return d.getFullYear() + "-" + String(d.getMonth() + 1).padStart(2, "0")
                           + "-" + String(d.getDate()).padStart(2, "0");
  }
  function money(v, sign) {
    if (v === null || v === undefined || !isFinite(v)) return "--";
    return (sign && v > 0 ? "+" : "") + "NT$" + Math.round(v).toLocaleString("en-US");
  }
  function pct(v) {
    if (v === null || v === undefined || !isFinite(v)) return "--";
    return (v > 0 ? "+" : "") + v.toFixed(2) + "%";
  }
  function cls(v) { return v > 0 ? "up" : (v < 0 ? "down" : ""); }
  function days(a, b) {
    var t1 = Date.parse(a), t2 = Date.parse(b || today());
    if (!isFinite(t1) || !isFinite(t2)) return null;
    return Math.max(0, Math.round((t2 - t1) / 86400000));
  }
  function esc(t) {
    return String(t == null ? "" : t).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function round2(v) { return (v === null || !isFinite(v)) ? null : Math.round(v * 100) / 100; }
  function $(id) { return document.getElementById(id); }

  /* ---------------- 讀寫（壞掉一律安全回空陣列） ---------------- */
  function load() {
    try {
      var raw = localStorage.getItem(KEY);
      if (!raw) return [];
      var arr = JSON.parse(raw);
      return Array.isArray(arr) ? arr.filter(valid) : [];
    } catch (e) { return []; }
  }
  function save(arr) { try { localStorage.setItem(KEY, JSON.stringify(arr)); } catch (e) {} }
  function valid(t) {
    return t && typeof t === "object" && t.symbol &&
           num(t.buy_price) > 0 && num(t.shares) > 0;
  }
  function uid() {
    return "t" + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
  }

  /* ---------------- 單筆計算 ---------------- */
  function calc(t) {
    var shares = num(t.shares) || 0;
    var cost = (num(t.buy_price) || 0) * shares + (num(t.buy_fee) || 0);
    var o = { cost: cost, shares: shares, open: t.status !== "CLOSED" };

    if (o.open) {
      var p = px(t.symbol);
      o.last = p ? p.last : null;
      o.prev = p ? p.prev : null;
      o.mv = o.last === null ? null : o.last * shares;
      o.pnl = o.mv === null ? null : o.mv - cost;
      o.ret = (o.pnl === null || cost <= 0) ? null : o.pnl / cost * 100;
      // 今日損益：沒有昨收就是 null，不猜
      o.today = (o.last === null || o.prev === null) ? null : (o.last - o.prev) * shares;
      o.today_pct = (o.last === null || !o.prev) ? null : (o.last / o.prev - 1) * 100;
      o.held = days(t.buy_date);
    } else {
      var proceeds = (num(t.sell_price) || 0) * shares - (num(t.sell_fee) || 0);
      o.pnl = proceeds - cost;
      o.ret = cost > 0 ? o.pnl / cost * 100 : null;
      o.held = days(t.buy_date, t.sell_date);
      o.proceeds = proceeds;
    }
    return o;
  }

  /* ---------------- 每日快照：同一天只留一筆 ---------------- */
  function snapshots() {
    try {
      var a = JSON.parse(localStorage.getItem(SNAP) || "[]");
      return Array.isArray(a) ? a : [];
    } catch (e) { return []; }
  }
  function snapshot(sum) {
    if (!sum || !sum.trades) return;
    var list = snapshots().filter(function (s) { return s && s.date !== today(); });
    list.push({
      date: today(),
      total_cost: round2(sum.total_cost), market_value: round2(sum.market_value),
      realized_pnl: round2(sum.realized), unrealized_pnl: round2(sum.unrealized),
      total_pnl: round2(sum.total_pnl), return_pct: round2(sum.return_pct)
    });
    list.sort(function (a, b) { return a.date < b.date ? -1 : 1; });
    try { localStorage.setItem(SNAP, JSON.stringify(list.slice(-400))); } catch (e) {}
  }

  /* ---------------- 彙總 ---------------- */
  function summarize(list) {
    var s = { total_cost: 0, market_value: 0, unrealized: 0, realized: 0,
              today: null, trades: list.length, closed: 0, open: 0,
              wins: 0, win_sum: 0, losses: 0, loss_sum: 0, unpriced: 0 };
    list.forEach(function (t) {
      var c = calc(t);
      s.total_cost += c.cost;
      if (c.open) {
        s.open++;
        if (c.mv === null) { s.unpriced++; return; }
        s.market_value += c.mv;
        s.unrealized += c.pnl;
        if (c.today !== null) s.today = (s.today === null ? 0 : s.today) + c.today;
      } else {
        s.closed++;
        s.realized += c.pnl;
        if (c.pnl > 0) { s.wins++; s.win_sum += c.pnl; }
        else { s.losses++; s.loss_sum += c.pnl; }
      }
    });
    s.total_pnl = s.realized + s.unrealized;                 // 總收益 = 已實現 + 未實現
    s.return_pct = s.total_cost > 0 ? s.total_pnl / s.total_cost * 100 : null;
    s.win_rate = s.closed > 0 ? s.wins / s.closed * 100 : null;
    s.avg_win = s.wins > 0 ? s.win_sum / s.wins : null;
    s.avg_loss = s.losses > 0 ? s.loss_sum / s.losses : null;
    return s;
  }

  /* =================================================================
     畫面
     ================================================================= */
  var filter = "all";

  function render() {
    var list = load();
    var sum = summarize(list);
    renderOverview(sum);
    renderOpen(list);
    renderHistory(list);
    renderStats(sum);
    snapshot(sum);              // 每天第一次開啟記一筆
    renderChart();
    markCards(list);
  }

  function cell(label, html) { return "<div><span>" + label + "</span>" + html + "</div>"; }

  /* 頂部四個主要數字 */
  function renderOverview(s) {
    var box = $("jn-overview");
    if (!box) return;
    box.innerHTML =
      cell("總持股市值", "<b>" + money(s.market_value) + "</b>") +
      cell("總投入成本", "<b>" + money(s.total_cost) + "</b>") +
      cell("總收益", '<b class="' + cls(s.total_pnl) + '">' + money(s.total_pnl, true) + "</b>") +
      cell("總報酬率", '<b class="' + cls(s.return_pct) + '">' + pct(s.return_pct) + "</b>");
    var sub = $("jn-overview-sub");
    if (sub) {
      sub.innerHTML =
        "<span>今日損益 <b class='" + (s.today === null ? "" : cls(s.today)) + "'>" +
          (s.today === null ? "--" : money(s.today, true)) + "</b></span>" +
        "<span>持有中 <b>" + s.open + "</b> 檔</span>" +
        "<span>已結束 <b>" + s.closed + "</b> 筆</span>" +
        (PX_DATE ? "<span>報價日 <b>" + esc(PX_DATE) + "</b></span>" : "") +
        (PX_UPDATED ? "<span>行情更新 <b>" + esc(PX_UPDATED) + "</b></span>" : "") +
        (s.unpriced ? "<span class='jn-note'>" + s.unpriced + " 檔暫無最新價</span>" : "");
    }
  }

  /* 目前持股（只有 OPEN） */
  function renderOpen(list) {
    var box = $("jn-open");
    if (!box) return;
    var opens = list.filter(function (t) { return t.status !== "CLOSED"; });
    if (!opens.length) {
      box.innerHTML = '<div class="jn-empty">目前沒有持股。回<a href="./">首頁</a>' +
        '在任何一張卡片按「＋加入我的持股」，或用下方「手動新增交易」輸入非排行榜的股票。</div>';
      return;
    }
    box.innerHTML = opens.map(function (t) {
      var c = calc(t);
      var name = esc(t.name || (px(t.symbol) ? px(t.symbol).name : ""));
      return '<div class="jn-row">' +
        '<div class="jn-r1"><span class="c">' + esc(t.symbol) + '</span>' +
          '<span class="n">' + name + '</span>' +
          '<span class="pnl ' + (c.pnl === null ? "" : cls(c.pnl)) + '">' +
            (c.pnl === null ? "暫無最新價" : money(c.pnl, true) + " (" + pct(c.ret) + ")") +
          "</span></div>" +
        '<div class="jn-r2">' +
          "<span>買入 <b>" + esc(t.buy_date) + "</b></span>" +
          "<span>買入價 <b>" + (num(t.buy_price) || 0).toFixed(2) + "</b></span>" +
          "<span>股數 <b>" + c.shares + "</b></span>" +
          "<span>成本 <b>" + money(c.cost) + "</b></span>" +
          "<span>最新價 <b>" + (c.last === null ? "--" : c.last.toFixed(2)) + "</b></span>" +
          "<span>市值 <b>" + (c.mv === null ? "--" : money(c.mv)) + "</b></span>" +
          "<span>今日 <b class='" + (c.today === null ? "" : cls(c.today)) + "'>" +
            (c.today === null ? "--" : money(c.today, true) + " " + pct(c.today_pct)) + "</b></span>" +
          "<span>總損益 <b class='" + (c.pnl === null ? "" : cls(c.pnl)) + "'>" +
            (c.pnl === null ? "--" : money(c.pnl, true)) + "</b></span>" +
          "<span>報酬率 <b class='" + (c.ret === null ? "" : cls(c.ret)) + "'>" + pct(c.ret) + "</b></span>" +
          "<span>持有 <b>" + (c.held === null ? "--" : c.held + " 天") + "</b></span>" +
        "</div>" +
        (t.note ? '<div class="jn-note">' + esc(t.note) + "</div>" : "") +
        '<div class="jn-ops">' +
          '<button type="button" class="sell" data-act="sell" data-id="' + t.id + '">賣出 / 結束交易</button>' +
          '<button type="button" data-act="edit" data-id="' + t.id + '">編輯</button>' +
        "</div></div>";
    }).join("");
  }

  /* 交易紀錄（OPEN + CLOSED） */
  function renderHistory(list) {
    var box = $("jn-history");
    if (!box) return;
    var rows = list.filter(function (t) {
      var st = t.status === "CLOSED" ? "CLOSED" : "OPEN";
      return filter === "all" || st === filter;
    }).slice().sort(function (a, b) {
      return (b.buy_date || "") < (a.buy_date || "") ? -1 : 1;
    });
    if (!rows.length) { box.innerHTML = '<div class="jn-empty">沒有符合的紀錄。</div>'; return; }

    box.innerHTML = rows.map(function (t) {
      var c = calc(t);
      var closed = !c.open;
      return '<div class="jn-row">' +
        '<div class="jn-r1"><span class="c">' + esc(t.symbol) + '</span>' +
          '<span class="n">' + esc(t.name || "") + '</span>' +
          '<span class="tag ' + (closed ? "done" : "live") + '">' +
            (closed ? "已實現" : "未實現") + "</span>" +
          '<span class="pnl ' + (c.pnl === null ? "" : cls(c.pnl)) + '">' +
            (c.pnl === null ? "--" : money(c.pnl, true) + " (" + pct(c.ret) + ")") +
          "</span></div>" +
        '<div class="jn-r2">' +
          "<span>買 <b>" + esc(t.buy_date) + " @ " + (num(t.buy_price) || 0).toFixed(2) + "</b></span>" +
          "<span>賣 <b>" + (closed ? esc(t.sell_date) + " @ " + (num(t.sell_price) || 0).toFixed(2) : "--") + "</b></span>" +
          "<span>股數 <b>" + c.shares + "</b></span>" +
          "<span>持有 <b>" + (c.held === null ? "--" : c.held + " 天") + "</b></span>" +
        "</div>" +
        (t.note ? '<div class="jn-note">' + esc(t.note) + "</div>" : "") +
        '<div class="jn-ops">' +
          (closed ? "" : '<button type="button" class="sell" data-act="sell" data-id="' + t.id + '">賣出</button>') +
          '<button type="button" data-act="edit" data-id="' + t.id + '">編輯</button>' +
          '<button type="button" data-act="del" data-id="' + t.id + '">刪除</button>' +
        "</div></div>";
    }).join("");
  }

  /* 歷史收益的幾個數字 */
  function renderStats(s) {
    var box = $("jn-stats");
    if (!box) return;
    box.innerHTML =
      cell("已實現收益", '<b class="' + cls(s.realized) + '">' + money(s.realized, true) + "</b>") +
      cell("未實現收益", '<b class="' + cls(s.unrealized) + '">' + money(s.unrealized, true) + "</b>") +
      cell("累積總收益", '<b class="' + cls(s.total_pnl) + '">' + money(s.total_pnl, true) + "</b>") +
      cell("勝率", "<b>" + (s.win_rate === null ? "--" : s.win_rate.toFixed(0) + "%") + "</b>") +
      cell("完成交易數", "<b>" + s.closed + "</b>") +
      cell("平均獲利", '<b class="up">' + (s.avg_win === null ? "--" : money(s.avg_win)) + "</b>") +
      cell("平均虧損", '<b class="down">' + (s.avg_loss === null ? "--" : money(s.avg_loss)) + "</b>");
  }

  /* 折線圖：原生 SVG，沒有引入任何圖表套件 */
  function renderChart() {
    var box = $("jn-chart");
    if (!box) return;
    var snaps = snapshots().filter(function (s) { return s && isFinite(s.return_pct); });
    if (snaps.length < 2) {
      box.innerHTML = '<div class="jn-empty">' +
        (snaps.length ? "已記錄 1 天，明天再打開就會開始畫線。" :
         "還沒有紀錄。有持股之後，每天第一次打開網站會自動記一筆當日累積報酬。") + "</div>";
      return;
    }
    var W = 320, H = 120, PAD = 6;
    var vals = snaps.map(function (s) { return s.return_pct; });
    var lo = Math.min.apply(null, vals.concat([0]));
    var hi = Math.max.apply(null, vals.concat([0]));
    if (hi - lo < 1) { hi += 0.5; lo -= 0.5; }
    var sx = function (i) { return PAD + i * (W - PAD * 2) / (snaps.length - 1); };
    var sy = function (v) { return PAD + (hi - v) * (H - PAD * 2) / (hi - lo); };
    var pts = snaps.map(function (s, i) { return sx(i).toFixed(1) + "," + sy(s.return_pct).toFixed(1); });
    var area = "M" + pts[0] + "L" + pts.slice(1).join("L") +
               "L" + sx(snaps.length - 1).toFixed(1) + "," + sy(lo).toFixed(1) +
               "L" + sx(0).toFixed(1) + "," + sy(lo).toFixed(1) + "Z";
    var last = snaps[snaps.length - 1];

    box.innerHTML =
      '<svg class="jn-chart" viewBox="0 0 ' + W + " " + H + '" preserveAspectRatio="none" role="img" ' +
        'aria-label="累積報酬率走勢，最新 ' + last.return_pct.toFixed(2) + '%">' +
        '<path class="area" d="' + area + '"></path>' +
        '<line class="zero" x1="' + PAD + '" y1="' + sy(0).toFixed(1) + '" x2="' + (W - PAD) +
          '" y2="' + sy(0).toFixed(1) + '"></line>' +
        '<polyline class="line" points="' + pts.join(" ") + '"></polyline>' +
      "</svg>" +
      '<div class="jn-r2"><span>' + esc(snaps[0].date) + " ～ " + esc(last.date) +
        "（" + snaps.length + " 筆）</span><span>最新 <b class='" + cls(last.return_pct) + "'>" +
        pct(last.return_pct) + "</b></span><span>區間 <b>" + lo.toFixed(1) + "% ～ " +
        hi.toFixed(1) + "%</b></span></div>";
  }

  /* 首頁卡片按鈕狀態 */
  function markCards(list) {
    var held = {};
    list.forEach(function (t) { if (t.status !== "CLOSED") held[t.symbol] = 1; });
    Array.prototype.forEach.call(document.querySelectorAll("[data-jbuy]"), function (b) {
      var on = !!held[b.dataset.code];
      b.classList.toggle("done", on);
      b.textContent = on ? "✓ 已在我的持股" : "＋加入我的持股";
    });
  }

  /* =================================================================
     表單（買入 / 賣出 / 編輯共用）
     ================================================================= */
  var modal = $("jn-modal"), formBox = $("jn-form"), errBox = $("jn-err"), titleEl = $("jn-title");
  var mode = "buy", editing = null;

  function field(id, label, type, value, full, step) {
    return '<div class="jn-field' + (full ? " full" : "") + '">' +
      '<label for="jf-' + id + '">' + label + "</label>" +
      '<input id="jf-' + id + '" type="' + type + '"' +
        (type === "number" ? ' inputmode="decimal" step="' + (step || "any") + '"' : "") +
        ' value="' + esc(value == null ? "" : value) + '"></div>';
  }

  function openForm(kind, trade) {
    if (!modal) return;
    mode = kind;
    // 只有「編輯」「賣出」是在改既有紀錄；從卡片帶進來的只是預填值
    editing = (kind === "buy") ? null : (trade || null);
    errBox.hidden = true; errBox.textContent = "";
    var t = trade || {};
    if (kind === "sell") {
      var p = px(t.symbol);
      titleEl.textContent = "賣出 " + t.symbol + " " + (t.name || "");
      formBox.innerHTML =
        field("sell_date", "賣出日期", "date", t.sell_date || today()) +
        field("sell_price", "賣出價格", "number", t.sell_price || (p ? p.last : "")) +
        field("sell_fee", "賣出手續費（含稅，可留 0）", "number", t.sell_fee || 0, true);
    } else {
      titleEl.textContent = kind === "edit" ? "編輯交易" : "新增買入";
      formBox.innerHTML =
        field("symbol", "股票代號", "text", t.symbol || "") +
        field("name", "股票名稱", "text", t.name || "") +
        field("buy_date", "買入日期", "date", t.buy_date || today()) +
        field("buy_price", "買入價格", "number", t.buy_price || "") +
        field("shares", "股數（一張＝1000 股）", "number", t.shares || "", false, "1") +
        field("buy_fee", "手續費（可留 0）", "number", t.buy_fee == null ? 0 : t.buy_fee) +
        field("note", "備註（可留空）", "text", t.note || "", true);
      if (kind === "edit" && t.status === "CLOSED") {
        formBox.innerHTML +=
          field("sell_date", "賣出日期", "date", t.sell_date || "") +
          field("sell_price", "賣出價格", "number", t.sell_price || "") +
          field("sell_fee", "賣出手續費", "number", t.sell_fee || 0, true);
      }
    }
    modal.hidden = false;
    var first = formBox.querySelector("input");
    if (first) setTimeout(function () { first.focus(); }, 50);
  }
  function closeForm() { if (modal) { modal.hidden = true; } editing = null; }
  function val(id) { var e = $("jf-" + id); return e ? e.value.trim() : ""; }
  function fail(msg) { errBox.textContent = msg; errBox.hidden = false; }

  function saveForm() {
    var list = load();
    if (mode === "sell") {
      var sp = num(val("sell_price"));
      if (!(sp > 0)) return fail("賣出價格必須大於 0。");
      var sf = num(val("sell_fee")) || 0;
      if (sf < 0) return fail("手續費不能是負數。");
      var sd = val("sell_date") || today();
      list.forEach(function (t) {
        if (editing && t.id === editing.id) {
          t.sell_date = sd; t.sell_price = sp; t.sell_fee = sf; t.status = "CLOSED";
        }
      });
    } else {
      var sym = val("symbol").toUpperCase();
      var bp = num(val("buy_price")), sh = num(val("shares")), bf = num(val("buy_fee")) || 0;
      if (!sym) return fail("請輸入股票代號。");
      if (!(bp > 0)) return fail("買入價格必須大於 0。");
      if (!(sh > 0)) return fail("股數必須大於 0。");
      if (bf < 0) return fail("手續費不能是負數。");

      var rec = {
        id: (editing && editing.id) || uid(),
        symbol: sym,
        name: val("name") || (px(sym) ? px(sym).name : ""),
        buy_date: val("buy_date") || today(),
        buy_price: bp, shares: sh, buy_fee: bf,
        sell_date: (editing && editing.sell_date) || null,
        sell_price: (editing && editing.sell_price) || null,
        sell_fee: (editing && editing.sell_fee) || 0,
        status: (editing && editing.status) || "OPEN",
        note: val("note") || ""
      };
      if (editing && editing.status === "CLOSED") {
        var esp = num(val("sell_price"));
        if (!(esp > 0)) return fail("已結束的交易，賣出價格必須大於 0。");
        rec.sell_date = val("sell_date") || editing.sell_date;
        rec.sell_price = esp;
        rec.sell_fee = num(val("sell_fee")) || 0;
      }
      list = editing ? list.map(function (t) { return t.id === rec.id ? rec : t; })
                     : list.concat([rec]);
    }
    save(list); closeForm(); render();
    var toast = $("jn-saved");
    if (toast) {
      toast.hidden = false;
      setTimeout(function () { toast.hidden = true; }, 2600);
    }
  }

  /* ---------------- 匯出 / 匯入 ---------------- */
  function exportJson() {
    var payload = { version: 1, exported_at: new Date().toISOString(),
                    trades: load(), snapshots: snapshots() };
    var blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "我的交易記帳-" + today() + ".json";
    document.body.appendChild(a); a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 500);
  }

  function importJson(file) {
    var reader = new FileReader();
    reader.onload = function () {
      var data;
      try { data = JSON.parse(reader.result); }
      catch (e) { alert("檔案不是合法的 JSON，原本的資料沒有被更動。"); return; }
      var trades = Array.isArray(data) ? data : (data && data.trades);
      if (!Array.isArray(trades)) {
        alert("看不到交易資料（缺少 trades 陣列），原本的資料沒有被更動。"); return;
      }
      var clean = trades.filter(valid).map(function (t) {
        return { id: t.id || uid(), symbol: String(t.symbol), name: t.name || "",
                 buy_date: t.buy_date || today(), buy_price: num(t.buy_price),
                 shares: num(t.shares), buy_fee: num(t.buy_fee) || 0,
                 sell_date: t.sell_date || null, sell_price: num(t.sell_price),
                 sell_fee: num(t.sell_fee) || 0,
                 status: t.status === "CLOSED" ? "CLOSED" : "OPEN", note: t.note || "" };
      });
      if (!clean.length) { alert("檔案裡沒有任何合法的交易紀錄，原本的資料沒有被更動。"); return; }
      if (!confirm("將匯入 " + clean.length + " 筆交易，並覆蓋目前的記帳資料。確定嗎？")) return;
      save(clean);
      if (data && Array.isArray(data.snapshots)) {
        try { localStorage.setItem(SNAP, JSON.stringify(data.snapshots.slice(-400))); } catch (e) {}
      }
      render();
      alert("已匯入 " + clean.length + " 筆交易。");
    };
    reader.readAsText(file);
  }

  /* ---------------- 事件 ---------------- */
  Array.prototype.forEach.call(document.querySelectorAll("[data-jbuy]"), function (b) {
    b.addEventListener("click", function (e) {
      e.preventDefault(); e.stopPropagation();   // 按鈕在 summary 裡，別把卡片展開
      openForm("buy", { symbol: b.dataset.code, name: b.dataset.name, buy_price: b.dataset.price });
    });
  });

  var addBtn = $("jn-add");
  if (addBtn) addBtn.addEventListener("click", function () { openForm("buy", null); });
  var cancel = $("jn-cancel");
  if (cancel) cancel.addEventListener("click", closeForm);
  var saveBtn = $("jn-save");
  if (saveBtn) saveBtn.addEventListener("click", saveForm);
  if (modal) modal.addEventListener("click", function (e) { if (e.target === modal) closeForm(); });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && modal && !modal.hidden) closeForm();
  });

  document.addEventListener("click", function (e) {
    var b = e.target.closest ? e.target.closest("[data-act]") : null;
    if (!b) return;
    var list = load();
    var t = list.filter(function (x) { return x.id === b.dataset.id; })[0];
    if (!t) return;
    if (b.dataset.act === "sell") openForm("sell", t);
    else if (b.dataset.act === "edit") openForm("edit", t);
    else if (b.dataset.act === "del") {
      if (confirm("確定刪除 " + t.symbol + " " + (t.name || "") + " 這筆交易？刪除後無法復原。")) {
        save(list.filter(function (x) { return x.id !== t.id; }));
        render();
      }
    }
  });

  Array.prototype.forEach.call(document.querySelectorAll("[data-jfilter]"), function (b) {
    b.addEventListener("click", function () {
      filter = b.dataset.jfilter;
      Array.prototype.forEach.call(document.querySelectorAll("[data-jfilter]"), function (x) {
        x.setAttribute("aria-pressed", x === b ? "true" : "false");
      });
      renderHistory(load());
    });
  });

  var ex = $("jn-export"); if (ex) ex.addEventListener("click", exportJson);
  var im = $("jn-import"), file = $("jn-file");
  if (im && file) {
    im.addEventListener("click", function () { file.click(); });
    file.addEventListener("change", function () {
      if (file.files && file.files[0]) importJson(file.files[0]);
      file.value = "";
    });
  }

  // 另一個分頁改了資料就同步（例如首頁加了一筆，交易頁自動更新）
  window.addEventListener("storage", function (e) {
    if (e.key === KEY || e.key === SNAP) render();
  });

  render();

  // 載入後再向伺服器要一次最新行情，拿到就重畫（買入紀錄不會被動到）
  refreshPrices().then(function (ok) { if (ok) render(); });
})();
