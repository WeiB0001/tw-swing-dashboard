/* =========================================================================
   radar.js — 雷達頁

   資料只讀 data/radar.json（build.py 每天產生），不呼叫任何股票 API。
   篩選可以組合：產業 × 市場 × 訊號 × 關鍵字搜尋。
   例如「科技 + 上櫃」就能專門找小型上櫃科技股。
   ========================================================================= */
(function () {
  "use strict";

  var PAGE = 40;                 // 一次顯示幾筆，其餘按「顯示更多」
  var rows = [], shown = PAGE;
  var f = { kind: "all", market: "all", sig: "all", q: "" };

  var listEl = document.getElementById("rd-list");
  var countEl = document.getElementById("rd-count");
  var moreEl = document.getElementById("rd-more");
  var totalEl = document.getElementById("rd-total");
  var statsEl = document.getElementById("rd-stats");
  var qEl = document.getElementById("rd-q");

  function esc(t) {
    return String(t == null ? "" : t).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function cls(v) { return v > 0 ? "up" : (v < 0 ? "down" : ""); }
  function wan(v) {
    if (!isFinite(v)) return "--";
    if (v >= 100000000) return (v / 100000000).toFixed(1) + " 億";
    return Math.round(v / 10000).toLocaleString("en-US") + " 萬";
  }
  var MARKET = { TWSE: "上市", TPEx: "上櫃", ETF: "ETF" };
  var KIND = { tech: "科技", finance: "金融", etf: "ETF" };

  function match(r) {
    if (f.kind !== "all" && r.kind !== f.kind) return false;
    if (f.market !== "all" && r.market !== f.market) return false;
    if (f.sig !== "all" && !(r.tags || []).some(function (t) { return t.key === f.sig; })) return false;
    if (f.q) {
      var hay = (r.symbol + " " + (r.name || "") + " " + (r.sector || "")).toLowerCase();
      if (hay.indexOf(f.q) < 0) return false;
    }
    return true;
  }

  function tagHtml(t) {
    var c = { quiet_volume: "hot", quiet_breakout: "hot", breakout: "up",
              oversold: "low", rebound: "up", extended: "" }[t.key] || "";
    return '<span class="rd-tag ' + c + '">' + t.icon + " " + esc(t.label) + "</span>";
  }

  function render() {
    if (!listEl) return;
    var hits = rows.filter(match);

    if (countEl) {
      var bits = [];
      if (f.kind !== "all") bits.push(KIND[f.kind]);
      if (f.market !== "all") bits.push(MARKET[f.market]);
      if (f.q) bits.push("「" + esc(f.q) + "」");
      countEl.innerHTML = (bits.length ? bits.join(" + ") + "：" : "") +
        "<b>" + hits.length + "</b> 檔（掃描 " + rows.length + " 檔）";
    }

    if (!hits.length) {
      listEl.innerHTML = '<div class="jn-empty">沒有符合的標的。' +
        "換個條件，或清掉搜尋關鍵字再看看。</div>";
      if (moreEl) moreEl.hidden = true;
      return;
    }

    listEl.innerHTML = hits.slice(0, shown).map(function (r) {
      var w52 = r.week52
        ? "<span>距 52 週高 <b>" + r.week52.from_high.toFixed(1) + "%</b></span>" +
          "<span>距 52 週低 <b>+" + r.week52.from_low.toFixed(1) + "%</b></span>"
        : "<span>資料未滿 52 週</span>";
      return '<div class="jn-row">' +
        '<div class="jn-r1">' +
          '<span class="c">' + esc(r.symbol) + "</span>" +
          '<span class="n">' + esc(r.name || "") + "</span>" +
          '<span class="pnl ' + cls(r.chg_pct) + '">' +
            r.close.toFixed(2) + " (" + (r.chg_pct > 0 ? "+" : "") + r.chg_pct.toFixed(2) + "%)" +
          "</span></div>" +
        '<div style="margin-top:7px">' +
          (r.is_core ? '<span class="rd-tag core">★ Core</span>' : "") +
          (r.tags || []).map(tagHtml).join("") +
        "</div>" +
        '<div class="jn-r2">' +
          "<span>" + esc(MARKET[r.market] || r.market) + " · " + esc(r.sector || KIND[r.kind] || "") + "</span>" +
          "<span>RSI <b>" + r.rsi.toFixed(0) + "</b></span>" +
          "<span>量能 <b>" + r.vol_ratio.toFixed(1) + "×</b></span>" +
          "<span>20日均額 <b>" + wan(r.avg_turnover) + "</b></span>" +
          "<span>距 20 日高 <b>-" + r.pct_below_high20.toFixed(1) + "%</b></span>" +
          w52 +
        "</div></div>";
    }).join("");

    if (moreEl) {
      moreEl.hidden = hits.length <= shown;
      moreEl.textContent = "顯示更多（還有 " + (hits.length - shown) + " 檔）";
    }
  }

  function bind(attr, key) {
    Array.prototype.forEach.call(document.querySelectorAll("[" + attr + "]"), function (b) {
      b.addEventListener("click", function () {
        f[key] = b.getAttribute(attr);
        Array.prototype.forEach.call(document.querySelectorAll("[" + attr + "]"), function (x) {
          x.setAttribute("aria-pressed", x === b ? "true" : "false");
        });
        shown = PAGE;
        render();
      });
    });
  }
  bind("data-kind", "kind");
  bind("data-market", "market");
  bind("data-sig", "sig");

  if (qEl) {
    qEl.addEventListener("input", function () {
      f.q = qEl.value.trim().toLowerCase();
      shown = PAGE;
      render();
    });
  }
  if (moreEl) moreEl.addEventListener("click", function () { shown += PAGE; render(); });

  fetch("./data/radar.json?t=" + Date.now(), { cache: "no-store" })
    .then(function (r) { return r.ok ? r.json() : null; })
    .then(function (j) {
      if (!j || !Array.isArray(j.rows)) throw new Error("no data");
      rows = j.rows;
      if (totalEl) totalEl.textContent = rows.length;
      if (statsEl && j.stats) {
        statsEl.innerHTML += "<br><b>股票池</b>：Core " + j.stats.core_total +
          " 檔｜Extended " + j.stats.extended_total + " 檔（科技 " + j.stats.tech +
          "、金融 " + j.stats.finance + "、ETF " + j.stats.etf +
          "｜上市 " + j.stats.twse + "、上櫃 " + j.stats.tpex + "）" +
          (j.date ? "　資料日 " + j.date : "");
      }
      render();
    })
    .catch(function () {
      if (totalEl) totalEl.textContent = "0";
      if (listEl) listEl.innerHTML = '<div class="jn-empty">' +
        "還沒有雷達資料。等下一次每日更新跑完（<code>data/radar.json</code> 產生）就會出現。" +
        "</div>";
      if (countEl) countEl.textContent = "";
    });
})();
