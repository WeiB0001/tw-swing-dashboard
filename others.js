/* =========================================================================
   others.js — 其他產業（非科技／金融／ETF）

   只讀 data/others.json，不呼叫任何股票 API。
   用的是首頁同一套技術分數，但這裡沒有歷史勝率分層，所以只排技術分數。
   ========================================================================= */
(function () {
  "use strict";

  var rows = [], sortBy = "score", q = "";
  var listEl = document.getElementById("rd-list");
  var countEl = document.getElementById("rd-count");
  var totalEl = document.getElementById("rd-total");
  var statsEl = document.getElementById("rd-stats");
  var moreEl = document.getElementById("rd-more");
  var qEl = document.getElementById("rd-q");

  function esc(t) {
    return String(t == null ? "" : t).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function cls(v) { return v > 0 ? "up" : (v < 0 ? "down" : ""); }

  function sorted() {
    var out = rows.filter(function (r) {
      if (!q) return true;
      return (r.symbol + " " + (r.name || "") + " " + (r.sector || "")).toLowerCase().indexOf(q) >= 0;
    });
    var key = { score: "score", chg: "chg_pct", vol: "vol_ratio" }[sortBy] || "score";
    return out.slice().sort(function (a, b) { return (b[key] || 0) - (a[key] || 0); });
  }

  function render() {
    if (!listEl) return;
    var hits = sorted();
    if (countEl) {
      countEl.innerHTML = (q ? "搜尋「" + esc(q) + "」：" : "") +
        "<b>" + hits.length + "</b> 檔達標";
    }
    if (!hits.length) {
      listEl.innerHTML = '<div class="jn-empty">今天其他產業沒有達標的標的，' +
        "或搜尋條件沒有符合的。門檻是技術分數 45 分以上。</div>";
      return;
    }
    listEl.innerHTML = hits.map(function (r, i) {
      return '<div class="jn-row">' +
        '<div class="jn-r1"><span class="c">' + (i + 1) + "　" + esc(r.symbol) + "</span>" +
          '<span class="n">' + esc(r.name || "") + "</span>" +
          '<span class="pnl ' + cls(r.chg_pct) + '">' + r.close.toFixed(2) +
            " (" + (r.chg_pct > 0 ? "+" : "") + r.chg_pct.toFixed(2) + "%)</span></div>" +
        '<div style="margin-top:7px">' +
          '<span class="rd-tag">' + esc(r.sector || "其他") + "</span>" +
          '<span class="rd-tag up">' + esc(r.kind || "") + "</span>" +
          '<span class="rd-tag core">技術分數 ' + Math.round(r.score) + "</span>" +
          '<span class="rd-tag low">風險 ' + esc(r.risk_level || "-") + "</span></div>" +
        '<div class="jn-r2">' +
          "<span>" + esc(r.headline || "") + "</span>" +
          "<span>RSI <b>" + r.rsi.toFixed(0) + "</b></span>" +
          "<span>量能 <b>" + r.vol_ratio.toFixed(1) + "×</b></span>" +
          "<span>風報 <b>" + (r.rr_ratio || 0).toFixed(1) + ":1</b></span>" +
          "<span>預期空間 <b>+" + (r.swing_low || 0).toFixed(1) + "% ～ +" +
            (r.swing_high || 0).toFixed(1) + "%</b></span>" +
        "</div>" +
        '<div class="jn-note">主要風險：' + esc(r.main_risk || "") + "</div>" +
        "</div>";
    }).join("");
    if (moreEl) moreEl.hidden = true;
  }

  Array.prototype.forEach.call(document.querySelectorAll("[data-sort]"), function (b) {
    b.addEventListener("click", function () {
      sortBy = b.dataset.sort;
      Array.prototype.forEach.call(document.querySelectorAll("[data-sort]"), function (x) {
        x.setAttribute("aria-pressed", x === b ? "true" : "false");
      });
      render();
    });
  });
  if (qEl) qEl.addEventListener("input", function () {
    q = qEl.value.trim().toLowerCase();
    render();
  });

  fetch("./data/others.json?t=" + Date.now(), { cache: "no-store" })
    .then(function (r) { return r.ok ? r.json() : null; })
    .then(function (j) {
      if (!j || !Array.isArray(j.rows)) throw new Error("no data");
      rows = j.rows;
      if (totalEl) totalEl.textContent = rows.length;
      if (statsEl) {
        statsEl.innerHTML += "<br><b>掃描範圍</b>：其他產業中當日成交金額前 " +
          (j.scanned || 0) + " 檔，列出技術分數達標的 " + rows.length + " 檔" +
          (j.date ? "　資料日 " + j.date : "");
      }
      render();
    })
    .catch(function () {
      if (totalEl) totalEl.textContent = "0";
      if (listEl) listEl.innerHTML = '<div class="jn-empty">' +
        "還沒有資料。等下一次每日更新跑完（<code>data/others.json</code> 產生）就會出現。</div>";
      if (countEl) countEl.textContent = "";
    });
})();
