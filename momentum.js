/* =========================================================================
   momentum.js — 明日強勢預測

   只讀 data/momentum_forecast.json，不呼叫任何股票 API。
   這一頁跟首頁排行是兩套東西：首頁看的是模型勝率與進場品質，
   這裡只回答「今天漲很多的股票，明天還會不會續強」。
   ========================================================================= */
(function () {
  "use strict";

  var TOP = 10;
  var rows = [], kind = "all", market = "all", q = "";

  var listEl = document.getElementById("rd-list");
  var countEl = document.getElementById("rd-count");
  var totalEl = document.getElementById("rd-total");
  var statsEl = document.getElementById("rd-stats");
  var moreEl = document.getElementById("rd-more");
  var qEl = document.getElementById("rd-q");
  var shown = TOP;

  function esc(t) {
    return String(t == null ? "" : t).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function wan(v) {
    if (!isFinite(v)) return "--";
    if (v >= 100000000) return (v / 100000000).toFixed(1) + " 億";
    return Math.round(v / 10000).toLocaleString("en-US") + " 萬";
  }
  var MARKET = { TWSE: "上市", TPEx: "上櫃", ETF: "ETF" };
  var KIND = { tech: "科技", finance: "金融", etf: "ETF" };

  function match(r) {
    if (kind !== "all" && r.kind !== kind) return false;
    if (market !== "all" && r.market !== market) return false;
    if (q && (r.symbol + " " + (r.name || "") + " " + (r.sector || ""))
             .toLowerCase().indexOf(q) < 0) return false;
    return true;
  }

  function histHtml(r) {
    if (!r.hist) {
      // 樣本不足就直說，不生一個假勝率
      return '<div class="mo-hist none">歷史樣本不足，無法估計隔日續強機率</div>';
    }
    var h = r.hist;
    return '<div class="mo-hist">' +
      "<span>隔日勝率 <b>" + h.calibrated_win_rate + "%</b></span>" +
      "<span>平均 <b class='" + (h.avg_r1 > 0 ? "up" : "down") + "'>" +
        (h.avg_r1 > 0 ? "+" : "") + h.avg_r1.toFixed(2) + "%</b></span>" +
      "<span>3 日 <b class='" + (h.avg_r3 > 0 ? "up" : "down") + "'>" +
        (h.avg_r3 > 0 ? "+" : "") + h.avg_r3.toFixed(2) + "%</b></span>" +
      "<span>樣本 <b>" + h.samples + "</b></span>" +
      '<span class="src">依據：' + esc(r.hist_source) + "</span></div>";
  }

  function render() {
    if (!listEl) return;
    var hits = rows.filter(match);
    if (countEl) {
      var bits = [];
      if (kind !== "all") bits.push(KIND[kind]);
      if (market !== "all") bits.push(MARKET[market]);
      if (q) bits.push("「" + esc(q) + "」");
      countEl.innerHTML = (bits.length ? bits.join(" + ") + "：" : "") +
        "<b>" + hits.length + "</b> 檔候選（今日漲幅 3% 以上的科技／金融／ETF）";
    }
    if (!hits.length) {
      listEl.innerHTML = '<div class="jn-empty">今天沒有符合條件的強勢股，' +
        "或篩選條件沒有命中。大盤弱勢時本來就可能一檔都沒有。</div>";
      if (moreEl) moreEl.hidden = true;
      return;
    }

    listEl.innerHTML = hits.slice(0, shown).map(function (r, i) {
      return '<div class="jn-row">' +
        '<div class="jn-r1"><span class="c">' + (i + 1) + "　" + esc(r.symbol) + "</span>" +
          '<span class="n">' + esc(r.name || "") + "</span>" +
          '<span class="pnl up">+' + r.chg_pct.toFixed(2) + "%</span></div>" +
        '<div style="margin-top:7px">' +
          '<span class="rd-tag">' + esc(MARKET[r.market] || r.market) + "</span>" +
          '<span class="rd-tag">' + esc(r.sector || KIND[r.kind] || "") + "</span>" +
          '<span class="rd-tag core">強勢分數 ' + Math.round(r.score) + "/100</span></div>" +
        '<div class="jn-r2">' +
          "<span>收盤 <b>" + r.close.toFixed(2) + "</b></span>" +
          "<span>量能 <b>" + r.vol_ratio.toFixed(1) + "×</b></span>" +
          "<span>收盤位置 <b>" + r.close_pos + "%</b></span>" +
          "<span>成交額 <b>" + wan(r.turnover) + "</b></span>" +
          "<span>距 20 日高 <b>-" + r.to_high20.toFixed(1) + "%</b></span>" +
        "</div>" +
        histHtml(r) +
        (r.pros && r.pros.length
          ? '<div class="mo-list ok">' + esc(r.pros.join("、")) + "</div>" : "") +
        (r.cons && r.cons.length
          ? '<div class="mo-list bad">⚠ ' + esc(r.cons.join("、")) + "</div>" : "") +
        "</div>";
    }).join("");

    if (moreEl) {
      moreEl.hidden = hits.length <= shown;
      moreEl.textContent = "顯示更多（還有 " + (hits.length - shown) + " 檔）";
    }
  }

  function bind(attr, set) {
    Array.prototype.forEach.call(document.querySelectorAll("[" + attr + "]"), function (b) {
      b.addEventListener("click", function () {
        set(b.getAttribute(attr));
        Array.prototype.forEach.call(document.querySelectorAll("[" + attr + "]"), function (x) {
          x.setAttribute("aria-pressed", x === b ? "true" : "false");
        });
        shown = TOP;
        render();
      });
    });
  }
  bind("data-kind", function (v) { kind = v; });
  bind("data-market", function (v) { market = v; });
  if (qEl) qEl.addEventListener("input", function () {
    q = qEl.value.trim().toLowerCase(); shown = TOP; render();
  });
  if (moreEl) moreEl.addEventListener("click", function () { shown += TOP; render(); });

  fetch("./data/momentum_forecast.json?t=" + Date.now(), { cache: "no-store" })
    .then(function (r) { return r.ok ? r.json() : null; })
    .then(function (j) {
      if (!j || !Array.isArray(j.rows)) throw new Error("no data");
      rows = j.rows;
      if (totalEl) totalEl.textContent = Math.min(rows.length, TOP);
      if (statsEl && j.stats) {
        var o = j.stats.overall;
        statsEl.innerHTML += "<br><b>歷史統計</b>：" +
          (j.stats.total_samples || 0) + " 筆強勢日樣本" +
          (o ? "，整體隔日勝率 " + o.calibrated_win_rate + "%、平均 " +
               (o.avg_r1 > 0 ? "+" : "") + o.avg_r1 + "%" : "") +
          "。" + esc(j.stats.entry_rule || "") +
          (j.date ? "　資料日 " + esc(j.date) : "");
      }
      render();
    })
    .catch(function () {
      if (totalEl) totalEl.textContent = "0";
      if (listEl) listEl.innerHTML = '<div class="jn-empty">' +
        "還沒有預測資料。等下一次每日更新跑完（<code>data/momentum_forecast.json</code> 產生）就會出現。</div>";
      if (countEl) countEl.textContent = "";
    });
})();
