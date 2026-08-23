/* =========================================================================
   alerts.js — 盤中觸發提醒

   只讀 data/intraday.json（由 GitHub Actions 盤中每 5 分鐘更新）。
   不重算排名、勝率、EV、PF，也不碰回測——那些一律以收盤資料為準。

   資料取得順序：
     1. GitHub Pages 上優先讀 raw.githubusercontent.com 的最新檔
        （Pages 部署有頻率限制，不可能每 5 分鐘重新部署一次，
          直接讀 raw 才拿得到剛 commit 的盤中狀態）
     2. 讀不到就退回同站的 ./data/intraday.json
   ========================================================================= */
(function () {
  "use strict";

  var POLL_MS = 60000;                 // 每分鐘拉一次（來源本身 5 分鐘才更新）
  var NOTIFY_KEY = "tw_swing_alert_notified_v1";
  var NOTIFY_ON = "tw_swing_alert_notify_on_v1";

  var listEl = document.getElementById("al-list");
  var metaEl = document.getElementById("al-meta");
  var liveEl = document.getElementById("al-live");
  var liveText = document.getElementById("al-live-text");
  var notifyBtn = document.getElementById("al-notify");
  var refreshBtn = document.getElementById("al-refresh");

  var STATUS = {
    waiting: "未觸發",
    triggered: "已突破觸發價",
    near_stop: "接近停損",
    stopped: "已觸及停損",
    target: "已達目標一"
  };

  function esc(t) {
    return String(t == null ? "" : t).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function cls(v) { return v > 0 ? "up" : (v < 0 ? "down" : ""); }
  function n2(v) { return (v === null || v === undefined || !isFinite(v)) ? "--" : Number(v).toFixed(2); }

  /* ---- 資料來源 ---- */
  function sources() {
    var urls = [];
    var host = location.hostname || "";
    var m = host.match(/^([^.]+)\.github\.io$/);
    if (m) {
      var repo = (location.pathname.split("/").filter(Boolean)[0]) || "";
      if (repo && !repo.endsWith(".html")) {
        ["main", "master"].forEach(function (br) {
          urls.push("https://raw.githubusercontent.com/" + m[1] + "/" + repo +
                    "/" + br + "/data/intraday.json");
        });
      }
    }
    urls.push("./data/intraday.json");
    return urls;
  }

  function load() {
    var urls = sources(), i = 0;
    function next() {
      if (i >= urls.length) return Promise.reject(new Error("all failed"));
      var u = urls[i++];
      return fetch(u + (u.indexOf("?") < 0 ? "?" : "&") + "t=" + Date.now(),
                   { cache: "no-store" })
        .then(function (r) { return r.ok ? r.json() : Promise.reject(new Error("bad")); })
        .catch(next);
    }
    return next();
  }

  /* ---- 通知去重：同一檔同一狀態，每天只通知一次 ---- */
  function notifiedMap() {
    try {
      var o = JSON.parse(localStorage.getItem(NOTIFY_KEY) || "{}");
      return (o && typeof o === "object") ? o : {};
    } catch (e) { return {}; }
  }
  function markNotified(date, key) {
    var o = notifiedMap();
    if (o.date !== date) o = { date: date, keys: {} };   // 換一天就整個重置
    o.keys = o.keys || {};
    o.keys[key] = 1;
    try { localStorage.setItem(NOTIFY_KEY, JSON.stringify(o)); } catch (e) {}
  }
  function alreadyNotified(date, key) {
    var o = notifiedMap();
    return o.date === date && o.keys && o.keys[key];
  }

  function notifyOn() {
    try { return localStorage.getItem(NOTIFY_ON) === "1"; } catch (e) { return false; }
  }
  function syncNotifyBtn() {
    if (!notifyBtn) return;
    var on = notifyOn() && ("Notification" in window) && Notification.permission === "granted";
    notifyBtn.textContent = on ? "🔔 通知已開啟" : "🔔 開啟通知";
    notifyBtn.classList.toggle("primary", on);
  }

  function maybeNotify(data) {
    if (!notifyOn() || !("Notification" in window) || Notification.permission !== "granted") return;
    var date = data.trade_date || "";
    (data.rows || []).forEach(function (r) {
      // 只有真的發生事情才通知，「未觸發」不吵人
      if (["triggered", "target", "stopped", "near_stop"].indexOf(r.status) < 0) return;
      var key = r.symbol + "_" + r.status;
      if (alreadyNotified(date, key)) return;
      try {
        new Notification(r.symbol + " " + (r.name || "") + "　" + STATUS[r.status], {
          body: "現價 " + n2(r.price) + "｜觸發 " + n2(r.trigger) +
                "｜停損 " + n2(r.stop) + "\n盤中訊號，尚未收盤確認",
          tag: key
        });
        markNotified(date, key);
      } catch (e) {}
    });
  }

  /* ---- 畫面 ---- */
  function renderClosed(data) {
    if (liveEl) { liveEl.classList.remove("open"); }
    if (liveText) liveText.textContent = "休市中";
    if (listEl) {
      listEl.innerHTML = '<div class="jn-empty" style="text-align:center">' +
        '<div class="al-dash">--</div>' +
        (esc(data && data.note) || "目前非交易時段，盤中狀態不顯示。") +
        "<br>收盤後的正式排行與交易計畫請看<a href=\"./\">首頁</a>。</div>";
    }
    if (metaEl) {
      metaEl.innerHTML = data && data.generated_at
        ? "最後檢查 <b>" + esc(data.generated_at) + "</b>"
        : "";
    }
  }

  function render(data) {
    if (!data || !data.market_open) return renderClosed(data);

    if (liveEl) liveEl.classList.add("open");
    if (liveText) liveText.textContent = "盤中";

    var rows = data.rows || [];
    if (metaEl) {
      var bits = [];
      if (data.data_time) bits.push("報價時間 <b>" + esc(data.data_time) + "</b>");
      if (data.delay_seconds != null) {
        var d = data.delay_seconds;
        bits.push("資料延遲 <b>" + (d < 60 ? d + " 秒" : Math.round(d / 60) + " 分鐘") + "</b>");
      }
      bits.push("最後更新 <b>" + esc(data.generated_at || "--") + "</b>");
      bits.push(esc(data.source || ""));
      metaEl.innerHTML = bits.join("　·　") +
        "<br>報價為延遲資料，排程每 5 分鐘更新一次，實際可能再延遲數分鐘。";
    }

    if (!rows.length) {
      listEl.innerHTML = '<div class="jn-empty">' +
        (esc(data.note) || "目前沒有可追蹤的交易計畫。") + "</div>";
      return;
    }

    listEl.innerHTML = rows.map(function (r) {
      var st = r.status || "waiting";
      var b = r.basis || {};
      return '<div class="al-row ' + st + '">' +
        '<div class="jn-r1"><span class="c">' + esc(r.symbol) + "</span>" +
          '<span class="n">' + esc(r.name || "") + "</span>" +
          '<span class="pnl ' + cls(r.chg_pct) + '">' + n2(r.price) +
            (r.chg_pct == null ? "" : " (" + (r.chg_pct > 0 ? "+" : "") + r.chg_pct.toFixed(2) + "%)") +
          "</span></div>" +
        '<div style="margin-top:7px">' +
          '<span class="al-st ' + st + '">' + STATUS[st] + "</span>　" +
          '<span style="font-size:12px;color:#9FACBA">' + esc(r.status_text || "") + "</span></div>" +
        '<div class="al-lvls">' +
          "<div><span>觸發價</span><b>" + n2(r.trigger) + "</b></div>" +
          "<div><span>停損</span><b class='down'>" + n2(r.stop) + "</b></div>" +
          "<div><span>目標一</span><b class='up'>" + n2(r.target1) + "</b></div>" +
          "<div><span>量能</span><b>" + (r.vol_ratio ? r.vol_ratio.toFixed(1) + "×" : "--") + "</b></div>" +
        "</div>" +
        '<div class="al-basis">' +
          "<b>觸發價依據</b>：" + esc(b.trigger || "") + "<br>" +
          "<b>停損依據</b>：" + esc(b.stop || "") + "<br>" +
          "<b>目標一依據</b>：" + esc(b.target1 || "") +
          (b.volume ? "<br><b>量能依據</b>：" + esc(b.volume) : "") +
          (b.ma20 ? "<br><b>均線</b>：" + esc(b.ma20) : "") +
        "</div></div>";
    }).join("");

    maybeNotify(data);
  }

  function tick() {
    load().then(render).catch(function () {
      if (liveText) liveText.textContent = "無法取得";
      if (listEl && !listEl.innerHTML) {
        listEl.innerHTML = '<div class="jn-empty">還沒有盤中資料。' +
          "盤中排程跑過一次（產生 <code>data/intraday.json</code>）之後就會出現。</div>";
      }
    });
  }

  if (notifyBtn) {
    notifyBtn.addEventListener("click", function () {
      if (!("Notification" in window)) {
        alert("這個瀏覽器不支援通知。");
        return;
      }
      if (notifyOn() && Notification.permission === "granted") {
        try { localStorage.setItem(NOTIFY_ON, "0"); } catch (e) {}
        syncNotifyBtn();
        return;
      }
      Notification.requestPermission().then(function (p) {
        // 回傳 granted 還不夠，實際的 permission 才算數
        // （非 HTTPS 或系統層關閉時，回傳 granted 但 permission 仍是 denied）
        if (p === "granted" && Notification.permission === "granted") {
          try { localStorage.setItem(NOTIFY_ON, "1"); } catch (e) {}
        } else {
          try { localStorage.setItem(NOTIFY_ON, "0"); } catch (e) {}
          alert("瀏覽器沒有允許通知。\n請確認網站是 https、且系統與瀏覽器的通知權限都有打開。");
        }
        syncNotifyBtn();
      });
    });
  }
  if (refreshBtn) refreshBtn.addEventListener("click", tick);
  document.addEventListener("visibilitychange", function () {
    if (!document.hidden) tick();
  });

  syncNotifyBtn();
  tick();
  setInterval(tick, POLL_MS);
})();
