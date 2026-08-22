/* =========================================================================
   sw.js — Service Worker
   目的：讓儀表板可以加到手機主畫面、開啟秒開、離線（捷運、電梯、地下室）
        也看得到上一次的結果。

   策略分兩種：
     - 頁面與資料（index.html、latest.json）→ 網路優先，失敗才用快取
       （盤後資料每天會變，一定要拿最新的）
     - 外觀資源（圖示、manifest）→ 快取優先
       （幾乎不會變，直接用快取最快）

   不需要改版本號：頁面每天更新時，網路優先會自動抓到新的。
   ========================================================================= */

const CACHE = "tw-swing-v1";

// 安裝時先把外殼存起來，第一次離線也打得開
const SHELL = [
  "./",
  "./index.html",
  "./portfolio.html",
  "./radar.html",
  "./radar.js",
  "./journal.js",
  "./journal.css",
  "./manifest.webmanifest",
  "./icons/icon-192.png",
  "./icons/icon-512.png",
  "./icons/apple-touch-icon.png",
];

self.addEventListener("install", (e) => {
  e.waitUntil(
    caches.open(CACHE)
      .then((c) => c.addAll(SHELL))
      .catch(() => {})          // 有任何一個檔案抓不到也不要讓安裝失敗
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;

  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;   // 外部資源不攔截

  const isFresh =
    req.mode === "navigate" ||
    url.pathname.endsWith(".html") ||
    url.pathname.endsWith(".json");

  if (isFresh) {
    // 網路優先：拿到新的就順手更新快取，斷網才退回快取
    e.respondWith(
      fetch(req)
        .then((res) => {
          const copy = res.clone();
          caches.open(CACHE).then((c) => c.put(req, copy)).catch(() => {});
          return res;
        })
        .catch(() =>
          caches.match(req).then((hit) => hit || caches.match("./index.html"))
        )
    );
  } else {
    // 快取優先：圖示這類幾乎不變的資源
    e.respondWith(
      caches.match(req).then((hit) =>
        hit ||
        fetch(req).then((res) => {
          const copy = res.clone();
          caches.open(CACHE).then((c) => c.put(req, copy)).catch(() => {});
          return res;
        })
      )
    );
  }
});
