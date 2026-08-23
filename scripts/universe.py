# -*- coding: utf-8 -*-
"""
universe.py — Core / Extended 雙層股票池

只收三類：**科技電子、金融、ETF**。其他產業（食品、航運、生技、傳產…）一律不掃。

  Core     首頁「今日機會排行」用。高流動性、當日成交金額前段 + 主要科技金融 ETF，
           約 150～220 檔。門檻與原本一致，排行行為不會因為這次改動而變樣。
  Extended 只給 radar.html 與搜尋用，約 250～400 檔。門檻放寬成
           「20 日平均成交金額」，但仍擋掉成交量極低的殭屍股。

產業判定優先用 FinMind TaiwanStockInfo 的 industry_category（含上市與上櫃）；
抓不到才退回 config 裡的手動清單（TECH_SECTORS / FINANCE_STOCKS / TW_ETFS），
手動清單一律保留不刪。

兩層共用 data/history 的同一份快取，不會分別重抓。
"""

from __future__ import annotations

import logging

import pandas as pd

import config as C

log = logging.getLogger("universe")

_MARKET = {"twse": "TWSE", "tpex": "TPEx", "otc": "TPEx"}


# ---------------------------------------------------------------------------
# 分類
# ---------------------------------------------------------------------------
def classify(code: str, category: str = "", market: str = "") -> tuple[str | None, str]:
    """
    回傳 (類別, 產業標籤)。類別是 tech / finance / etf / None（None = 不收）。
    有 FinMind 的 industry_category 就用它，沒有才退回手動清單。
    """
    code = str(code or "").strip()
    cat = str(category or "").strip()

    # ETF：代號 00 開頭最準，產業分類是輔助
    if code.startswith("00") or cat in C.ETF_CATEGORIES:
        if code in C.TW_ETFS:
            return "etf", "ETF·" + C.TW_ETFS[code]
        return "etf", "ETF"

    if cat:
        if cat in C.TECH_CATEGORIES:
            return "tech", cat
        if cat in C.FINANCE_CATEGORIES:
            return "finance", cat
        # 有分類但不屬於指定三類 → 明確不收
        return None, cat

    # 沒有分類資料 → 退回手動清單
    if code in C.TECH_SECTORS:
        return "tech", C.TECH_SECTORS[code]
    if code in C.FINANCE_STOCKS:
        return "finance", C.FINANCE_STOCKS[code]
    return None, ""


def _is_leveraged_etf(code: str) -> bool:
    """槓桿 L／反向 R／債券 B 一律排除（期貨型、商品型多半也用這些字尾）。"""
    return bool(code) and code[-1].upper() in ("L", "R", "B")


# ---------------------------------------------------------------------------
# Core：首頁排行用
# ---------------------------------------------------------------------------
def build_core(snapshot: pd.DataFrame, info: pd.DataFrame | None) -> pd.DataFrame:
    """
    Core = 當日成交金額前段的科技／金融／ETF ＋ 手動清單裡的主要標的。
    門檻與原本相同：股價 >= 10、當日成交金額 >= 5000 萬。
    """
    if snapshot.empty:
        return snapshot

    cat_map = _category_map(info)
    rows = []
    for _, r in snapshot.iterrows():
        code = r["code"]
        kind, sector = classify(code, cat_map.get(code, ("", ""))[0])
        if kind is None:
            continue
        if kind == "etf" and not C.ETF_ALLOW_LEVERAGED and _is_leveraged_etf(code):
            continue
        rows.append({**r.to_dict(), "kind": kind, "sector": sector,
                     "market": cat_map.get(code, ("", "TWSE"))[1]})
    if not rows:
        return snapshot.iloc[0:0]

    df = pd.DataFrame(rows)
    liquid = df[(df["close"] >= C.CORE_MIN_PRICE) & (df["turnover"] >= C.CORE_MIN_TURNOVER)]

    if C.MAX_CLOSE_PRICE and C.MAX_CLOSE_PRICE > 0:
        liquid = liquid[liquid["close"] <= C.MAX_CLOSE_PRICE]

    top = liquid.sort_values("turnover", ascending=False).head(C.TOP_N_BY_TURNOVER)
    watch = set(C.CORE_WEIGHTED_STOCKS) | set(C.TECH_STOCKS) | set(C.ETF_STOCKS) | set(C.FINANCE_STOCKS)
    listed = liquid[liquid["code"].isin(watch)]

    core = pd.concat([top, listed]).drop_duplicates(subset="code")
    core = core.sort_values("turnover", ascending=False).head(C.CORE_MAX_UNIVERSE)

    n = {k: int((core["kind"] == k).sum()) for k in ("tech", "finance", "etf")}
    log.info("Core 股票池：%d 檔（科技 %d、金融 %d、ETF %d）",
             len(core), n["tech"], n["finance"], n["etf"])
    return core.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Extended：雷達與搜尋用
# ---------------------------------------------------------------------------
def build_extended(snapshot: pd.DataFrame, info: pd.DataFrame | None,
                   core_codes: set[str]) -> list[dict]:
    """
    候選清單（還沒套流動性門檻——那要有歷史資料才算得出來，
    由 radar.py 拿到 history 之後再過濾）。

    來源：
      1. FinMind 基本資料裡所有科技／金融／ETF（含上櫃）
      2. 當日快照裡符合條件的（補上基本資料可能漏掉的）
      3. 手動清單（永遠保留）
    """
    out: dict[str, dict] = {}

    def add(code, name, kind, sector, market):
        code = str(code).strip()
        if not code or code in out:
            return
        if kind == "etf" and not C.ETF_ALLOW_LEVERAGED and _is_leveraged_etf(code):
            return
        out[code] = {"symbol": code, "name": name or "", "sector": sector or "",
                     "kind": kind, "market": market,
                     "is_core": code in core_codes}

    # 1) 基本資料（這是上櫃股票唯一的來源）
    if info is not None and len(info):
        for _, r in info.iterrows():
            code = str(r.get("stock_id", "")).strip()
            if not code or not code[:4].isdigit():
                continue
            kind, sector = classify(code, r.get("industry_category", ""))
            if kind is None:
                continue
            market = "ETF" if kind == "etf" else _MARKET.get(
                str(r.get("type", "")).lower(), "TWSE")
            add(code, r.get("stock_name", ""), kind, sector, market)

    # 2) 當日快照（上市，補漏）
    if snapshot is not None and len(snapshot):
        cat_map = _category_map(info)
        for _, r in snapshot.iterrows():
            code = r["code"]
            kind, sector = classify(code, cat_map.get(code, ("", ""))[0])
            if kind is None:
                continue
            add(code, r.get("name", ""), kind, sector,
                "ETF" if kind == "etf" else "TWSE")

    # 3) 手動清單永遠保留
    for code, sector in C.TECH_SECTORS.items():
        add(code, "", "tech", sector, "TWSE")
    for code, sector in C.FINANCE_STOCKS.items():
        add(code, "", "finance", sector, "TWSE")
    for code, sector in C.TW_ETFS.items():
        add(code, "", "etf", "ETF·" + sector, "ETF")

    lst = list(out.values())
    log.info("Extended 候選：%d 檔（科技 %d、金融 %d、ETF %d｜上市 %d、上櫃 %d）",
             len(lst),
             sum(1 for x in lst if x["kind"] == "tech"),
             sum(1 for x in lst if x["kind"] == "finance"),
             sum(1 for x in lst if x["kind"] == "etf"),
             sum(1 for x in lst if x["market"] == "TWSE"),
             sum(1 for x in lst if x["market"] == "TPEx"))
    return lst


def _category_map(info: pd.DataFrame | None) -> dict[str, tuple[str, str]]:
    """{代號: (產業分類, 市場別)}"""
    m: dict[str, tuple[str, str]] = {}
    if info is None or not len(info):
        return m
    for _, r in info.iterrows():
        code = str(r.get("stock_id", "")).strip()
        if code:
            m[code] = (str(r.get("industry_category", "") or ""),
                       _MARKET.get(str(r.get("type", "")).lower(), "TWSE"))
    return m


# ---------------------------------------------------------------------------
# 其他產業：不進首頁排行，只挑表現亮眼的放獨立分頁
# ---------------------------------------------------------------------------
def build_others(snapshot: pd.DataFrame, info: pd.DataFrame | None) -> list[dict]:
    """
    非科技／金融／ETF 的股票（食品、航運、生技、傳產…）。
    首頁不掃這些，但成交金額前段的仍值得看一眼，所以獨立成一頁。
    只取當日成交金額前 OTHERS_MAX 檔，避免流量暴增。
    """
    if snapshot is None or snapshot.empty:
        return []
    cat_map = _category_map(info)
    rows = []
    for _, r in snapshot.iterrows():
        code = r["code"]
        kind, sector = classify(code, cat_map.get(code, ("", ""))[0])
        if kind is not None:            # 屬於指定三類 → 不是「其他」
            continue
        if r["close"] < C.CORE_MIN_PRICE or r["turnover"] < C.OTHERS_MIN_TURNOVER:
            continue
        rows.append({"symbol": code, "name": r.get("name", ""),
                     "sector": sector or "其他", "kind": "other",
                     "market": cat_map.get(code, ("", "TWSE"))[1],
                     "turnover": float(r["turnover"]), "is_core": False})
    rows.sort(key=lambda x: -x["turnover"])
    rows = rows[: C.OTHERS_MAX]
    log.info("其他產業候選：%d 檔", len(rows))
    return rows


# ---------------------------------------------------------------------------
# 統計
# ---------------------------------------------------------------------------
def stats(core: pd.DataFrame, extended: list[dict]) -> dict:
    def cnt(items, key, val):
        return sum(1 for x in items if x.get(key) == val)
    return {
        "core_total": int(len(core)),
        "extended_total": len(extended),
        "tech": cnt(extended, "kind", "tech"),
        "finance": cnt(extended, "kind", "finance"),
        "etf": cnt(extended, "kind", "etf"),
        "twse": cnt(extended, "market", "TWSE"),
        "tpex": cnt(extended, "market", "TPEx"),
    }
