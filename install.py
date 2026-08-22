# -*- coding: utf-8 -*-
"""
install.py — 一鍵安裝

這支程式會幫你：
  1. 在你的 GitHub 建一個新的 repo
  2. 把整個專案上傳上去
  3. 打開 GitHub Pages 與 Actions 權限
  4. 立刻跑第一次，等它跑完
  5. 給你網址，並自動用瀏覽器打開

你只需要準備一個 GitHub 權杖（token），程式會一步一步告訴你怎麼拿。
只用 Python 內建功能，不需要先安裝任何套件。

用法：
    python3 install.py        （macOS / Linux）
    py install.py             （Windows）
或直接雙擊 setup.command（Mac）／ setup.bat（Windows）。
"""

from __future__ import annotations

import argparse
import base64
import getpass
import json
import os
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

API = "https://api.github.com"
ROOT = Path(__file__).resolve().parent

# 建立 token 的網址，scope 已經預先勾好，使用者只要按「Generate token」
TOKEN_URL = (
    "https://github.com/settings/tokens/new"
    "?scopes=repo,workflow&description=台股價差儀表板"
)

# 這些不上傳
SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", ".idea", ".vscode"}
SKIP_FILES = {".DS_Store", "Thumbs.db"}
SKIP_SUFFIX = {".pyc", ".pyo", ".zip"}


# ---------------------------------------------------------------------------
# 終端機輸出小工具
# ---------------------------------------------------------------------------
def _c(code: str, text: str) -> str:
    """在支援的終端機上加顏色；Windows 舊版 cmd 會自動略過。"""
    if os.name == "nt" and not os.environ.get("WT_SESSION"):
        return text
    return f"\033[{code}m{text}\033[0m"


def step(n: int, total: int, msg: str) -> None:
    print(f"\n{_c('1;33', f'[{n}/{total}]')} {_c('1', msg)}")


def ok(msg: str) -> None:
    print(f"  {_c('32', '✓')} {msg}")


def warn(msg: str) -> None:
    print(f"  {_c('33', '!')} {msg}")


def die(msg: str, hint: str = "") -> None:
    print(f"\n{_c('1;31', '✗ 安裝中止')}：{msg}")
    if hint:
        print(f"  {hint}")
    input("\n按 Enter 關閉…")
    sys.exit(1)


# ---------------------------------------------------------------------------
# GitHub API
# ---------------------------------------------------------------------------
class GitHub:
    def __init__(self, token: str):
        self.token = token

    def call(self, method: str, path: str, body: dict | None = None, ok_codes=(200, 201, 204)):
        url = path if path.startswith("http") else API + path
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        req.add_header("User-Agent", "tw-swing-dashboard-installer")
        if data:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read()
                return r.status, (json.loads(raw) if raw else {})
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                payload = json.loads(raw)
            except Exception:
                payload = {"message": raw.decode(errors="replace")[:300]}
            return e.code, payload
        except urllib.error.URLError as e:
            die("連不上 GitHub", f"請檢查網路連線。原始錯誤：{e.reason}")


# ---------------------------------------------------------------------------
# 收集要上傳的檔案
# ---------------------------------------------------------------------------
def collect_files() -> list[Path]:
    files = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if p.name in SKIP_FILES or p.suffix in SKIP_SUFFIX:
            continue
        # 本機示範跑出來的暫存結果不用上傳，第一次 Actions 會產生真的
        if rel.parts[:2] == ("data", "history"):
            continue
        if str(rel).replace("\\", "/") == "data/latest.json":
            continue
        files.append(p)
    return sorted(files)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def ensure_ready(gh: GitHub, owner: str, name: str) -> tuple[str, str | None]:
    """
    確保 repo 可以接受檔案上傳，回傳 (分支名稱, 現有 tree sha)。

    要處理兩件事：
      1. 空 repo（一次 commit 都沒有）不能用 Git Data API 上傳，
         會回 "Git Repository is empty."。這時先用 Contents API 塞一個
         檔案進去，把第一個 commit 生出來。
      2. 分支不一定叫 main。有些帳號的預設分支是 master，
         寫死 main 會導致後面觸發自動更新時失敗。
    """
    for attempt in range(12):
        code, repo = gh.call("GET", f"/repos/{owner}/{name}")
        if code != 200:
            die("讀取 repo 資訊失敗", repo.get("message", ""))
        branch = repo.get("default_branch") or "main"

        code, ref = gh.call("GET", f"/repos/{owner}/{name}/git/ref/heads/{branch}")
        if code == 200:
            # 拿現有 commit 的 tree，之後上傳時當 base_tree 用
            commit_sha = ref["object"]["sha"]
            code, commit = gh.call("GET", f"/repos/{owner}/{name}/git/commits/{commit_sha}")
            return branch, (commit.get("tree", {}).get("sha") if code == 200 else None)

        if attempt == 0:
            # repo 還是空的：先放一個檔案把分支建起來
            warn("repo 目前是空的，先建立第一個 commit…")
            gh.call("PUT", f"/repos/{owner}/{name}/contents/README.md", {
                "message": "初始化",
                "content": base64.b64encode(
                    "# 台股價差機會儀表板\n\n安裝中，稍候會被正式內容覆蓋。\n".encode()
                ).decode(),
                "branch": branch,
            })
        time.sleep(2)

    die("repo 一直處於空的狀態",
        "請到 GitHub 上打開這個 repo，隨便按一次 'Add a README file' 建立第一個檔案，再重新執行安裝程式。")
    return "main", None   # 不會走到，只是讓型別清楚


def main(args) -> None:
    TOTAL = 6
    print(_c("1;33", "\n╔══════════════════════════════════════════╗"))
    print(_c("1;33", "║   台股價差機會儀表板 · 一鍵安裝           ║"))
    print(_c("1;33", "╚══════════════════════════════════════════╝"))
    print("\n這支程式會把儀表板架在你自己的 GitHub 上，全程免費。")
    print("貼上 GitHub 權杖就會全自動跑完，大約 5 分鐘。")
    print(_c("2", "\n本工具僅供個人技術分析參考，不構成任何投資建議，投資有風險。"))

    if sys.version_info < (3, 8):
        die("Python 版本太舊", "請安裝 Python 3.8 以上：https://www.python.org/downloads/")

    files = collect_files()
    if not any(f.name == "build.py" for f in files):
        die("找不到專案檔案",
            "請確認 install.py 和 scripts/ 資料夾在同一層，並且是在解壓縮後的資料夾裡執行。")

    # --- 1) 取得權杖 ---
    step(1, TOTAL, "輸入 GitHub 權杖")

    # 三種來源，依序嘗試：命令列參數 > 環境變數 > 手動貼上
    token = (args.token or os.environ.get("GITHUB_TOKEN") or "").strip()
    if token:
        ok("已從" + ("參數" if args.token else "環境變數 GITHUB_TOKEN") + "讀取權杖")
    else:
        print(_c("2", "  需要一組有 repo 與 workflow 權限的權杖。"))
        print(_c("2", f"  還沒有的話到這裡產生（權限已預先勾好）：{TOKEN_URL}"))
        while not token:
            token = getpass.getpass("\n  貼上權杖後按 Enter（畫面不會顯示，這是正常的）：").strip()
            if not token:
                warn("沒有讀到內容，請再貼一次。")

    gh = GitHub(token)
    code, me = gh.call("GET", "/user")
    if code != 200:
        die("權杖無效", f"GitHub 回覆：{me.get('message', '')}。請重新產生一次權杖。")
    owner = me["login"]
    ok(f"哈囉，{owner}")

    # --- 2) 建立 repo ---
    step(2, TOTAL, "建立存放的 repo")
    default_name = "tw-swing-dashboard"
    name = (args.repo.strip() or
            input(f"  repo 名稱（直接按 Enter 用 {default_name}）：").strip() or
            default_name)

    code, res = gh.call("POST", "/user/repos", {
        "name": name,
        "description": "台股價差機會自動儀表板（僅供技術分析參考，非投資建議）",
        "private": False,      # GitHub Pages 免費版需要 public
        # 必須是 True：完全空的 repo 沒有任何 commit，
        # GitHub 的檔案上傳 API 會直接拒絕（Git Repository is empty）。
        "auto_init": True,
        "has_issues": False,
        "has_wiki": False,
    })
    if code == 201:
        ok(f"已建立 {owner}/{name}")
    elif code == 422:
        warn(f"{owner}/{name} 已經存在。")
        print("     如果這是上次安裝失敗留下的，直接按 y 繼續就好，程式會沿用它。")
        print("     如果是你另外在用的 repo，請按 Enter 取消，換一個名稱重跑。")
        confirm = input("  繼續使用這個 repo？(y/N)：").strip().lower()
        if confirm != "y":
            die("已取消", "換一個 repo 名稱再跑一次就好。")
    else:
        die("建立 repo 失敗", f"GitHub 回覆：{res.get('message', '')}")

    # --- 3) 上傳檔案 ---
    # 上傳前一定要先確認 repo 不是空的，而且要知道分支到底叫 main 還是 master
    branch, base_tree = ensure_ready(gh, owner, name)
    ok(f"目標分支：{branch}")

    step(3, TOTAL, f"上傳 {len(files)} 個檔案")
    tree = []
    for i, f in enumerate(files, 1):
        rel = str(f.relative_to(ROOT)).replace("\\", "/")
        content = base64.b64encode(f.read_bytes()).decode()
        code, res = gh.call("POST", f"/repos/{owner}/{name}/git/blobs",
                            {"content": content, "encoding": "base64"})
        if code != 201:
            die(f"上傳 {rel} 失敗", f"GitHub 回覆：{res.get('message', '')}")
        tree.append({"path": rel, "mode": "100644", "type": "blob", "sha": res["sha"]})
        print(f"\r  {i}/{len(files)}  {rel[:52]:<52}", end="", flush=True)
    print()

    # 帶 base_tree：只新增／覆蓋我們的檔案，保留 repo 裡既有的東西
    # （例如之前累積的 data/history/，重跑安裝不會被清掉）
    body = {"tree": tree}
    if base_tree:
        body["base_tree"] = base_tree
    code, res = gh.call("POST", f"/repos/{owner}/{name}/git/trees", body)
    if code != 201:
        die("建立檔案樹失敗", res.get("message", ""))
    tree_sha = res["sha"]

    code, ref = gh.call("GET", f"/repos/{owner}/{name}/git/ref/heads/{branch}")
    parents = [ref["object"]["sha"]] if code == 200 else []

    code, res = gh.call("POST", f"/repos/{owner}/{name}/git/commits", {
        "message": "安裝台股價差機會儀表板",
        "tree": tree_sha,
        "parents": parents,
    })
    if code != 201:
        die("建立 commit 失敗", res.get("message", ""))
    commit_sha = res["sha"]

    if parents:
        code, res = gh.call("PATCH", f"/repos/{owner}/{name}/git/refs/heads/{branch}",
                            {"sha": commit_sha, "force": True})
    else:
        code, res = gh.call("POST", f"/repos/{owner}/{name}/git/refs",
                            {"ref": f"refs/heads/{branch}", "sha": commit_sha})
    if code not in (200, 201):
        die("寫入分支失敗", res.get("message", ""))
    ok("檔案已上傳")

    # --- 4) 設定權限與 Pages ---
    step(4, TOTAL, "打開自動更新與網站發布")
    code, res = gh.call("PUT", f"/repos/{owner}/{name}/actions/permissions/workflow", {
        "default_workflow_permissions": "write",
        "can_approve_pull_request_reviews": False,
    })
    ok("已允許自動更新寫回結果") if code in (200, 204) else warn(
        "自動寫回權限設定失敗，請手動到 Settings → Actions → General 選 Read and write permissions")

    code, res = gh.call("POST", f"/repos/{owner}/{name}/pages", {"build_type": "workflow"})
    if code in (201, 204):
        ok("已開啟 GitHub Pages")
    elif code == 409:
        gh.call("PUT", f"/repos/{owner}/{name}/pages", {"build_type": "workflow"})
        ok("GitHub Pages 已是開啟狀態")
    else:
        warn("Pages 自動開啟失敗，請手動到 Settings → Pages，Source 選 GitHub Actions")

    # --- 5) 跑第一次 ---
    step(5, TOTAL, "抓取今天的資料並產生儀表板")
    code = None
    for attempt in range(6):
        time.sleep(4)   # 等 GitHub 認得剛推上去的 workflow 檔
        code, res = gh.call(
            "POST", f"/repos/{owner}/{name}/actions/workflows/daily.yml/dispatches",
            {"ref": branch})
        if code == 204:
            break
        if attempt == 0:
            print("  等 GitHub 認得剛上傳的設定檔…")
    if code != 204:
        warn("自動觸發失敗，請到 repo 的 Actions 分頁手動按 Run workflow。")
    else:
        print("  執行中，通常需要 3～6 分鐘。可以先去泡杯咖啡。")
        status = wait_for_run(gh, owner, name)
        if status == "success":
            ok("儀表板已產生")
        elif status == "failure":
            warn("這次執行失敗了。到 Actions 分頁點進去看 log，多半是資料來源暫時異常，"
                 "等下一次排程（16:20 / 18:00）通常就正常了。")
        else:
            warn("等太久了，先幫你收尾。稍後到 Actions 分頁看結果即可。")

    # --- 6) 完成 ---
    site = f"https://{owner}.github.io/{name}/"
    step(6, TOTAL, "完成")
    print(f"\n  你的儀表板網址：\n  {_c('1;36', site)}\n")
    print("  " + _c("1", "手機要怎麼變成 app："))
    print("   iPhone：用 Safari 打開上面網址 → 按下方「分享」→「加入主畫面」")
    print("   Android：用 Chrome 打開 → 網址列旁的選單 →「安裝應用程式」")
    print("\n  之後每個交易日 16:20 和 18:00 會自動更新，你什麼都不用做。")
    print(_c("2", "\n  提醒：本工具僅供個人技術分析參考，不構成任何投資建議，投資有風險。"))

    (ROOT / "我的儀表板網址.txt").write_text(
        f"{site}\n\nrepo：https://github.com/{owner}/{name}\n", encoding="utf-8")
    ok("網址也存成「我的儀表板網址.txt」放在同一個資料夾")

    try:
        webbrowser.open(site)
    except Exception:
        pass
    input("\n按 Enter 關閉…")


def wait_for_run(gh: GitHub, owner: str, name: str, timeout: int = 600) -> str:
    """等 workflow 跑完，回傳 success / failure / timeout。"""
    deadline = time.time() + timeout
    spin = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    i = 0
    while time.time() < deadline:
        code, res = gh.call("GET", f"/repos/{owner}/{name}/actions/runs?per_page=1")
        runs = res.get("workflow_runs", []) if code == 200 else []
        if runs:
            run = runs[0]
            if run["status"] == "completed":
                print()
                return run.get("conclusion") or "failure"
            print(f"\r  {spin[i % len(spin)]} {run['status']}…   ", end="", flush=True)
        i += 1
        time.sleep(5)
    print()
    return "timeout"


def parse_args():
    ap = argparse.ArgumentParser(description="台股價差機會儀表板 一鍵安裝")
    ap.add_argument("--token", default="", help="GitHub 權杖，直接帶入就不會再問")
    ap.add_argument("--repo", default="", help="repo 名稱，預設 tw-swing-dashboard")
    return ap.parse_args()


if __name__ == "__main__":
    try:
        main(parse_args())
    except KeyboardInterrupt:
        print("\n\n已取消。")
