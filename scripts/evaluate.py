"""Read-only validation entry point; identical strategy and rank to backtest.py."""
import argparse
import backtest


def main():
    ap = argparse.ArgumentParser(description="淨利達標排名的逐段樣本外驗證（不寫入資料）")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--days", type=int, default=250)
    args = ap.parse_args()
    result = backtest.run_backtest(backtest.load_universe_history(args.demo), args.days,
                                   backtest.load_regimes(args.demo))
    backtest.print_report(result, args.demo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
