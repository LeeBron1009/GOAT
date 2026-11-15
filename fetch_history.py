# fetch_history.py - 下載台股真實歷史資料，產生 history.json（750 天版，含保護機制）
# Colab：先執行  !pip install -q yfinance  再貼上整份程式
import json, time, sys
import yfinance as yf

CODES = ["2330","2317","2454","2412","2882","2881","2308","2303","1301","2002",
         "2603","2609","2615","3711","2382","3231","2357","2379","2891","2886",
         "1216","2912","0050","0056","00878","00919"]
DAYS = 750
PERIOD = "5y"

def get(code):
    for suf in (".TW", ".TWO"):
        try:
            df = yf.Ticker(code + suf).history(period=PERIOD, interval="1d", auto_adjust=False)
            if not df.empty:
                return df
        except Exception as e:
            print("  錯誤", code + suf, e)
    return None

out = {}
for c in CODES:
    df = get(c)
    if df is None:
        print("失敗", c); continue
    df = df.dropna().tail(DAYS)
    if len(df) < 70:
        print("略過", c, "資料太少"); continue
    out[c] = {
        "d": [d.strftime("%Y-%m-%d") for d in df.index],
        "o": [round(float(x), 2) for x in df["Open"]],
        "h": [round(float(x), 2) for x in df["High"]],
        "l": [round(float(x), 2) for x in df["Low"]],
        "c": [round(float(x), 2) for x in df["Close"]],
        "v": [int(x) for x in df["Volume"]],
    }
    print("OK", c, len(df), "筆，最新", out[c]["d"][-1], out[c]["c"][-1])
    time.sleep(0.5)

# 保護：抓到的股票太少就不要覆蓋舊檔
if len(out) < 10:
    print("資料太少（%d 檔），不覆蓋舊的 history.json" % len(out))
    sys.exit(1)

with open("history.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
print("完成，共", len(out), "檔")

try:
    from google.colab import files
    files.download("history.json")
except ImportError:
    pass
