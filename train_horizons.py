# train_horizons.py - 比較「隔天 / 未來5天 / 未來10天」哪個比較有預測訊號
# Colab：先執行  !pip install -q yfinance scikit-learn scipy   再貼上整份程式
# 輸出：horizon_results.json（放到網站資料夾）
import json, time, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from scipy.stats import binomtest
warnings.filterwarnings("ignore")

CODES = ["2330","2317","2454","2412","2882","2881","2308","2303","1301","2002",
         "2603","2609","2615","3711","2382","3231","2357","2379","2891","2886",
         "1216","2912","0050","0056","00878","00919"]
PERIOD = "5y"
HORIZONS = [1, 5, 10]     # 預測幾天後
FOLDS = 5                 # 走動式驗證的折數
THR = 0.55                # 「高信心」門檻
COST = 0.004275           # 一買一賣的大約成本（股票）

FEATURES = ["ret1","ret5","ret20","ma5_gap","ma20_gap","ma60_gap","trend","rsi","macd_h",
            "vol_ratio","vol20","range","rule"]

def load_data():
    import yfinance as yf
    out = {}
    for c in CODES:
        df = None
        for suf in (".TW", ".TWO"):
            try:
                d = yf.Ticker(c + suf).history(period=PERIOD, interval="1d", auto_adjust=True)
                if not d.empty: df = d; break
            except Exception as e:
                print("錯誤", c, e)
        if df is None or len(df) < 300:
            print("略過", c); continue
        df = df[["Open","High","Low","Close","Volume"]].dropna()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        out[c] = df; print("OK", c, len(df)); time.sleep(0.4)
    return out

def rsi(close, n=14):
    d = close.diff()
    g = d.clip(lower=0).rolling(n).mean(); l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def make_features(df):
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    f = pd.DataFrame(index=df.index)
    f["ret1"] = c.pct_change(); f["ret5"] = c.pct_change(5); f["ret20"] = c.pct_change(20)
    m5, m20, m60 = c.rolling(5).mean(), c.rolling(20).mean(), c.rolling(60).mean()
    f["ma5_gap"] = c / m5 - 1; f["ma20_gap"] = c / m20 - 1; f["ma60_gap"] = c / m60 - 1
    f["trend"] = m20 / m60 - 1
    f["rsi"] = rsi(c)
    e12, e26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    dif = e12 - e26; dea = dif.ewm(span=9, adjust=False).mean(); hist = dif - dea
    f["macd_h"] = hist / c
    f["vol_ratio"] = v / v.rolling(5).mean()
    f["vol20"] = f["ret1"].rolling(20).std()
    f["range"] = (h - l) / c
    sc = 50 + np.where(c > m20, 10, -10) + np.where(m5 > m20, 10, -10) + np.where(m20 > m60, 8, -8)
    sc = sc + np.where((hist > 0) & (hist > hist.shift(1)), 10, -10)
    sc = sc + np.where(f["rsi"] > 70, -10, np.where(f["rsi"] < 30, 10, 0))
    f["rule"] = np.clip(sc, 0, 100)
    for H in HORIZONS:
        f[f"fwd{H}"] = c.shift(-H) / c - 1
    return f.replace([np.inf, -np.inf], np.nan)

def new_models():
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=500)),
        "gbm": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150,
                                              l2_regularization=5.0, min_samples_leaf=100, random_state=0),
    }

def walk_forward(allf, H):
    """擴張視窗走動式驗證：每一折只用『過去』訓練，並在訓練與測試間隔 H 天（purge）避免標籤重疊洩漏。"""
    d = allf.dropna(subset=FEATURES + [f"fwd{H}"]).copy()
    d["y"] = (d[f"fwd{H}"] > 0).astype(int)
    dates = np.sort(d["date"].unique())
    cut = np.linspace(int(len(dates) * 0.4), len(dates), FOLDS + 1).astype(int)
    res = {m: dict(folds=[], preds=[]) for m in new_models()}
    rows = []
    for k in range(FOLDS):
        tr_end = dates[cut[k] - H - 1] if cut[k] - H - 1 >= 0 else dates[0]
        te_a, te_b = dates[cut[k]], dates[min(cut[k + 1], len(dates)) - 1]
        tr = d[d["date"] <= tr_end]; te = d[(d["date"] >= te_a) & (d["date"] <= te_b)]
        if len(tr) < 500 or len(te) < 100 or te["y"].nunique() < 2: continue
        for name, m in new_models().items():
            m.fit(tr[FEATURES], tr["y"]); p = m.predict_proba(te[FEATURES])[:, 1]
            res[name]["folds"].append(dict(start=str(te_a)[:10], end=str(te_b)[:10], n=int(len(te)),
                                           auc=float(roc_auc_score(te["y"], p)),
                                           acc=float(((p >= .5) == te["y"].values).mean()),
                                           base=float(te["y"].mean())))
            t = te[["code", "date", "y", f"fwd{H}"]].copy(); t["p"] = p; t["model"] = name; rows.append(t)
    return res, pd.concat(rows) if rows else None

def summarize(H, res, allp):
    out = {}
    for name in res:
        fl = res[name]["folds"]
        if not fl: continue
        sub = allp[allp["model"] == name]
        non = sub.sort_values("date").groupby("code").apply(lambda g: g.iloc[::H]).reset_index(drop=True)
        base = float(non["y"].mean()); k = int(((non["p"] >= .5).astype(int) == non["y"]).sum())
        pv = float(binomtest(k, len(non), max(base, 1 - base), alternative="greater").pvalue)
        hi = non[non["p"] >= THR]; allret = non[f"fwd{H}"].mean()
        out[name] = dict(
            auc=float(np.mean([f["auc"] for f in fl])), auc_std=float(np.std([f["auc"] for f in fl])),
            acc=float(np.mean([f["acc"] for f in fl])), base=float(np.mean([f["base"] for f in fl])),
            folds_pos=int(sum(f["auc"] > .5 for f in fl)), folds=len(fl),
            p_value=pv, n_nonoverlap=int(len(non)),
            hi_n=int(len(hi)), hi_ret=float(hi[f"fwd{H}"].mean()) if len(hi) else None,
            all_ret=float(allret), cost=COST, fold_aucs=[round(f["auc"], 3) for f in fl])
    return out

def main():
    data = load_data()
    if len(data) < 10:
        raise SystemExit("資料太少，不訓練、不覆蓋舊結果")
    feats = {c: make_features(df) for c, df in data.items()}
    allf = pd.concat(feats, names=["code", "date"]).reset_index()
    res_all = {"meta": dict(stocks=len(data), folds=FOLDS, thr=THR, period=PERIOD), "horizons": {}, "latest": {}}
    for H in HORIZONS:
        r, allp = walk_forward(allf, H)
        if allp is None: print("資料不足", H); continue
        s = summarize(H, r, allp); res_all["horizons"][str(H)] = s
        print(f"\n===== 預測 {H} 天後漲跌 =====")
        for m, v in s.items():
            print(f"[{m}] AUC {v['auc']:.3f}±{v['auc_std']:.3f}（{v['folds_pos']}/{v['folds']} 折 >0.5） "
                  f"準確率 {v['acc']:.1%}（基準 {v['base']:.1%}） p={v['p_value']:.3f}")
            if v["hi_ret"] is not None:
                print(f"     高信心(≥{THR}) {v['hi_n']} 次：平均 {H} 天報酬 {v['hi_ret']:.2%}，全部平均 {v['all_ret']:.2%}，成本約 {COST:.2%}")
    lines = []
    for H, s in res_all["horizons"].items():
        if not s: continue
        m = max(s, key=lambda k: s[k]["auc"]); v = s[m]
        ok = v["p_value"] < 0.05 and v["folds_pos"] >= v["folds"] - 1 and v["auc"] > .52
        lines.append(f"預測 {H} 天：最好的是 {m}，AUC {v['auc']:.3f}，" + ("有一定訊號，但仍需扣成本檢驗。" if ok else "訊號不明顯，可能只是運氣。"))
    best = max(((H, max(s.values(), key=lambda x: x["auc"])["auc"]) for H, s in res_all["horizons"].items()), key=lambda t: t[1])
    res_all["verdict"] = " ".join(lines) + f" 三者中 AUC 最高的是『{best[0]} 天』（{best[1]:.3f}）。AUC 0.5＝隨機，0.55 以上才稍有意義。"
    print("\n" + res_all["verdict"])
    for H in HORIZONS:
        d = allf.dropna(subset=FEATURES + [f"fwd{H}"]); y = (d[f"fwd{H}"] > 0).astype(int)
        ms = {n: m.fit(d[FEATURES], y) for n, m in new_models().items()}
        for c, f in feats.items():
            row = f.dropna(subset=FEATURES).iloc[[-1]]
            e = res_all["latest"].setdefault(c, {"date": str(row.index[0])[:10]})
            e[str(H)] = {n: round(float(m.predict_proba(row[FEATURES])[0, 1]), 4) for n, m in ms.items()}
    with open("horizon_results.json", "w", encoding="utf-8") as fp:
        json.dump(res_all, fp, ensure_ascii=False, separators=(",", ":"))
    print("已輸出 horizon_results.json")
    try:
        from google.colab import files; files.download("horizon_results.json")
    except ImportError:
        pass

if __name__ == "__main__":
    main()
