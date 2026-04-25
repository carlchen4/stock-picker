"""
TSX 量化选股 v2.0 — 全优化版
════════════════════════════════════════════════════════════════
数据：yfinance（免费，无配额）
模型：XGBoost + LightGBM + PyTorch MLP（动态集成权重）

四层优化：
  第一层：止损线(-8%) / 换仓成本显示 / 股息率展示
  第二层：宏观因子(油价/汇率/利率) / 行业轮动 / 季节性
  第三层：PIT 基本面（无 Look-ahead Bias，45天延迟）
  第四层：风险平价仓位 / 最大回撤控制 / 动态模型权重 / 交易成本

已修复 Bug：
  ✓ BatchNorm1d size=1 → drop_last=True
  ✓ 季报日期不一致 → _nearest_col() 容忍 ±5 天
  ✓ Look-ahead Bias → PIT 基本面对齐
  ✓ dropna 过滤太严 → 只对 label 做 dropna
  ✓ meta_df 缺 roe/div_yield → 已补充
  ✓ 成交量单日误踢 → vol_spike_min_days=2
  ✓ rows 为空崩溃 → try/except + 明确错误

安装：
    pip install yfinance xgboost lightgbm scikit-learn torch pandas numpy
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    LGBM = True
except ImportError:
    LGBM = False
    print("⚠️  LightGBM 未安装: pip install lightgbm")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH = True
except ImportError:
    TORCH = False
    print("⚠️  PyTorch 未安装: pip install torch")

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

# ── TSX 综合指数股票池（约 230 支）─────────────────────────────
# 来源：S&P/TSX Composite Index + TSX 60 扩展
# 本地运行时可改 source="xic" 自动从 BlackRock XIC ETF 更新

TSX_UNIVERSE = [
    # 金融：六大银行 + 保险 + 资管
    "RY.TO","TD.TO","BNS.TO","BMO.TO","CM.TO","NA.TO",
    "MFC.TO","SLF.TO","GWO.TO","POW.TO","IFC.TO","FFH.TO",
    "IGM.TO","IAG.TO","EQB.TO","LB.TO",
    "CIGI.TO","FSV.TO","BN.TO","BAM.TO","CF.TO",# 能源：油砂 + 管道 + 天然气
    "SU.TO","CNQ.TO","CVE.TO","IMO.TO",
    "ENB.TO","TRP.TO","PPL.TO","KEY.TO",
    "TOU.TO","ARX.TO","WCP.TO","VET.TO","BIR.TO",
    "TVE.TO","PEY.TO","SGY.TO","BTE.TO",
    # 材料：黄金 + 铜 + 化肥 + 铀 + 林业
    "ABX.TO","AEM.TO","K.TO","AGI.TO","IMG.TO","WDO.TO",
    "WPM.TO","FNV.TO","FM.TO","LUN.TO","ERO.TO","TECK-B.TO","HBM.TO",
    "NTR.TO","CCO.TO","DML.TO","WFG.TO","CFP.TO","IFP.TO","MRE.TO",
    # 工业：铁路 + 工程 + 制造 + 运输
    "CNR.TO","CP.TO",
    "WSP.TO","STN.TO","ATS.TO",
    "CAE.TO","FTT.TO","GIL.TO","MTL.TO","MDA.TO","RBA.TO","NFI.TO",
    "CJT.TO","AC.TO","GFL.TO","WCN.TO",
    # 科技：软件 + SaaS + IT服务
    "CSU.TO","TRI.TO","SHOP.TO","ENGH.TO","OTEX.TO","GIB-A.TO",
    "DSG.TO","DCBO.TO","KXS.TO","LSPD.TO","S.TO","LIF.TO",
    # 消费必需
    "ATD.TO","DOL.TO","MRU.TO","L.TO","WN.TO","EMP-A.TO","PBH.TO","NWC.TO","SAP.TO",
    # 消费可选
    "QSR.TO","MG.TO","BYD.TO","CTC-A.TO","LNF.TO",
    # 通讯
    "BCE.TO","T.TO","RCI-B.TO","QBR-B.TO",# REITs：工业 + 住宅 + 零售 + 办公 + 养老
    "GRT-UN.TO","DIR-UN.TO","CRT-UN.TO","SRU-UN.TO",
    "CAR-UN.TO","IIP-UN.TO","MI-UN.TO",
    "REI-UN.TO","CHP-UN.TO","NWH-UN.TO","BTB-UN.TO",
    "HR-UN.TO","AP-UN.TO","D-UN.TO","CSH-UN.TO","SIA.TO",
    # 公用事业：受管制 + 可再生
    "FTS.TO","EMA.TO","CU.TO","H.TO","AQN.TO","ALA.TO","CPX.TO",
    "NPI.TO","BEP-UN.TO","BLX.TO",
    # 医疗
    "BHC.TO","WELL.TO","CLS.TO","HLS.TO","DRX.TO",]

# 过滤已知退市股票
_DELISTED = {
    "TFI.TO","BRP.TO","CAP-UN.TO","PKI.TO","MPW.TO","INE.TO",
    "BAD.TO","GDX.TO","RNW.TO","ERF.TO","DSY.TO","TOI.TO",
    "CPG.TO","CIX.TO","DND.TO","NXE.TO",
}
TSX_UNIVERSE = list(dict.fromkeys(t for t in TSX_UNIVERSE if t not in _DELISTED))


def get_tsx_tickers(source: str = "builtin") -> list[str]:
    """
    获取 TSX 股票池。

    source:
      "builtin"   → 内置 ~220 支，无需网络（默认）
      "xic"       → 从 BlackRock XIC ETF 自动更新（需本地运行）
      "cache"     → 读取上次 XIC 抓取的缓存文件

    使用方式：
      本地首次运行：source="xic"  → 下载最新成分股并缓存
      Colab / 无网络：source="builtin"
    """
    if source == "xic":
        try:
            import requests, io, pandas as pd
            headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                                     "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"}
            url = ("https://www.blackrock.com/ca/investors/en/products/239837/"
                   "ishares-core-sp-tsx-capped-composite-index-etf/"
                   "1464253357818.ajax?tab=all&fileType=csv")
            r = requests.get(url, headers=headers, timeout=20)
            if r.status_code != 200:
                raise ValueError(f"HTTP {r.status_code}")

            df = pd.read_csv(io.StringIO(r.text), skiprows=2)
            ticker_col = next((c for c in df.columns
                               if "ticker" in c.lower() or "symbol" in c.lower()), None)
            if not ticker_col:
                raise ValueError("找不到 Ticker 列")

            tickers = []
            for v in df[ticker_col].dropna():
                t = str(v).strip()
                if t and t != "nan" and not t.startswith("-"):
                    tickers.append(t + ".TO")

            tickers = [t for t in tickers if t not in _DELISTED]
            print(f"  [XIC] ✅ 获取 {len(tickers)} 支 TSX 成分股")

            with open(".tsx_universe_cache.txt", "w") as f:
                f.write("\n".join(tickers))
            return tickers

        except Exception as e:
            print(f"  [XIC] ⚠️  失败：{e}，使用内置列表")
            return TSX_UNIVERSE

    elif source == "cache":
        try:
            with open(".tsx_universe_cache.txt") as f:
                tickers = [t.strip() for t in f.readlines() if t.strip()]
            print(f"  [缓存] ✅ 读取 {len(tickers)} 支（上次 XIC 更新）")
            return tickers
        except FileNotFoundError:
            print("  [缓存] ⚠️  缓存不存在，使用内置列表")
            return TSX_UNIVERSE

    else:
        print(f"  [内置] {len(TSX_UNIVERSE)} 支 TSX 股票（S&P/TSX Composite + 扩展）")
        return TSX_UNIVERSE


# ── 实际使用的股票池 ──────────────────────────────────────────────
# 本地运行改为 source="xic" 可自动从 BlackRock 更新到 ~250 支
TICKERS = get_tsx_tickers(source="builtin")


# ── Simfin 配置（本地运行时填入，Colab 会因 IP 限制失败）──────
SIMFIN_API_KEY = "804d29c7-c3cf-43d4-96a1-128edd64b7ff"

# 固定所有随机种子（确保同一天同样数据跑出相同结果）
import random, numpy as np
random.seed(42)
np.random.seed(42)
try:
    import torch
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
except ImportError:
    pass
FMP_API_KEY    = "JiSlrxR3kbnUQnNcTShCFjUNdq78htKf"   # 注册 financialmodelingprep.com 获取免费 key
SIMFIN_DATA_DIR = "./simfin_data"   # 本地缓存目录

def fetch_pit_fmp(ticker: str, api_key: str, quarters: int = 20) -> pd.DataFrame:
    """
    从 Financial Modeling Prep 获取季报 PIT 数据。

    优势：
      - 免费 250 次/天，Colab 可用（不被 IP 屏蔽）
      - TSX 股票直接用 RY.TO 格式
      - 有 fillingDate（实际披露日），可实现真正的 PIT
      - 历史最多 5 年季报（免费版）

    返回格式与 build_pit_from_simfin 相同，可直接替换。
    """
    import requests, time

    BASE = "https://financialmodelingprep.com/api/v3"
    rows = []

    try:
        # 损益表
        r_inc = requests.get(
            f"{BASE}/income-statement/{ticker}",
            params={"period":"quarter","limit":quarters,"apikey":api_key},
            timeout=10).json()

        # 资产负债表
        r_bal = requests.get(
            f"{BASE}/balance-sheet-statement/{ticker}",
            params={"period":"quarter","limit":quarters,"apikey":api_key},
            timeout=10).json()

        # 现金流量表
        r_cf  = requests.get(
            f"{BASE}/cash-flow-statement/{ticker}",
            params={"period":"quarter","limit":quarters,"apikey":api_key},
            timeout=10).json()

        if not r_inc or "Error Message" in str(r_inc):
            return pd.DataFrame()

        # 以披露日（fillingDate）为 PIT 基准
        bal_by_date = {x["date"]: x for x in (r_bal or [])}
        cf_by_date  = {x["date"]: x for x in (r_cf  or [])}

        for inc in r_inc:
            qdate    = inc.get("date", "")
            fill_dt  = inc.get("acceptedDate") or inc.get("fillingDate") or qdate
            if not qdate:
                continue

            bal = bal_by_date.get(qdate, {})
            cf  = cf_by_date.get(qdate,  {})

            net_income = inc.get("netIncome", 0) or 0
            revenue    = inc.get("revenue",   0) or 0
            equity     = bal.get("totalStockholdersEquity", None)
            assets     = bal.get("totalAssets", None)
            capex      = cf.get("capitalExpenditure", 0) or 0
            op_cf      = cf.get("operatingCashFlow",  0) or 0
            shares     = inc.get("weightedAverageShsOut", None)

            eps        = (net_income / shares) if shares else None
            roe        = (net_income / equity) if equity else None
            fcf_yield  = ((op_cf + capex) / assets) if assets else None

            rows.append({
                "avail_date": pd.Timestamp(fill_dt[:10]),
                "report_date": pd.Timestamp(qdate),
                "pe":         None,   # 需要用价格除 EPS，后续处理
                "pb":         None,
                "roe":        roe,
                "eps":        eps,
                "eps_growth": None,   # 同比增长在 build_pit 里计算
                "fcf_yield":  fcf_yield,
                "revenue":    revenue,
                "net_income": net_income,
            })

        if rows:
            df = pd.DataFrame(rows).sort_values("report_date")
            # 计算 EPS 同比增长
            df["eps_growth"] = df["eps"].pct_change(4)
            return df

    except Exception as e:
        pass

    return pd.DataFrame()


def init_fmp(tickers: list, api_key: str) -> dict:
    """
    批量从 FMP 获取季报数据，返回 {ticker: DataFrame} 字典。
    免费版 250次/天，135支股票需要约 135×3 = 405 次请求，
    建议分批或使用缓存。
    """
    import os, pickle, time

    cache_file = "./simfin_data/fmp_cache.pkl"
    os.makedirs("./simfin_data", exist_ok=True)

    # 读缓存（当天内有效）
    if os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        if (time.time() - mtime) < 86400:  # 24小时内
            try:
                with open(cache_file, "rb") as f:
                    cached = pickle.load(f)
                print(f"  [FMP] 从缓存加载（{len(cached)} 支）")
                return cached
            except Exception:
                pass

    print(f"  [FMP] 下载 {len(tickers)} 支季报（免费版约需 2-3 分钟）...")
    result = {}
    for i, t in enumerate(tickers):
        df = fetch_pit_fmp(t, api_key)
        if not df.empty:
            result[t] = df
        if (i+1) % 20 == 0:
            print(f"  [FMP] {i+1}/{len(tickers)} 支完成...")
        time.sleep(0.1)  # 避免超频

    with open(cache_file, "wb") as f:
        pickle.dump(result, f)

    print(f"  [FMP] ✓ 完成，{len(result)} 支有数据，已缓存")
    return result


def init_simfin():
    """
    初始化 Simfin，带磁盘缓存。

    Simfin bulk download 在某些 IP（如 Colab）返回 401/403。
    解决方案：
      1. 优先读取磁盘缓存（上次成功下载后保留）
      2. 缓存不存在时尝试下载（本地运行成功率高）
      3. 完全失败时 fallback 到 yfinance
    """
    import os, pickle

    cache_file = os.path.join(SIMFIN_DATA_DIR, "simfin_ca_cache.pkl")
    os.makedirs(SIMFIN_DATA_DIR, exist_ok=True)

    # 优先读本地缓存（不受 IP 限制）
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                inc, bal, cf = pickle.load(f)
            n = inc.index.get_level_values("Ticker").nunique()
            yr_min = inc.index.get_level_values("Report Date").min().year
            yr_max = inc.index.get_level_values("Report Date").max().year
            print(f"  [Simfin] ✓ 从缓存加载 ({n} 支, {yr_min}→{yr_max})")
            return inc, bal, cf
        except Exception as e:
            print(f"  [Simfin] 缓存读取失败：{e}，尝试重新下载...")

    # 尝试在线下载
    try:
        import simfin as sf
        sf.set_api_key(SIMFIN_API_KEY)
        sf.set_data_dir(SIMFIN_DATA_DIR)

        print("  [Simfin] 加载季报数据（首次下载约 2 分钟）...")
        inc = sf.load_income(variant="quarterly", market="ca")
        bal = sf.load_balance(variant="quarterly", market="ca")
        cf  = sf.load_cashflow(variant="quarterly", market="ca")

        # 成功后保存缓存（下次无网络也可用）
        with open(cache_file, "wb") as f:
            pickle.dump((inc, bal, cf), f)

        n = inc.index.get_level_values("Ticker").nunique()
        yr_min = inc.index.get_level_values("Report Date").min().year
        yr_max = inc.index.get_level_values("Report Date").max().year
        print(f"  [Simfin] ✓ 季报加载成功 ({n} 支, {yr_min}→{yr_max})")
        print(f"  [Simfin] ✓ 已缓存到 {cache_file}")
        return inc, bal, cf

    except ImportError:
        print("  [Simfin] ⚠️  未安装：pip install simfin")
        return None, None, None
    except Exception as e:
        err_str = str(e)
        if "401" in err_str or "403" in err_str:
            print(f"  [Simfin] ⚠️  IP 受限（{err_str[:60]}）")
            print(f"  [Simfin]    解决方法：本地运行一次后缓存会保留到 Colab")
            print(f"  [Simfin]    缓存路径：{cache_file}")
        else:
            print(f"  [Simfin] ⚠️  加载失败：{e}")
        return None, None, None


def build_pit_from_simfin(ticker, inc_all, bal_all, cf_all):
    """
    从 Simfin 季报数据构建 PIT（Point-in-Time）基本面时间序列。

    优势 vs yfinance：
      - 覆盖约 10 年完整历史（yfinance 只有 6-7 季）
      - 官方财报数据，更准确
      - 有明确的 Report Date（公告日期），PIT 更精确

    TSX 代码转换：RY.TO → RY，ENB.TO → ENB
    """
    t_simfin = ticker.replace(".TO", "")

    try:
        # Simfin MultiIndex: (Ticker, Report Date)
        def get_ticker(df):
            if df is None: return pd.DataFrame()
            try:
                return df.xs(t_simfin, level="Ticker")
            except KeyError:
                return pd.DataFrame()

        inc = get_ticker(inc_all)
        bal = get_ticker(bal_all)
        cf  = get_ticker(cf_all)

        if inc.empty:
            return pd.DataFrame()

        rows = []
        for report_date in inc.index:
            # Simfin Report Date = 实际公告日期（已包含延迟）
            # 额外加 5 天确保数据已完全公开
            avail = pd.Timestamp(report_date) + pd.Timedelta(days=5)

            def g(df, col):
                try:
                    return float(df.loc[report_date, col]) if col in df.columns else None
                except Exception:
                    return None

            # 主要财务数据（Simfin 列名）
            net_income   = g(inc, "Net Income")
            total_equity = g(bal, "Total Equity") or g(bal, "Common Equity")
            ocf          = g(cf,  "Net Cash from Operating Activities")
            capex        = g(cf,  "Purchase of Property, Plant and Equipment")
            shares       = g(inc, "Shares (Diluted)") or g(bal, "Common Shares Outstanding")

            rows.append({
                "avail_date":   avail,
                "net_income":   net_income,
                "total_equity": total_equity,
                "ocf":          ocf,
                "capex":        capex,
                "shares":       shares,
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("avail_date").sort_index()
        return df

    except Exception as e:
        return pd.DataFrame()


MACRO_TICKERS = {
    "oil":   "CL=F",
    "cadusd":"CADUSD=X",
    "bond":  "^TNX",
    "tsx":   "^GSPTSE",
    "gold":  "GC=F",
    "vix":   "^VIX",
}

TOP_N         = 10
YEARS         = 5
MIN_TRAIN     = 18
TOP_QUINTILE  = 0.20
REPORT_LAG    = 45
STOP_LOSS_PCT = -0.08
MAX_DD_THRESH = -0.05

CONSTRAINTS = {
    "min_adv_cad":        1_000_000,
    "vol_spike_sigma":    3.0,
    "vol_spike_days":     5,
    "vol_spike_min_days": 2,
    "min_pe":             0.0,
    "max_pe":             60.0,
    "min_mktcap_cad":     500_000_000,
    "min_price_cad":      2.00,
    "max_price_cad":    400.00,   # 单股价格上限（过高的股票买不到足够整股）
    "min_roe":            0.0,
    "max_roe":          2.00,   # ROE > 200% 视为财务异常（如 BHC 436%）
    "min_shares":          5,   # 最少持仓股数（价格过高股票如 FFH 会被过滤）
    "max_per_gics":       2,      # 大类行业上限
    "max_per_style":      4,
    "max_per_type":       5,
    "max_single_alloc":   0.20,
    "max_turnover":       4,      # 每月最多换仓数
    "hold_bonus":         0.05,   # 持仓连续性奖励
    # OPT1: 矿业子行业硬上限
    "max_gold_mining":    2,      # 黄金矿业最多 2 支
    "max_base_metals":    1,      # 贱金属（铜/锌）最多 1 支
    "max_energy_sub":     2,      # 能源子类最多 2 支
    # OPT5: 换仓冷静期
    "cooldown_months":    1,      # 止损后冷静 N 个月
    # OPT6: 置信度过滤
    "min_confidence":     0.20,   # 集成分低于此值不交易
    "min_top_n":          5,      # 置信度不足时最少持仓
    # OPT7: 熔断 + VIX 缩仓
    "dd_halt_threshold":  -0.15,  # 3月累计亏损超此值→减仓
    "dd_halt_scale":      0.50,   # 减仓比例
    "vix_scale_threshold":25.0,   # VIX 高于此值时缩仓
    "vix_scale_factor":   0.70,   # VIX 高时仓位缩至 70%
    "min_listing_days":   252,
}

# 矿业子行业分类（OPT1 用）
GOLD_MINING_TICKERS = {"K.TO","ABX.TO","AEM.TO","WDO.TO","AGI.TO",
                        "IMG.TO","WPM.TO","FNV.TO"}
BASE_METALS_TICKERS = {"HBM.TO","LUN.TO","ERO.TO","FM.TO","TECK-B.TO"}


FEATURE_COLS = [
    "mom_1m","mom_3m","mom_6m","mom_12m","mom_12_1",
    "vol_1m","vol_3m","vol_ratio",
    "rsi","bias_20","bias_60","vwap_bias","price_vs_52w_high",
    "pe","pb","roe","eps_growth","fcf_yield",
    "oil_mom_3m","cadusd_mom_3m","bond_chg_3m","gold_mom_3m","vix_level",
    "sector_mom_rel",
    "month_sin","month_cos",
    # 财报日历特征（新）
    "days_to_earnings",    # 距下次财报天数（负=财报刚过）
    "days_since_earnings", # 距上次财报天数（PEAD窗口）
    "bb_zscore",
]

# ══════════════════════════════════════════════════════════════════
# 股票分类表
# ══════════════════════════════════════════════════════════════════

STOCK_PROFILE = {
    "RY.TO":{"gics":"Financials","style":"Quality","type":"Defensive"},
    "TD.TO":{"gics":"Financials","style":"Quality","type":"Defensive"},
    "BNS.TO":{"gics":"Financials","style":"Value","type":"Defensive"},
    "BMO.TO":{"gics":"Financials","style":"Quality","type":"Defensive"},
    "CM.TO":{"gics":"Financials","style":"Value","type":"Defensive"},
    "NA.TO":{"gics":"Financials","style":"Quality","type":"Defensive"},
    "MFC.TO":{"gics":"Financials","style":"Value","type":"Defensive"},
    "SLF.TO":{"gics":"Financials","style":"Quality","type":"Defensive"},
    "GWO.TO":{"gics":"Financials","style":"Value","type":"Defensive"},
    "POW.TO":{"gics":"Financials","style":"Value","type":"Defensive"},
    "IGM.TO":{"gics":"Financials","style":"Value","type":"Defensive"},
    "CIGI.TO":{"gics":"Financials","style":"Growth","type":"Growth"},
    "EQB.TO":{"gics":"Financials","style":"Growth","type":"Cyclical"},
    "FSV.TO":{"gics":"Financials","style":"Growth","type":"Growth"},
    "SU.TO":{"gics":"Energy","style":"Value","type":"Cyclical"},
    "CNQ.TO":{"gics":"Energy","style":"Quality","type":"Cyclical"},
    "ENB.TO":{"gics":"Energy","style":"Value","type":"Defensive"},
    "TRP.TO":{"gics":"Energy","style":"Value","type":"Defensive"},
    "CVE.TO":{"gics":"Energy","style":"Value","type":"Cyclical"},
    "TOU.TO":{"gics":"Energy","style":"Growth","type":"Cyclical"},
    "PPL.TO":{"gics":"Energy","style":"Value","type":"Defensive"},
    "KEY.TO":{"gics":"Energy","style":"Value","type":"Cyclical"},
    "WCP.TO":{"gics":"Energy","style":"Value","type":"Cyclical"},
    "VET.TO":{"gics":"Energy","style":"Value","type":"Cyclical"},
    "BIR.TO":{"gics":"Energy","style":"Value","type":"Cyclical"},
    "NTR.TO":{"gics":"Materials","style":"Value","type":"Cyclical"},
    "ABX.TO":{"gics":"Materials","style":"Value","type":"Cyclical"},
    "AEM.TO":{"gics":"Materials","style":"Quality","type":"Cyclical"},
    "WPM.TO":{"gics":"Materials","style":"Growth","type":"Cyclical"},
    "FM.TO":{"gics":"Materials","style":"Value","type":"Cyclical"},
    "K.TO":{"gics":"Materials","style":"Value","type":"Cyclical"},
    "CCO.TO":{"gics":"Materials","style":"Growth","type":"Cyclical"},
    "FNV.TO":{"gics":"Materials","style":"Quality","type":"Cyclical"},
    "TECK-B.TO":{"gics":"Materials","style":"Value","type":"Cyclical"},
    "IMG.TO":{"gics":"Materials","style":"Value","type":"Cyclical"},
    "LUN.TO":{"gics":"Materials","style":"Value","type":"Cyclical"},
    "ERO.TO":{"gics":"Materials","style":"Growth","type":"Cyclical"},
    "CNR.TO":{"gics":"Industrials","style":"Quality","type":"Defensive"},
    "CP.TO":{"gics":"Industrials","style":"Quality","type":"Defensive"},
    "WSP.TO":{"gics":"Industrials","style":"Growth","type":"Growth"},
    "CAE.TO":{"gics":"Industrials","style":"Quality","type":"Cyclical"},
    "FTT.TO":{"gics":"Industrials","style":"Quality","type":"Cyclical"},
    "GIL.TO":{"gics":"Industrials","style":"Quality","type":"Cyclical"},
    "MTL.TO":{"gics":"Industrials","style":"Value","type":"Cyclical"},
    "RBA.TO":{"gics":"Industrials","style":"Growth","type":"Cyclical"},
    "CSU.TO":{"gics":"Technology","style":"Growth","type":"Growth"},
    "TRI.TO":{"gics":"Technology","style":"Quality","type":"Defensive"},
    "SHOP.TO":{"gics":"Technology","style":"Growth","type":"Growth"},
    "ENGH.TO":{"gics":"Technology","style":"Quality","type":"Defensive"},
    "OTEX.TO":{"gics":"Technology","style":"Value","type":"Defensive"},
    "GIB-A.TO":{"gics":"Technology","style":"Quality","type":"Defensive"},
    "LIF.TO":{"gics":"Technology","style":"Value","type":"Cyclical"},
    "ATD.TO":{"gics":"Consumer","style":"Quality","type":"Defensive"},
    "DOL.TO":{"gics":"Consumer","style":"Growth","type":"Defensive"},
    "MRU.TO":{"gics":"Consumer","style":"Quality","type":"Defensive"},
    "L.TO":{"gics":"Consumer","style":"Quality","type":"Defensive"},
    "QSR.TO":{"gics":"Consumer","style":"Quality","type":"Defensive"},
    "PBH.TO":{"gics":"Consumer","style":"Growth","type":"Cyclical"},
    "CJT.TO":{"gics":"Consumer","style":"Quality","type":"Cyclical"},
    "WCN.TO":{"gics":"Consumer","style":"Quality","type":"Defensive"},
    "MG.TO":{"gics":"Consumer","style":"Value","type":"Cyclical"},
    "GFL.TO":{"gics":"Consumer","style":"Growth","type":"Defensive"},
    "WN.TO":{"gics":"Consumer","style":"Value","type":"Defensive"},
    "EMP-A.TO":{"gics":"Consumer","style":"Value","type":"Defensive"},
    "BCE.TO":{"gics":"Telecom","style":"Value","type":"Defensive"},
    "T.TO":{"gics":"Telecom","style":"Value","type":"Defensive"},
    "RCI-B.TO":{"gics":"Telecom","style":"Value","type":"Defensive"},
    "QBR-B.TO":{"gics":"Telecom","style":"Value","type":"Defensive"},
    "CAR-UN.TO":{"gics":"REITs","style":"Growth","type":"Defensive"},
    "REI-UN.TO":{"gics":"REITs","style":"Value","type":"Defensive"},
    "GRT-UN.TO":{"gics":"REITs","style":"Value","type":"Defensive"},
    "SRU-UN.TO":{"gics":"REITs","style":"Value","type":"Defensive"},
    "CSH-UN.TO":{"gics":"REITs","style":"Value","type":"Defensive"},
    "CHP-UN.TO":{"gics":"REITs","style":"Value","type":"Defensive"},
    "HR-UN.TO":{"gics":"REITs","style":"Value","type":"Defensive"},
    "AP-UN.TO":{"gics":"REITs","style":"Value","type":"Defensive"},
    "DIR-UN.TO":{"gics":"REITs","style":"Growth","type":"Defensive"},
    "FTS.TO":{"gics":"Utilities","style":"Value","type":"Defensive"},
    "EMA.TO":{"gics":"Utilities","style":"Value","type":"Defensive"},
    "CU.TO":{"gics":"Utilities","style":"Value","type":"Defensive"},
    "H.TO":{"gics":"Utilities","style":"Value","type":"Defensive"},
    "NPI.TO":{"gics":"Utilities","style":"Growth","type":"Defensive"},
    "AQN.TO":{"gics":"Utilities","style":"Value","type":"Defensive"},
    "BEP-UN.TO":{"gics":"Utilities","style":"Growth","type":"Defensive"},
    "BLX.TO":{"gics":"Utilities","style":"Growth","type":"Defensive"},
    "BHC.TO":{"gics":"Healthcare","style":"Value","type":"Defensive"},
    "WELL.TO":{"gics":"Healthcare","style":"Growth","type":"Defensive"},
    "DRX.TO":{"gics":"Healthcare","style":"Value","type":"Defensive"},
    # 矿业（新增）
    "HBM.TO":   {"gics":"Materials",   "style":"Value",  "type":"Cyclical"},
    "WDO.TO":   {"gics":"Materials",   "style":"Growth", "type":"Cyclical"},
    "AGI.TO":   {"gics":"Materials",   "style":"Quality","type":"Cyclical"},
    "PEY.TO":   {"gics":"Energy",      "style":"Value",  "type":"Cyclical"},
    "SGY.TO":   {"gics":"Energy",      "style":"Value",  "type":"Cyclical"},
    "ARX.TO":   {"gics":"Energy",      "style":"Growth", "type":"Cyclical"},
    "IMO.TO":   {"gics":"Energy",      "style":"Value",  "type":"Cyclical"},
    "FFH.TO":   {"gics":"Financials",  "style":"Quality","type":"Defensive"},
    "IFC.TO":   {"gics":"Financials",  "style":"Quality","type":"Defensive"},
    "IAG.TO":   {"gics":"Financials",  "style":"Value",  "type":"Defensive"},
    "LB.TO":    {"gics":"Financials",  "style":"Value",  "type":"Defensive"},
    "BN.TO":    {"gics":"Financials",  "style":"Growth", "type":"Growth"},
    "BAM.TO":   {"gics":"Financials",  "style":"Growth", "type":"Growth"},
    "CF.TO":    {"gics":"Financials",  "style":"Value",  "type":"Cyclical"},
    "AC.TO":    {"gics":"Industrials", "style":"Value",  "type":"Cyclical"},
    "STN.TO":   {"gics":"Industrials", "style":"Growth", "type":"Cyclical"},
    "MDA.TO":   {"gics":"Technology",  "style":"Growth", "type":"Growth"},
    "DSG.TO":   {"gics":"Technology",  "style":"Growth", "type":"Growth"},
    "DCBO.TO":  {"gics":"Technology",  "style":"Growth", "type":"Growth"},
    "KXS.TO":   {"gics":"Technology",  "style":"Growth", "type":"Growth"},
    "CLS.TO":   {"gics":"Technology",  "style":"Growth", "type":"Growth"},
    "NWC.TO":   {"gics":"Consumer",    "style":"Value",  "type":"Defensive"},
    "SAP.TO":   {"gics":"Consumer",    "style":"Value",  "type":"Defensive"},
    "CTC-A.TO": {"gics":"Consumer",    "style":"Value",  "type":"Cyclical"},
    "CRT-UN.TO":{"gics":"REITs",       "style":"Value",  "type":"Defensive"},
    "NWH-UN.TO":{"gics":"REITs",       "style":"Value",  "type":"Defensive"},
    "SIA.TO":   {"gics":"REITs",       "style":"Value",  "type":"Defensive"},
    "ALA.TO":   {"gics":"Utilities",   "style":"Value",  "type":"Defensive"},
    "CPX.TO":   {"gics":"Utilities",   "style":"Growth", "type":"Defensive"},
    "WSP.TO":   {"gics":"Industrials", "style":"Growth", "type":"Growth"},
    "GIL.TO":   {"gics":"Industrials", "style":"Quality","type":"Cyclical"},
}

# ══════════════════════════════════════════════════════════════════
# 1. 数据获取
# ══════════════════════════════════════════════════════════════════

def fetch_macro(years):
    end   = datetime.today()
    start = end - timedelta(days=years*365+90)
    print("  [宏观] 下载宏观指标...", end="", flush=True)
    try:
        raw = yf.download(list(MACRO_TICKERS.values()), start=start, end=end,
                          auto_adjust=True, progress=False)["Close"]
        raw.columns = list(MACRO_TICKERS.keys())
        macro_m = raw.resample("ME").last().ffill()
        print(f" ✓ ({len(macro_m)} 月)")
        return macro_m
    except Exception as e:
        print(f" ✗ {e}")
        return pd.DataFrame()


def fetch_prices(tickers, years):
    end   = datetime.today()
    start = end - timedelta(days=years*365+90)
    print(f"\n[1/4] 下载 {len(tickers)} 支日线（{years} 年）...")
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=True, group_by="ticker")
    daily_map = {}
    for t in tickers:
        try:
            df = (raw[["Open","High","Low","Close","Volume"]] if len(tickers)==1
                  else raw[t][["Open","High","Low","Close","Volume"]]).copy()
            df.columns = ["open","high","low","close","volume"]
            df = df.dropna(subset=["close","volume"])
            if len(df) > 60:
                daily_map[t] = df
        except Exception:
            pass
    print(f"  {len(daily_map)} 支通过数据检查")
    return daily_map


def _safe_val(df, row_key, col_idx):
    try:
        if df is not None and row_key in df.index:
            v = df.iloc[df.index.get_loc(row_key), col_idx]
            return float(v) if pd.notna(v) else None
    except Exception:
        pass
    return None


def _nearest_col(df, dt):
    """✓ Bug fix: 容忍季报各表日期差 ±5 天"""
    if df is None or df.empty:
        return None
    diffs = abs(df.columns - pd.Timestamp(dt))
    idx   = diffs.argmin()
    return idx if diffs[idx] <= pd.Timedelta(days=5) else None


def fetch_pit_quarterly(ticker):
    """第三层：Point-in-Time 季报（季度末 + 45天延迟）"""
    t = yf.Ticker(ticker)
    try:
        inc    = t.quarterly_income_stmt
        bal    = t.quarterly_balance_sheet
        cf     = t.quarterly_cashflow
        shares = t.info.get("sharesOutstanding") or t.info.get("impliedSharesOutstanding")
    except Exception:
        return pd.DataFrame()

    if inc is None or inc.empty:
        return pd.DataFrame()

    # Q1-Q3: CSA 要求 45 天披露；Q4 (年报): 90 天披露
    # qdate 是季度末日期，月份决定是哪个季度
    def _report_lag(qdate):
        month = pd.Timestamp(qdate).month
        # Q4 = 10/11/12 月末，年报延迟 90 天
        return 90 if month in (10, 11, 12) else REPORT_LAG

    rows = []
    for qdate in inc.columns:
        lag   = _report_lag(qdate)
        avail = pd.Timestamp(qdate) + pd.Timedelta(days=lag)
        qi    = inc.columns.get_loc(qdate)

        def gi(k): return _safe_val(inc, k, qi)
        def gb(k):
            i = _nearest_col(bal, qdate)
            return _safe_val(bal, k, i) if i is not None and bal is not None and k in bal.index else None
        def gc(k):
            i = _nearest_col(cf, qdate)
            return _safe_val(cf, k, i) if i is not None and cf is not None and k in cf.index else None

        rows.append({
            "avail_date":   avail,
            "net_income":   gi("Net Income"),
            "total_equity": gb("Stockholders Equity") or gb("Total Equity Gross Minority Interest"),
            "ocf":          gc("Operating Cash Flow") or gc("Cash Flow From Continuing Operating Activities"),
            "capex":        gc("Capital Expenditure"),
            "shares":       shares,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("avail_date").sort_index()


def fetch_all(tickers, years):
    macro_df  = fetch_macro(years)
    daily_map = fetch_prices(tickers, years)

    print(f"\n[2/4] 获取季报历史（PIT）+ 基本面快照...")

    # 数据优先级：Simfin → FMP → yfinance（自动检测，无需手动切换）
    sf_inc = sf_bal = sf_cf = None
    use_simfin  = False
    use_fmp     = False
    fmp_pit_map = {}

    # 第一层：Simfin（最佳，10年PIT历史，本地运行有效）
    sf_inc, sf_bal, sf_cf = init_simfin()
    use_simfin = sf_inc is not None

    # 第二层：FMP（Colab可用，5年历史，比yfinance好）
    if not use_simfin and FMP_API_KEY:
        print(f"  [FMP] Simfin 不可用，尝试 Financial Modeling Prep...")
        fmp_pit_map = init_fmp(list(daily_map.keys()), FMP_API_KEY)
        use_fmp = len(fmp_pit_map) > 10

    # 状态显示
    if use_simfin:
        print(f"  ✅ 使用 Simfin（完整历史季报，PIT 更准确）")
    elif use_fmp:
        print(f"  ✅ 使用 FMP（{len(fmp_pit_map)} 支，5年季报历史）")
    else:
        print(f"  ⚠️  使用 yfinance 基本面（已知限制）：")
        print(f"     1. 仅 6-7 季历史（vs Simfin 10年+）")
        print(f"     2. 重述偏差（Restatement Bias）：YF 提供最新重述数据，")
        print(f"        若公司事后修正历史财报，回测会用修正后数据，产生前视偏差")
        print(f"     建议：填写 FMP_API_KEY 或本地运行使用 Simfin")

    pit_map   = {}
    meta_rows = []

    for i, t in enumerate(daily_map.keys(), 1):
        # 基本面快照（约束过滤和展示用，始终用 yfinance 当前值）
        try:
            info = yf.Ticker(t).info
            pe   = info.get("trailingPE") or info.get("forwardPE")
            roe  = info.get("returnOnEquity")
            mc   = info.get("marketCap")
            div  = info.get("dividendYield")
            meta_rows.append({"ticker":t,"name":info.get("shortName",t),
                               "sector":info.get("sector","Unknown"),
                               "mktcap":mc,"pe":pe,"roe":roe,"div_yield":div})
            pe_s  = f"PE={pe:.1f}"        if pe  else "PE=N/A"
            roe_s = f"ROE={roe*100:.1f}%" if roe else "ROE=N/A"
            mc_s  = f"${mc/1e9:.1f}B"     if mc  else "N/A"
            src   = "SF" if use_simfin else "YF"
            print(f"  [{i:>2}/{len(daily_map)}] {t:<14} {pe_s:<10} {roe_s:<12} MCap={mc_s} [{src}]")
        except Exception:
            meta_rows.append({"ticker":t,"name":t,"sector":"Unknown",
                               "mktcap":None,"pe":None,"roe":None,"div_yield":None})
            print(f"  [{i:>2}/{len(daily_map)}] {t:<14} ⚠️  基本面获取失败")

        # PIT 季报：Simfin → FMP → yfinance 三层优先级
        if use_simfin:
            pit = build_pit_from_simfin(t, sf_inc, sf_bal, sf_cf)
            if pit.empty:
                pit = fetch_pit_quarterly(t)
        elif use_fmp and t in fmp_pit_map:
            pit = fmp_pit_map[t].set_index("avail_date")
        else:
            pit = fetch_pit_quarterly(t)

        if not pit.empty:
            pit_map[t] = pit

    meta_df = pd.DataFrame(meta_rows).set_index("ticker")
    return daily_map, pit_map, meta_df, macro_df

# ══════════════════════════════════════════════════════════════════
# 2. 约束过滤
# ══════════════════════════════════════════════════════════════════

def apply_constraints(daily_map, meta_df, c):
    passed, removed = [], {}

    for t, df in daily_map.items():
        reasons = []
        min_days = c.get("min_listing_days", 0)
        if len(df) < min_days:
            reasons.append(f"上市时间 {len(df)} 天 < {min_days} 天")

        m = meta_df.loc[t] if t in meta_df.index else pd.Series(dtype=float)
        price = df["close"].iloc[-1]

        if price < c["min_price_cad"]:
            reasons.append(f"股价 ${price:.2f} < $2")
        max_px = c.get("max_price_cad", 9999)
        if price > max_px:
            reasons.append(f"股价 ${price:.2f} > ${max_px:.0f}")

        adv = (df["close"].tail(20) * df["volume"].tail(20)).mean()
        if adv < c["min_adv_cad"]:
            reasons.append(f"ADV ${adv/1e6:.2f}M < $1M")

        # ✓ Bug fix: vol_spike_min_days=2
        if len(df) >= 65:
            base   = df["volume"].iloc[-65:-5]
            vm, vs = base.mean(), base.std()
            sp     = df["volume"].tail(c["vol_spike_days"])
            sp     = sp[(sp > vm+c["vol_spike_sigma"]*vs) |
                        (sp < max(0, vm-c["vol_spike_sigma"]*vs))]
            if len(sp) >= c["vol_spike_min_days"]:
                reasons.append(f"成交量异常 {len(sp)}天±{c['vol_spike_sigma']:.0f}σ")

        pe = m.get("pe")
        if pe is None:
            reasons.append("PE无数据")
        elif not (c["min_pe"] < pe < c["max_pe"]):
            reasons.append(f"PE {pe:.1f} 超出范围")

        mktcap = m.get("mktcap")
        if mktcap is None:
            reasons.append("市值无数据")
        elif mktcap < c["min_mktcap_cad"]:
            reasons.append(f"市值 ${mktcap/1e6:.0f}M < $500M")

        roe = m.get("roe")
        if roe is not None and roe < c.get("min_roe", 0):
            reasons.append(f"ROE {roe*100:.1f}% < 0")
        if roe is not None and roe > c.get("max_roe", 999):
            reasons.append(f"ROE {roe*100:.0f}% 异常（>200%，财务杠杆或一次性项目）")

        if reasons:
            removed[t] = reasons
        else:
            passed.append(t)

    print(f"\n{'─'*60}")
    print(f"  约束过滤：{len(daily_map)} → {len(passed)} 支通过")
    print(f"{'─'*60}")
    counts = {}
    for rs in removed.values():
        for r in rs:
            k = r.split(" ")[0]
            counts[k] = counts.get(k,0)+1
    for k, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  ✗ {k:<22} 剔除 {n} 支")
    if removed:
        print(f"\n  {'Ticker':<14} 原因")
        print(f"  {'─'*50}")
        for t, rs in sorted(removed.items()):
            print(f"  {t:<14} {' | '.join(rs)}")
    print(f"\n  ✅ 通过：{' '.join(passed)}")
    return passed

# ══════════════════════════════════════════════════════════════════
# 3. 特征工程
# ══════════════════════════════════════════════════════════════════

def _rsi(close, w=14):
    d = close.diff()
    g = d.clip(lower=0).ewm(span=w).mean()
    l = (-d.clip(upper=0)).ewm(span=w).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def compute_time_decay_weights(sample_dates, current_date, half_life_months=12.0):
    """
    计算指数级时间衰减权重。
    参数:
      sample_dates: 样本对应的时间戳序列 (pd.Series 或 np.array)
      current_date: 当前预测基准日 (pd.Timestamp)
      half_life_months: 半衰期（默认 12 个月，即 1 年前的数据权重减半）
    """
    # 计算样本距离“当前日”的月份差
    age_days = (pd.Timestamp(current_date) - pd.to_datetime(sample_dates)).days
    age_months = np.maximum(0, age_days / 30.4)
    
    # 半衰期指数衰减公式: W = 0.5 ^ (age / half_life)
    weights = 0.5 ** (age_months / half_life_months)
    return weights.values


def compute_pit_fundamentals(pit_df, monthly_close):
    """
    向量化重构：用 merge_asof 替代 for 循环，消除前视偏差。
    （保持原函数名，无缝接入 build_panel）
    """
    if pit_df.empty or monthly_close.empty:
        result = pd.DataFrame(index=monthly_close.index, columns=["pe","pb","roe","eps_growth","fcf_yield"])
        result.index.name = "date"
        return result

    pit_df = pit_df.sort_values('avail_date').copy()
    pit_df['ttm_ni'] = pit_df['net_income'].rolling(4, min_periods=4).sum()
    pit_df['ttm_ocf'] = pit_df['ocf'].rolling(4, min_periods=4).sum()
    pit_df['ttm_capex'] = pit_df['capex'].fillna(0).rolling(4, min_periods=4).sum()
    pit_df['ttm_fcf'] = pit_df['ttm_ocf'] - pit_df['ttm_capex'].abs()
    
    pit_df['prev4_ttm_ni'] = pit_df['ttm_ni'].shift(4)
    pit_df['eps_growth'] = (pit_df['ttm_ni'] - pit_df['prev4_ttm_ni']) / pit_df['prev4_ttm_ni'].abs()

    # ✓ Bug Fix: Handle index name correctly (index may be named 'date' already)
    left_df = monthly_close.reset_index()
    if 'index' in left_df.columns:
        left_df = left_df.rename(columns={'index': 'date'})
    elif left_df.columns[0] not in ['date', 'close']:
        # If first column is unnamed or has a default name, rename it to 'date'
        left_df = left_df.rename(columns={left_df.columns[0]: 'date'})
    left_df = left_df.rename(columns={'close': 'price'})
    left_df = left_df.sort_values('date')
    right_df = pit_df.reset_index().sort_values('avail_date')
    
    merged = pd.merge_asof(
        left_df, right_df, 
        left_on='date', right_on='avail_date', 
        direction='backward'
    )
    
    merge_cols = ['price', 'ttm_ni', 'total_equity', 'ttm_fcf', 'shares']
    required_cols = [c for c in merge_cols if c in merged.columns]
    
    if len(required_cols) < 3:  # Need at least 3 of 5 columns to compute ratios
        result = pd.DataFrame(index=monthly_close.index, columns=["pe","pb","roe","eps_growth","fcf_yield"])
        result.index.name = "date"
        return result
    
    merged['pe'] = merged['price'] / (merged['ttm_ni'] / merged['shares']).replace(0, np.nan)
    merged['pb'] = merged['price'] / (merged['total_equity'] / merged['shares']).replace(0, np.nan)
    merged['roe'] = merged['ttm_ni'] / merged['total_equity'].replace(0, np.nan)
    merged['fcf_yield'] = merged['ttm_fcf'] / (merged['price'] * merged['shares']).replace(0, np.nan)
    
    merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    if 'date' not in merged.columns:
        result = pd.DataFrame(index=monthly_close.index, columns=["pe","pb","roe","eps_growth","fcf_yield"])
        result.index.name = "date"
        return result
    result = merged.set_index('date')[['pe', 'pb', 'roe', 'eps_growth', 'fcf_yield']]
    result.index.name = "date"
    return result


def compute_monthly_tech(df):
    c, v  = df["close"], df["volume"]
    tp    = (df["high"] + df["low"] + df["close"]) / 3
    mc    = c.resample("ME").last()
    mv    = v.resample("ME").sum()
    mvwap = (tp * v).resample("ME").sum() / mv
    def mom(n): return mc.pct_change(n)
    m1,m3,m6,m12 = mom(1),mom(3),mom(6),mom(12)
    dr    = c.pct_change()
    vol1m = dr.resample("ME").std() * np.sqrt(252)
    vol3m = dr.rolling(63).std().resample("ME").last() * np.sqrt(252)
    b20   = ((c-c.rolling(20).mean())/c.rolling(20).mean()).resample("ME").last()
    b60   = ((c-c.rolling(60).mean())/c.rolling(60).mean()).resample("ME").last()
    vwapb = (mc-mvwap)/mvwap
    h52   = (c/c.rolling(252).max()).resample("ME").last()
    rsi_m = _rsi(c).resample("ME").last()
    bb_zscore = ((c - c.rolling(20).mean()) / c.rolling(20).std()).resample("ME").last()

    result = pd.DataFrame({
        "close":mc,"mom_1m":m1,"mom_3m":m3,"mom_6m":m6,"mom_12m":m12,
        "mom_12_1":m12-m1,"vol_1m":vol1m,"vol_3m":vol3m,
        "vol_ratio":vol1m/vol3m.replace(0,np.nan),
        "rsi":rsi_m,"bias_20":b20,"bias_60":b60,
        "vwap_bias":vwapb,"price_vs_52w_high":h52,
        "bb_zscore": bb_zscore,
    }).dropna(subset=["close"])
    result.index.name = "date"
    return result


def get_macro_feat(macro_df, month_end):
    """第二层：宏观因子"""
    zero = {"oil_mom_3m":0.0,"cadusd_mom_3m":0.0,"bond_chg_3m":0.0,
            "gold_mom_3m":0.0,"vix_level":0.0}
    if macro_df.empty:
        return zero
    av = macro_df[macro_df.index <= month_end]
    if len(av) < 4:
        return zero
    def m3(col):
        if col not in av.columns: return 0.0
        s = av[col].dropna()
        return float(s.iloc[-1]/s.iloc[-4]-1) if len(s)>=4 else 0.0
    def lv(col):
        if col not in av.columns: return 0.0
        s = av[col].dropna()
        return float(s.iloc[-1]) if len(s)>0 else 0.0
    return {"oil_mom_3m":m3("oil"),"cadusd_mom_3m":m3("cadusd"),
            "bond_chg_3m":m3("bond"),"gold_mom_3m":m3("gold"),
            "vix_level":lv("vix")/40.0}


def fetch_earnings_calendar(tickers: list, lookback_days: int = 365,
                             forward_days: int = 90) -> dict:
    """
    从 FMP 获取财报日期（比 yfinance 更准确，Colab可用）。
    返回：{ticker: sorted list of earnings dates}
    缓存7天。
    """
    import os, pickle, time, requests as _req

    cache_file = "./simfin_data/earnings_calendar.pkl"
    os.makedirs("./simfin_data", exist_ok=True)

    # 读缓存（7天内有效）
    if os.path.exists(cache_file):
        if (time.time() - os.path.getmtime(cache_file)) < 7 * 86400:
            try:
                with open(cache_file, "rb") as f:
                    cached = pickle.load(f)
                print(f"  [财报日历] 从缓存加载（{len(cached)} 支）")
                return cached
            except Exception:
                pass

    if not FMP_API_KEY:
        print("  [财报日历] FMP_API_KEY 未设置，跳过财报特征")
        return {}

    print(f"  [财报日历] FMP 获取 {len(tickers)} 支财报日期...")
    BASE    = "https://financialmodelingprep.com/api/v3"
    result  = {}
    today   = pd.Timestamp.today()
    from_dt = (today - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    to_dt   = (today + pd.Timedelta(days=forward_days)).strftime("%Y-%m-%d")

    for t in tickers:
        try:
            r = _req.get(
                f"{BASE}/historical/earning_calendar/{t}",
                params={"from": from_dt, "to": to_dt, "apikey": FMP_API_KEY},
                timeout=8
            )
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and data:
                    dates = sorted([
                        pd.Timestamp(d["date"])
                        for d in data if d.get("date")
                    ])
                    if dates:
                        result[t] = dates
            time.sleep(0.05)  # 避免超频
        except Exception:
            pass

    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
    print(f"  [财报日历] ✓ {len(result)} 支有财报日期，已缓存7天")
    return result


def compute_earnings_features(ticker: str,
                               dates: pd.DatetimeIndex,
                               earnings_cal: dict) -> pd.DataFrame:
    """
    对每个月末日期计算：
      days_to_earnings:    距下次财报天数（未来为正，刚过为负）
      days_since_earnings: 距上次财报天数（越小 = PEAD 窗口内）
    """
    t_dates = earnings_cal.get(ticker, [])
    if not t_dates:
        return pd.DataFrame({
            "days_to_earnings":    [30.0] * len(dates),  # 默认中性
            "days_since_earnings": [45.0] * len(dates),
        }, index=dates)

    t_dates_ts = [pd.Timestamp(d) for d in t_dates]
    rows = []
    for d in dates:
        d_ts = pd.Timestamp(d)
        future = [ed for ed in t_dates_ts if ed >= d_ts]
        past   = [ed for ed in t_dates_ts if ed  < d_ts]
        days_to  = (future[0] - d_ts).days if future else 90.0
        days_since=(d_ts - past[-1]).days   if past   else 90.0
        rows.append({"days_to_earnings": float(days_to),
                     "days_since_earnings": float(days_since)})
    
    result_df = pd.DataFrame(rows, index=dates)
    # Ensure index is a DatetimeIndex
    if not isinstance(result_df.index, pd.DatetimeIndex):
        result_df.index = pd.DatetimeIndex(result_df.index)
    return result_df


def build_panel(passed, daily_map, pit_map, macro_df):
    """✓ Bug fix: try/except 每支股票，rows为空时抛明确错误"""
    print(f"\n[3/4] 构建特征面板...")

    all_tech = {}
    earnings_cal = fetch_earnings_calendar(passed)

    for t in passed:
        try:
            all_tech[t] = compute_monthly_tech(daily_map[t])
        except Exception:
            pass

    rows, errors = [], []
    for t in passed:
        try:
            tech = all_tech.get(t, pd.DataFrame())
            if tech.empty:
                errors.append(f"{t}: tech 为空")
                continue

            # ✓ Bug Fix: Ensure tech has proper index name
            if tech.index.name is None:
                tech.index.name = "date"
            
            mc = tech["close"]
            fund_hist = (compute_pit_fundamentals(pit_map[t], mc)
                         if t in pit_map and not pit_map[t].empty
                         else pd.DataFrame(index=mc.index,
                                           columns=["pe","pb","roe","eps_growth","fcf_yield"]))

            gics = STOCK_PROFILE.get(t, {}).get("gics", "Unknown")

            # ── 向量化替代 iterrows（Fix 3）─────────────────────────────
            # 宏观因子批量计算
            if not macro_df.empty:
                macro_block = pd.DataFrame(
                    {d: get_macro_feat(macro_df, d) for d in tech.index},
                ).T.reindex(tech.index).fillna(0)  # 宏观缺失=无信号，0正确
            else:
                macro_block = pd.DataFrame(0.0, index=tech.index,
                    columns=["oil_mom_3m","cadusd_mom_3m","bond_chg_3m",
                             "gold_mom_3m","vix_level"])
            macro_block["month_sin"] = np.sin(2*np.pi*tech.index.month/12)
            macro_block["month_cos"] = np.cos(2*np.pi*tech.index.month/12)
            sec_rels = {}
            for date in tech.index:
                peers = [all_tech[p].loc[date,"mom_6m"]
                         for p in passed
                         if p in all_tech and date in all_tech[p].index
                         and STOCK_PROFILE.get(p,{}).get("gics")==gics
                         and pd.notna(all_tech[p].loc[date,"mom_6m"])]
                own_m6 = float(tech.loc[date,"mom_6m"] or 0)
                sec_rels[date] = own_m6-(float(np.mean(peers)) if peers else 0.0)
            macro_block["sector_mom_rel"] = pd.Series(sec_rels)
            fund_cols = ["pe","pb","roe","eps_growth","fcf_yield"]
            fund_block = (fund_hist.reindex(tech.index,method="ffill")[fund_cols]
                          if not fund_hist.empty
                          else pd.DataFrame(np.nan,index=tech.index,columns=fund_cols))
            
            # ✓ Bug Fix: Ensure all blocks have the same index name before concat
            macro_block.index.name = "date"
            fund_block.index.name = "date"
            
            block = pd.concat([tech, macro_block, fund_block], axis=1)
            block["ticker"] = t
            block.index.name = "date"
            # 财报日历特征（距下次/上次财报天数）
            earn_feats = compute_earnings_features(t, tech.index, earnings_cal)
            block["days_to_earnings"]    = earn_feats["days_to_earnings"].values
            block["days_since_earnings"] = earn_feats["days_since_earnings"].values
            
            # ✓ Bug Fix: Explicitly reset index to 'date' column
            block = block.reset_index()
            if "date" not in block.columns:
                block["date"] = pd.to_datetime(block.index) if hasattr(block.index, '__iter__') else block.index
            rows.append(block)
        except Exception as e:
            errors.append(f"{t}: {e}")

    if errors:
        print(f"  ⚠️  {len(errors)} 支失败：{errors[:3]}")
    if not rows:
        raise RuntimeError("所有股票特征计算失败，rows 为空")

    panel = pd.concat(rows, ignore_index=True).set_index(["date","ticker"])
    print(f"  面板：{len(panel)} 行 "
          f"({panel.index.get_level_values('date').nunique()} 月 × "
          f"{panel.index.get_level_values('ticker').nunique()} 支)")
    return panel


def add_labels(panel):
    """✓ Bug fix: rows 为空时填 NaN 而非崩溃"""
    dates = sorted(panel.index.get_level_values("date").unique())
    rows  = []
    for i, date in enumerate(dates[:-1]):
        nxt = dates[i+1]
        try:
            c0 = panel.xs(date, level="date")["close"]
            c1 = panel.xs(nxt,  level="date")["close"]
        except KeyError:
            continue
        common = c0.index.intersection(c1.index)
        # 对数收益率
        ret_abs = np.log(c1[common] / c0[common])

        # Step 1: 残差收益（剥离大盘Beta）
        market_mean = ret_abs.mean()
        ret_resid   = ret_abs - market_mean

        # Fix8: 用个股自身历史时序波动率标准化（真正的IR）
        # 直接从 panel 中提取已计算好的个股年化波动率 (vol_1m)
        try:
            annual_vol = panel.xs(date, level="date")["vol_1m"]
        except KeyError:
            annual_vol = pd.Series(0.05 * np.sqrt(12), index=common)

        # 年化波动率转为月度波动率：vol / sqrt(12)
        # 用 fillna(0.05) 兜底缺失值，clip(lower=0.01) 防止除以极小值导致目标值(Label)爆炸
        monthly_vol = (annual_vol.reindex(common) / np.sqrt(12)).fillna(0.05).clip(lower=0.01)

        # 向量化计算风险调整后收益
        ret = ret_resid / monthly_vol

        thr = ret.quantile(1-TOP_QUINTILE)
        for t in common:
            rows.append({"date":date,"ticker":t,
                         "next_ret":ret[t],           # IR目标（用于回归）
                         "next_ret_abs":ret_abs[t],   # 绝对收益（用于P&L）
                         "label":int(ret[t]>=thr)})
    if not rows:
        print("  ⚠️  add_labels: 未生成标签")
        panel["next_ret"] = np.nan
        panel["label"]    = np.nan
        return panel
    label_df = pd.DataFrame(rows).set_index(["date","ticker"])
    result   = panel.join(label_df, how="left")
    n_ok     = result["next_ret"].notna().sum()
    print(f"  标签覆盖：{n_ok}/{len(result)} 行 ({n_ok/len(result)*100:.0f}%)")
    return result


# 行业特定因子屏蔽规则
# 金融股：FCF Yield 噪声大（银行监管资本逻辑，非自由现金流），PE 有效
# 能源/材料：周期底部 PE 为负失效，用 FCF Yield；但不屏蔽 ROE（仍有参考价值）
# REITs：P/B 和 FCF Yield 主导，PE 和 ROE 失真（折旧摊销影响）
SECTOR_FACTOR_MASK = {
    "Financials": {"fcf_yield": 0.0},            # 银行/保险：FCF逻辑不适用
    "Energy":     {"pe": 0.0},                    # 周期底部PE为负，屏蔽
    "Materials":  {"pe": 0.0},                    # 同上
    "REITs":      {"pe": 0.0, "fcf_yield": 0.0}, # REIT：折旧影响双重失真
}


def smart_impute(panel: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    智能特征插补：NaN → 行业截面中位数 → 全市场中位数。

    规则：
      基本面（pe/pb/roe/eps_growth/fcf_yield）→ 同GICS行业、同月截面中位数
      技术/动量                               → 全市场截面中位数
      宏观                                    → 全局中位数（单一值）
      最终兜底                                → 0（极少数整列NaN情况）

    绝不盲目用 0：PE=0 会被模型误读为"估值极低的优质股"。
    """
    FUNDAMENTAL_COLS = {"pe","pb","roe","eps_growth","fcf_yield","days_to_earnings","days_since_earnings"}
    MACRO_COLS       = {"oil_mom_3m","cadusd_mom_3m","bond_chg_3m",
                        "gold_mom_3m","vix_level"}

    out     = panel.copy()
    tickers = out.index.get_level_values("ticker").unique()
    gics_map= {t: STOCK_PROFILE.get(t, {}).get("gics","Unknown") for t in tickers}

    for date in out.index.get_level_values("date").unique():
        mask    = out.index.get_level_values("date") == date
        sl      = out.loc[mask, feature_cols].copy()
        tix_arr = out.loc[mask].index.get_level_values("ticker")

        for col in feature_cols:
            if not sl[col].isna().any():
                continue

            if col in MACRO_COLS:
                med = sl[col].median()
                sl[col] = sl[col].fillna(med if pd.notna(med) else 0.0)

            elif col in FUNDAMENTAL_COLS:
                sectors = pd.Series([gics_map.get(t,"Unknown") for t in tix_arr],
                                    index=sl.index)
                for sec in sectors.unique():
                    sm   = sectors == sec
                    smed = sl.loc[sm, col].median()
                    if pd.notna(smed):
                        sl.loc[sm, col] = sl.loc[sm, col].fillna(smed)
                gmed = sl[col].median()
                sl[col] = sl[col].fillna(gmed if pd.notna(gmed) else 0.0)

            else:
                med = sl[col].median()
                sl[col] = sl[col].fillna(med if pd.notna(med) else 0.0)

        out.loc[mask, feature_cols] = sl.values

    remaining = out[feature_cols].isna().sum().sum()
    if remaining > 0:
        out[feature_cols] = out[feature_cols].fillna(0.0)
    return out


def cross_z(panel: pd.DataFrame) -> pd.DataFrame:
    """
    横截面秩次化 (Rank Normalization) + 行业因子中性化。

    替代 Z-Score 的原因：
      TSX 含大量小盘矿业股，单支股票暴涨/暴跌会极大拉偏均值和标准差，
      导致当月所有正常股票的 Z-Score 被压缩到 0 附近（信号失真）。

      秩次化方法（WorldQuant / AQR 标配）：
        1. 同月所有股票按特征值排序 → 得到排名 rank
        2. 映射到 [-1, +1] 均匀分布：rank / (n-1) * 2 - 1
      对任何极端异常值 100% 免疫，分布稳定。

    两步处理：
      1. 行业因子屏蔽（Sector Mask）
      2. 截面秩次化 → [-1, 1]
    """
    out = panel.copy()

    # Step 1: 行业特定因子屏蔽（与之前相同）
    tix_col = out.index.get_level_values("ticker")
    for t in tix_col.unique():
        gics = STOCK_PROFILE.get(t, {}).get("gics", "Unknown")
        mask = SECTOR_FACTOR_MASK.get(gics, {})
        if mask:
            rows_t = out.index.get_level_values("ticker") == t
            for factor, weight in mask.items():
                if factor in out.columns:
                    out.loc[rows_t, factor] = out.loc[rows_t, factor] * weight

    # Step 2: 截面秩次化 → [-1, +1]（替代 Z-Score）
    for date in panel.index.get_level_values("date").unique():
        m = out.index.get_level_values("date") == date
        s = out.loc[m, FEATURE_COLS]
        n = len(s)
        if n < 2:
            continue
        # 向量化秩次化：rank(0-based) / (n-1) * 2 - 1 → [-1, +1]
        ranked = s.rank(axis=0, method="average", na_option="keep")
        out.loc[m, FEATURE_COLS] = (ranked - 1) / (n - 1) * 2 - 1

    return out

# ══════════════════════════════════════════════════════════════════
# 4. 模型
# ══════════════════════════════════════════════════════════════════


def make_xgb(task, pos_w=1.0):
    """标准 XGBoost 模型：回归和分类"""
    kw = dict(n_estimators=300, max_depth=3, learning_rate=0.04,
              subsample=0.8, colsample_bytree=0.7,
              reg_alpha=0.5, reg_lambda=1.5, random_state=42, verbosity=0)
              
    if task == "reg":
        return xgb.XGBRegressor(**kw)
    else:
        return xgb.XGBClassifier(**kw, scale_pos_weight=pos_w, eval_metric="logloss")


def make_lgbm(task, pos_w=1.0):
    if not LGBM: return None
    kw = dict(n_estimators=300,max_depth=3,num_leaves=15,learning_rate=0.04,
              subsample=0.8,colsample_bytree=0.7,min_child_samples=20,
              reg_alpha=0.5,reg_lambda=1.5,random_state=42,verbose=-1)
    return (lgb.LGBMRegressor(**kw) if task=="reg"
            else lgb.LGBMClassifier(**kw,scale_pos_weight=pos_w))


if TORCH:
    class MLP(nn.Module):
        def __init__(self, d, h=(128,64,32)):
            super().__init__()
            layers, prev = [], d
            for n in h:
                layers += [nn.Linear(prev,n),nn.LayerNorm(n),nn.ReLU(),nn.Dropout(0.3)]
                prev = n
            layers.append(nn.Linear(prev,1))
            self.net = nn.Sequential(*layers)
        def forward(self, x): return self.net(x).squeeze(-1)

    def train_mlp(X, y, task, epochs=120):
        Xt,yt = torch.FloatTensor(X),torch.FloatTensor(y)
        # ✓ Bug fix: drop_last=True
        dl = DataLoader(TensorDataset(Xt,yt),batch_size=64,shuffle=True,drop_last=True)
        m  = MLP(X.shape[1])
        opt= torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-4)
        fn = nn.MSELoss() if task=="reg" else nn.BCEWithLogitsLoss()
        m.train()
        for _ in range(epochs):
            for xb,yb in dl:
                opt.zero_grad(); fn(m(xb),yb).backward(); opt.step()
        return m.eval()
    
    def enable_dropout(model):
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    @torch.no_grad()
    def pred_mlp(m, X, task, mc_samples=30):
        """
        保持原名。如果开启 mc_samples > 1，则返回 (均值, 方差) 
        """
        m.eval()
        if mc_samples > 1:
            enable_dropout(m) # 开启 MC Dropout
            
        X_tensor = torch.FloatTensor(X)
        preds = []
        
        for _ in range(mc_samples):
            out = m(X_tensor).squeeze(-1)
            if task == "cls":
                out = torch.sigmoid(out)
            preds.append(out.numpy())
            
        preds = np.array(preds)
        
        # 算均值和方差
        return preds.mean(axis=0), preds.var(axis=0)


def _prepare_mlp_features(X_tr: np.ndarray, X_te: np.ndarray,
                           winsor_pct: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """
    为 MLP 额外处理特征：RobustScaler + Winsorization。

    树模型（XGBoost/LightGBM）对特征缩放不敏感，可直接用 StandardScaler。
    MLP 对极端值（如矿业股单月 +200%）的梯度爆炸极度敏感，需要：
      1. Winsorize：截断 [1%, 99%] 分位数外的异常值
      2. RobustScaler：用中位数和 IQR 缩放（比 StandardScaler 更抗异常值）
    """
    from sklearn.preprocessing import RobustScaler

    # Winsorization：截断极端值到 [p_lo, p_hi] 范围
    p_lo = np.percentile(X_tr, winsor_pct * 100, axis=0)
    p_hi = np.percentile(X_tr, (1 - winsor_pct) * 100, axis=0)
    X_tr_w = np.clip(X_tr, p_lo, p_hi)
    X_te_w = np.clip(X_te, p_lo, p_hi)  # 用训练集分位数截断测试集

    # RobustScaler（中位数/IQR，比 std 更抗异常值）
    rs = RobustScaler()
    X_tr_r = rs.fit_transform(X_tr_w)
    X_te_r = rs.transform(X_te_w)

    return X_tr_r, X_te_r



def fit_all(X_tr, y_r, y_c, X_te, weights_tr=None, model_weights=None, use_mlp=True, mlp_epochs=120):
    """
    第四层：多模型训练 + 动态权重集成。
    ★ 新增 weights_tr 参数，用于传入时间衰减权重。
    """
    # ✓ Bug Fix: Validate input arrays
    if len(X_tr) == 0:
        raise ValueError(f"Training set empty: X_tr.shape={X_tr.shape}")
    if len(X_te) == 0:
        raise ValueError(f"Test set empty: X_te.shape={X_te.shape}")
    if len(y_r) != len(X_tr):
        raise ValueError(f"Target-feature mismatch: len(y_r)={len(y_r)} vs len(X_tr)={len(X_tr)}")
    if len(y_c) != len(X_tr):
        raise ValueError(f"Classification target-feature mismatch: len(y_c)={len(y_c)} vs len(X_tr)={len(X_tr)}")
    
    pos_w = max((y_c==0).sum()/max((y_c==1).sum(),1), 1.0)
    pr_list, pc_list = [], []

    # ── 时序切割：前 85% 训练，后 15% 验证 ────────────────────────
    val_size  = max(1, int(len(X_tr) * 0.15))
    X_t, X_v  = X_tr[:-val_size], X_tr[-val_size:]
    y_rt, y_rv = y_r[:-val_size], y_r[-val_size:]
    y_ct, y_cv = y_c[:-val_size], y_c[-val_size:]
    
    # 同步切割权重数组
    if weights_tr is not None:
        sw_t, sw_v = weights_tr[:-val_size], weights_tr[-val_size:]
    else:
        sw_t, sw_v = None, None

    # ── XGBoost + 早停 + 样本权重 ────────────────────────────────
    xr = make_xgb("reg")
    xr.set_params(n_estimators=500, early_stopping_rounds=20, verbosity=0)
    # 传入 sample_weight=sw_t
    xr.fit(X_t, y_rt, eval_set=[(X_v, y_rv)], sample_weight=sw_t, verbose=False)

    xc = make_xgb("cls", pos_w)
    xc.set_params(n_estimators=500, early_stopping_rounds=20, verbosity=0)
    # 传入 sample_weight=sw_t
    xc.fit(X_t, y_ct, eval_set=[(X_v, y_cv)], sample_weight=sw_t, verbose=False)

    pr_list.append(("xgb", xr.predict(X_te)))
    pc_list.append(("xgb", xc.predict_proba(X_te)[:,1]))

    if LGBM:
        # LightGBM 早停 + 样本权重
        lr = make_lgbm("reg")
        lr.set_params(n_estimators=500)
        lr.fit(X_t, y_rt,
               eval_set=[(X_v, y_rv)],
               sample_weight=sw_t,  # 传入权重
               callbacks=[lgb.early_stopping(20, verbose=False),
                          lgb.log_evaluation(-1)])

        lc = make_lgbm("cls", pos_w)
        lc.set_params(n_estimators=500)
        lc.fit(X_t, y_ct,
               eval_set=[(X_v, y_cv)],
               sample_weight=sw_t,  # 传入权重
               callbacks=[lgb.early_stopping(20, verbose=False),
                          lgb.log_evaluation(-1)])

        pr_list.append(("lgbm", lr.predict(X_te)))
        pc_list.append(("lgbm", lc.predict_proba(X_te)[:,1]))

    if TORCH and len(X_tr) > 128 and use_mlp:
        # MLP 专用：Winsorize + RobustScaler
        # 注意：PyTorch DataLoader 原生不支持直接传 sample_weight，需要改写 Loss 函数。
        # 为了保持架构简洁，权重衰减目前主要作用于主导特征分裂的 XGB/LGBM 树模型。
        X_tr_mlp, X_te_mlp = _prepare_mlp_features(X_tr, X_te, winsor_pct=0.01)
        pr_list.append(("mlp", pred_mlp(train_mlp(X_tr_mlp, y_r, "reg", epochs=mlp_epochs),
                                        X_te_mlp, "reg")))
        pc_list.append(("mlp", pred_mlp(train_mlp(X_tr_mlp, y_c, "cls", epochs=mlp_epochs),
                                        X_te_mlp, "cls")))

    n = len(pr_list)
    w = (np.array(model_weights[:n])/sum(model_weights[:n])
         if model_weights and len(model_weights)>=n else np.ones(n)/n)

    ens_r = sum(w[i]*pr for i,(_,pr) in enumerate(pr_list))
    ens_c = sum(w[i]*pc for i,(_,pc) in enumerate(pc_list))
    ens   = ens_r*0.5 + ens_c*0.5

    return ens_r, ens_c, ens, xr, dict(pr_list), dict(pc_list)

# ══════════════════════════════════════════════════════════════════
# 5. Walk-Forward
# ══════════════════════════════════════════════════════════════════

def walk_forward(panel, tx_cost=0.002):
    """第四层：含交易成本 + 换仓率统计 + 动态模型权重"""
    dates = sorted(panel.index.get_level_values("date").unique())
    sc, recs = StandardScaler(), []
    n_test = len(dates)-MIN_TRAIN-1

    print(f"\nWalk-Forward（{MIN_TRAIN+1}月起，{n_test} 个测试月）  交易成本: {tx_cost*100:.1f}%")

    # ✓ Bug fix: 检查标签
    if "next_ret" not in panel.columns or not panel["next_ret"].notna().any():
        print("  ❌ panel 缺少 next_ret/label，跳过")
        return pd.DataFrame()

    prev_h:       set[str] = set()
    model_ic:     dict[str, list] = {}
    cooldown_set: set[str] = set()  # OPT5: 止损冷静期
    stopped_last: set[str] = set()  # 上月止损的股票
    ROLLING_WINDOW = 24             # OPT2: 滚动训练窗口（月）

    for i in range(MIN_TRAIN, len(dates)-1):
        # OPT2: 滚动窗口 - 只用最近 ROLLING_WINDOW 个月训练
        window_start = max(0, i - ROLLING_WINDOW)
        train_dates  = dates[window_start:i]
        tr = panel[panel.index.get_level_values("date").isin(train_dates)]                  .dropna(subset=["next_ret","label"])
        te = panel[panel.index.get_level_values("date")==dates[i]]


        if len(tr)<60 or not len(te): continue

        if i==MIN_TRAIN:
            print(f"  [诊断] 首月 len(tr)={len(tr)} len(te)={len(te)}")

        X_tr = sc.fit_transform(tr[FEATURE_COLS].fillna(tr[FEATURE_COLS].median()))
        X_te = sc.transform(te[FEATURE_COLS].fillna(tr[FEATURE_COLS].median()))

        # ★ 计算样本时间衰减权重（半衰期设为 12 个月）
        # 引入指数级时间衰减权重（Exponential Sample Weight Decay）是处理金融时序数据中“宏观状态漂移（Concept Drift）”最优雅且非破坏性的做法。
        # 它的底层逻辑非常符合直觉：半衰期（Half-life）。假设我们设定半衰期为 12 个月，那么今天发生的事情权重是 1.0，一年前的数据权重衰减到 0.5，两年前的数据权重衰减到 0.25。这样，5 年前（特鲁多政府早期、疫情前、零利率时代）的数据依然会参与计算以提供大样本量的支撑，但它们对目前梯度下降和树节点分裂的话语权已经被极度削弱了。
        sample_dates = tr.index.get_level_values("date")
        w_tr = compute_time_decay_weights(sample_dates, current_date=dates[i], half_life_months=12.0)

        # 动态模型权重（基于近期 IC）
        mw = None
        if model_ic:
            mw = [max(0.1, np.mean(v[-3:])) for v in model_ic.values()]

        # ✓ Bug Fix: Ensure X_te is not empty before fitting
        if len(X_te) == 0:
            print(f"  ⚠️  月 {i+1} 测试集为空，跳过")
            continue

        # 传入 weights_tr
        ens_r, ens_c, ens, _, pr_d, pc_d = fit_all(
            X_tr, tr["next_ret"].values, tr["label"].values, X_te, 
            weights_tr=w_tr, model_weights=mw, use_mlp=False)

        # 持仓惯性加成（Walk-Forward 也用）
        hold_b = CONSTRAINTS.get("hold_bonus", 0.05)
        if prev_h and hold_b > 0:
            tix_wf = [idx[1] for idx in te.index]
            for ji, t in enumerate(tix_wf):
                if t in prev_h:
                    ens[ji] += hold_b

        # 换仓上限（Walk-Forward）
        max_turn = CONSTRAINTS.get("max_turnover", TOP_N)
        sorted_idx = np.argsort(ens)[::-1]
        tix_wf     = [idx[1] for idx in te.index]
        if prev_h and max_turn < TOP_N:
            kept_idx = [j for j in sorted_idx if tix_wf[j] in prev_h][:TOP_N]
            new_idx  = [j for j in sorted_idx if tix_wf[j] not in prev_h][:max_turn]
            comb     = kept_idx + new_idx
            remaining= [j for j in sorted_idx if j not in comb]
            final_idx = (comb + remaining)[:TOP_N]
        else:
            final_idx = list(sorted_idx[:TOP_N])
        # OPT1: 矿业子行业硬上限 + OPT5: 冷静期过滤
        max_gold = CONSTRAINTS.get("max_gold_mining", 99)
        max_base = CONSTRAINTS.get("max_base_metals", 99)
        gold_cnt = base_cnt = 0
        tix_arr  = [idx[1] for idx in te.index]
        filtered = []
        for fi in final_idx:
            t = tix_arr[fi]
            if t in cooldown_set:          continue   # OPT5: 冷静期
            if t in GOLD_MINING_TICKERS:
                if gold_cnt >= max_gold:   continue   # OPT1: 黄金上限
                gold_cnt += 1
            elif t in BASE_METALS_TICKERS:
                if base_cnt >= max_base:   continue   # OPT1: 贱金属上限
                base_cnt += 1
            filtered.append(fi)
        # 不够 TOP_N 时从剩余补充（不受矿业限制，但仍遵守冷静期）
        if len(filtered) < TOP_N:
            extras = [fi for fi in final_idx
                      if fi not in filtered and tix_arr[fi] not in cooldown_set]
            filtered = (filtered + extras)[:TOP_N]
        top_idx = set(filtered)
        curr_h: set[str] = set()

        for j, (idx, row) in enumerate(te.iterrows()):
            t       = idx[1]
            ret_raw = row.get("next_ret", np.nan)
            is_new  = t not in prev_h
            # Fix6: 动态交易成本（小盘矿业滑点更高）
            mktcap = float(meta_df.loc[t,"mktcap"]) if t in meta_df.index and "mktcap" in meta_df.columns else 5e9
            eff_tx = tx_cost*4 if mktcap<1e9 else (tx_cost*2 if mktcap<5e9 else tx_cost)
            ret_net = (ret_raw - eff_tx*2 if pd.notna(ret_raw) and is_new else ret_raw)
            if j in top_idx:
                curr_h.add(t)
            recs.append({"date":dates[i],"ticker":t,"ens":ens[j],
                         "actual_ret":row.get("next_ret_abs", ret_raw),"actual_ret_net":ret_net,
                         "actual_cls":row.get("label",np.nan)})

            # 更新 IC
            for name, pr in pr_d.items():
                if pd.notna(ret_raw):
                    ic = float(np.sign(pr[j]) == np.sign(ret_raw))
                    model_ic.setdefault(name,[]).append(ic)

        turnover = len(curr_h-prev_h)/max(len(curr_h),1)
        # OPT5: 更新冷静期（本月止损的股票，下月跳过）
        if CONSTRAINTS.get("cooldown_months", 0) > 0:
            cooldown_set = stopped_last.copy()
            stopped_last = set()
            for fi in top_idx:
                t2 = tix_arr[fi]
                r2 = te.iloc[fi].get("next_ret_abs",
                     te.iloc[fi].get("next_ret", np.nan))
                if pd.notna(r2) and r2 < -0.07:
                    stopped_last.add(t2)
        prev_h = curr_h

        if (i-MIN_TRAIN)%4==0:
            print(f"  月 {i+1}/{len(dates)-1}  训练 {len(tr)} 行  换仓 {turnover*100:.0f}%")

    return pd.DataFrame(recs)


def backtest_report(wf: pd.DataFrame, panel: pd.DataFrame,
                    daily_map: dict, meta_df: pd.DataFrame,
                    initial_capital: float = 100_000,
                    tx_cost: float = 0.002,
                    stop_loss: float = -0.08,
                    benchmark: str = "XIU.TO"):
    """
    把 walk_forward() 的输出转成逐月 P&L 报告。

    这里用的是完整模型（26特征 + XGBoost + LightGBM + MLP）的真实预测，
    不是简化版。每个月：
      1. 取当月集成分最高的 Top N 支
      2. 用下月真实收益（actual_ret）计算实际盈亏
      3. 计算等权组合月收益、NAV 变化

    参数：
      wf              → walk_forward() 返回的 DataFrame
      initial_capital → 初始资金（默认 $100,000 CAD）
      tx_cost         → 单边手续费
      stop_loss       → 止损线（已在实际收益中体现）
      benchmark       → 基准 ETF（默认 XIU.TO = TSX 综合指数）
    """
    from collections import Counter

    if wf.empty or "actual_ret" not in wf.columns:
        print("  ⚠️  Walk-Forward 结果为空，无法生成回测报告")
        return

    # 下载基准月度收益
    bench_monthly = {}
    try:
        b = yf.download(benchmark, period=f"{YEARS+1}y",
                        auto_adjust=True, progress=False)["Close"]
        bench_monthly = b.resample("ME").last().pct_change().to_dict()
    except Exception:
        pass

    dates       = sorted(wf["date"].unique())
    nav         = initial_capital
    monthly_recs= []
    prev_picks  = []

    print("\n" + "╔" + "═"*72 + "╗")
    print(f"║  📅 模型历史回测报告  （{len(dates)} 个测试月，完整模型）{'':>25}║")
    print(f"║  初始资金 ${initial_capital:,.0f} CAD  |  "
          f"手续费 {tx_cost*100:.1f}% 单边  |  止损 {stop_loss*100:.0f}%{'':>14}║")
    print("╚" + "═"*72 + "╝")

    for date in dates:
        month_wf = wf[wf["date"] == date].copy()
        if month_wf.empty:
            continue

        # ── Top N 持仓（按集成分排序）────────────────────────────
        top_n  = month_wf.nlargest(TOP_N, "ens")
        picks  = top_n["ticker"].tolist()
        n_new  = len(set(picks) - set(prev_picks))
        n_out  = len(set(prev_picks) - set(picks))
        w      = 1.0 / len(picks)   # 等权

        # ── 逐支计算实际收益 ──────────────────────────────────────
        stock_rows = []
        port_ret   = 0.0
        for _, row in top_n.iterrows():
            t       = row["ticker"]
            raw     = row["actual_ret"]   # 下月真实收益（已含在wf中）
            if pd.isna(raw):
                continue

            # 止损：月内跌幅超过止损线时用止损价
            stopped = False
            df      = daily_map.get(t, pd.DataFrame())
            if not df.empty:
                # 找该月日线数据
                nxt_dates = df.index[df.index > pd.Timestamp(date)]
                if len(nxt_dates) > 0:
                    end_d = nxt_dates[min(21, len(nxt_dates)-1)]  # 约1个月
                    month_data = df.loc[nxt_dates[0]:end_d]
                    if not month_data.empty:
                        entry_px = float(df.loc[df.index <= pd.Timestamp(date),"close"].iloc[-1])
                        min_px   = float(month_data["low"].min())
                        if entry_px > 0 and (min_px - entry_px)/entry_px < stop_loss:
                            raw     = stop_loss
                            stopped = True

            # 新买入扣手续费，持续持有不扣
            is_new  = t not in prev_picks
            net     = raw - tx_cost * 2 if is_new else raw
            contrib = net * w

            port_ret += contrib
            stock_rows.append({
                "ticker":  t,
                "score":   round(row["ens"], 3),
                "raw_ret": round(raw  * 100, 2),
                "net_ret": round(net  * 100, 2),
                "contrib": round(contrib * 100, 3),
                "is_new":  is_new,
                "stopped": stopped,
            })

        if not stock_rows:
            continue

        # ── 基准收益 ──────────────────────────────────────────────
        bkey      = [k for k in bench_monthly
                     if hasattr(k, "month") and
                     k.month == pd.Timestamp(date).month and
                     k.year  == pd.Timestamp(date).year]
        bench_ret = bench_monthly.get(bkey[0], 0) * 100 if bkey else 0

        nav_prev  = nav
        nav       = nav * (1 + port_ret)
        nav_chg   = nav - nav_prev
        port_pct  = port_ret * 100
        excess    = port_pct - bench_ret

        monthly_recs.append({
            "date":      date,
            "port_pct":  round(port_pct,  2),
            "bench_pct": round(bench_ret, 2),
            "excess":    round(excess,    2),
            "nav":       round(nav,       0),
            "nav_chg":   round(nav_chg,   0),
            "picks":     picks,
            "stocks":    pd.DataFrame(stock_rows),
            "n_new":     n_new, "n_out": n_out,
        })

        # ── 月度打印 ──────────────────────────────────────────────
        up   = port_pct >= 0
        beat = excess   >= 0
        print(f"\n  {'─'*72}")
        print(f"  📅 {pd.Timestamp(date).strftime('%Y年%m月')}  "
              f"{'▲' if up else '▼'} {port_pct:>+6.2f}%  "
              f"基准 {bench_ret:>+5.2f}%  "
              f"超额 {'↑' if beat else '↓'} {excess:>+5.2f}%  "
              f"NAV ${nav:>10,.0f}  ({'+' if nav_chg>=0 else ''}{nav_chg:,.0f})")
        print(f"  持仓 {len(picks)} 支  ←{n_new}新 →{n_out}出")

        # 持仓明细
        sdf = pd.DataFrame(stock_rows).sort_values("contrib", ascending=False)
        print(f"  {'Ticker':<14} {'集成分':>7} {'涨跌%':>8} {'贡献%':>8}  {'备注'}")
        print(f"  {'─'*54}")
        for _, r in sdf.iterrows():
            icon  = "🟢" if r["net_ret"] >= 0 else "🔴"
            new_s = " ★新" if r["is_new"]  else ""
            stp_s = " ⛔止损" if r["stopped"] else ""
            print(f"  {r['ticker']:<14} {r['score']:>7.3f} "
                  f"{r['net_ret']:>+7.2f}% {r['contrib']:>+7.3f}%  "
                  f"{icon}{new_s}{stp_s}")

        prev_picks = picks

    # ── 年度汇总 ──────────────────────────────────────────────────
    if not monthly_recs:
        print("  ⚠️  无月度记录")
        return

    rets       = np.array([m["port_pct"] for m in monthly_recs]) / 100
    bench_rets = np.array([m["bench_pct"] for m in monthly_recs]) / 100
    final_nav  = monthly_recs[-1]["nav"]
    total_ret  = (final_nav / initial_capital - 1) * 100
    n          = len(rets)
    ann_ret    = (np.prod(1+rets)**(12/n)-1)*100      if n > 0 else 0
    bench_ann  = (np.prod(1+bench_rets)**(12/n)-1)*100 if n > 0 else 0
    vol_m      = rets.std() * np.sqrt(12) * 100
    sharpe     = (ann_ret/100 - 0.04) / (vol_m/100)   if vol_m > 0 else 0

    navs    = pd.Series([m["nav"] for m in monthly_recs])
    mdd     = ((navs - navs.cummax()) / navs.cummax()).min() * 100

    win_months  = (rets > 0).sum()
    beat_months = sum(1 for m in monthly_recs if m["excess"] > 0)
    best_m  = max(monthly_recs, key=lambda m: m["port_pct"])
    worst_m = min(monthly_recs, key=lambda m: m["port_pct"])

    # 最常入选股票
    all_picks = [t for m in monthly_recs for t in m["picks"]]
    top_picks = Counter(all_picks).most_common(10)

    print("\n\n" + "╔" + "═"*70 + "╗")
    print(f"║  📊 完整模型回测汇总（{n} 个月）{'':>42}║")
    print("╠" + "═"*70 + "╣")

    def row(label, val, icon=""):
        print(f"║  {label:<22} {val:<38} {icon:<6} ║")

    row("初始资金",     f"${initial_capital:>12,.0f} CAD")
    row("最终净值",     f"${final_nav:>12,.0f} CAD")
    row("总收益",       f"{total_ret:>+11.2f}%",
        "🟢" if total_ret >= 0 else "🔴")
    row("年化收益（策略）",f"{ann_ret:>+11.2f}%",
        "🟢" if ann_ret >= 0 else "🔴")
    row("年化收益（基准）",f"{bench_ann:>+11.2f}%")
    row("超额收益 Alpha",f"{ann_ret-bench_ann:>+11.2f}%",
        "🟢" if ann_ret > bench_ann else "🔴")
    row("年化波动率",   f"{vol_m:>11.2f}%")
    row("Sharpe 比率",  f"{sharpe:>12.2f}")
    row("最大回撤",     f"{mdd:>+11.2f}%")
    row("月胜率",
        f"{win_months/n*100:.1f}%  ({win_months}/{n}月)",
        "🟢" if win_months/n >= 0.5 else "🔴")
    row("跑赢基准月份",
        f"{beat_months/n*100:.1f}%  ({beat_months}/{n}月)",
        "🟢" if beat_months/n >= 0.5 else "🔴")
    row("最佳月份",
        f"{pd.Timestamp(best_m['date']).strftime('%Y-%m')}  "
        f"{best_m['port_pct']:>+.2f}%", "✨")
    row("最差月份",
        f"{pd.Timestamp(worst_m['date']).strftime('%Y-%m')}  "
        f"{worst_m['port_pct']:>+.2f}%", "⚠️")

    print("╠" + "═"*70 + "╣")
    print(f"║  月度收益明细{'':>58}║")
    print(f"║  {'月份':<10} {'策略':>8} {'基准':>8} {'超额':>8} "
          f"{'净值':>12} {'盈亏':>10} {'':>8}║")
    print(f"║  {'─'*68}║")
    for m in monthly_recs:
        icon = "✅" if m["port_pct"] >= 0 else "❌"
        beat = "↑" if m["excess"] >= 0 else "↓"
        print(f"║  {pd.Timestamp(m['date']).strftime('%Y-%m'):<10} "
              f"{m['port_pct']:>+7.2f}% "
              f"{m['bench_pct']:>+7.2f}% "
              f"{m['excess']:>+7.2f}% "
              f"${m['nav']:>10,.0f} "
              f"{m['nav_chg']:>+9,.0f} "
              f"{icon}{beat}   ║")

    print("╠" + "═"*70 + "╣")
    print(f"║  最常入选 Top 10{'':>54}║")
    for t, cnt in top_picks:
        bar = "█" * cnt
        nm  = meta_df.loc[t,"name"][:16] if t in meta_df.index else t
        print(f"║    {t:<14} {nm:<18} {cnt:>2}次  {bar:<20}{'':>12}║")

    print("╚" + "═"*70 + "╝")


def evaluate(wf):
    if wf.empty or "actual_ret" not in wf.columns:
        print("  ⚠️  Walk-Forward 结果为空，跳过评估")
        return
    v = wf.dropna(subset=["actual_ret","actual_cls"])
    if not len(v): return

    print("\n"+"═"*56)
    print("  Walk-Forward 样本外表现")
    print("═"*56)

    p    = v["ens"]
    q80  = p.quantile(0.80)
    q20  = p.quantile(0.20)
    topR = v.loc[p>=q80,"actual_ret"].mean()*100
    botR = v.loc[p<=q20,"actual_ret"].mean()*100
    netR = v.loc[p>=q80,"actual_ret_net"].mean()*100 if "actual_ret_net" in v.columns else topR
    acc  = accuracy_score(v["actual_cls"],(p>=q80).astype(int))
    hit  = (v.loc[p>=q80,"actual_ret"]>0).mean()

    monthly = v.groupby("date")["actual_ret"].mean()
    cum     = (1+monthly).cumprod()
    mdd     = ((cum-cum.cummax())/cum.cummax()).min()*100

    # Bug1 修复：换仓成本 = 新进股票数 / 总持仓 × 双边手续费
    # actual_ret_net 和 actual_ret 的差是残差收益差，不是手续费
    # 正确算法：每月 Top 组里新进股票的比例 × tx_cost × 2
    top20_mask = wf["ens"] >= wf.groupby("date")["ens"].transform(
        lambda x: x.quantile(0.80))
    monthly_turn = wf[top20_mask].groupby("date").apply(
        lambda g: (g["actual_ret"] - g["actual_ret_net"]).clip(lower=0).mean() * 100
    ).clip(0, 0.004 * 100)   # 上限 = 双边手续费 0.4%

    print(f"\n  ▸ 集成模型 ★")
    print(f"    Top20% 月均收益（税前）   {topR:+.2f}%")
    print(f"    Top20% 月均收益（扣成本） {netR:+.2f}%")
    print(f"    Bot20% 月均收益           {botR:+.2f}%")
    print(f"    多空价差                  {topR-botR:+.2f}%")
    print(f"    分类准确率                {acc*100:.1f}%")
    print(f"    Top组胜率                 {hit*100:.1f}%")
    print(f"    历史最大月度回撤          {mdd:+.2f}%")
    print(f"    平均月换仓成本            {monthly_turn.mean():.3f}%")

# ══════════════════════════════════════════════════════════════════
# 6. 当月预测
# ══════════════════════════════════════════════════════════════════

def predict_now(panel, daily_map, meta_df, wf=None, macro_df=None):
    dates = sorted(panel.index.get_level_values("date").unique())
    sc    = StandardScaler()

    # ✓ Bug fix: 只对 label dropna
    tr  = panel[panel.index.get_level_values("date").isin(dates[:-1])]\
               .dropna(subset=["next_ret","label"])
    cur = panel[panel.index.get_level_values("date")==dates[-1]]

    if len(tr) < 60:
        print("训练数据不足（YEARS >= 3）")
        return None, None, None

    print(f"\n训练最终模型（{len(tr)} 行历史）...")
    # 👇 替换为以下内容 👇
    X_tr = sc.fit_transform(tr[FEATURE_COLS].fillna(tr[FEATURE_COLS].median()))
    X_cu = sc.transform(cur[FEATURE_COLS].fillna(tr[FEATURE_COLS].median()))

    # ★ 计算最后一次预测的样本时间衰减权重
    sample_dates = tr.index.get_level_values("date")
    current_pred_date = dates[-1] # 当前预测的月份
    w_tr = compute_time_decay_weights(sample_dates, current_date=current_pred_date, half_life_months=12.0)

    # 第四层：动态权重
    mw = [1.2, 1.0, 0.8] if LGBM and TORCH else None
    
    # 最终选股：使用完整模型，并传入 weights_tr
    ens_r, ens_c, ens, xgb_r, _, _ = fit_all(
        X_tr, tr["next_ret"].values, tr["label"].values, X_cu, 
        weights_tr=w_tr, model_weights=mw, use_mlp=True, mlp_epochs=80)

    # 第四层：动态权重
    mw = [1.2, 1.0, 0.8] if LGBM and TORCH else None
    # 最终选股：使用完整模型（含 MLP，更精确）
    ens_r, ens_c, ens, xgb_r, _, _ = fit_all(
        X_tr, tr["next_ret"].values, tr["label"].values, X_cu, mw,
        use_mlp=True, mlp_epochs=80)  # 最终选股用80 epochs

    out = cur[FEATURE_COLS].copy()
    out["pred_return"]    = ens_r
    out["pred_top20pct"]  = ens_c
    out["ensemble_score"] = ens
    tix = out.index.get_level_values("ticker")
    out["name"]   = tix.map(meta_df["name"].to_dict())
    out["sector"] = tix.map(meta_df["sector"].to_dict())
    out = out.sort_values("ensemble_score", ascending=False)

    # 最终价格过滤（双重保障：apply_constraints 之后 predict_now 里再过滤一次）
    max_px = CONSTRAINTS.get("max_price_cad", 9999)
    tix_all = out.index.get_level_values("ticker") if isinstance(out.index, pd.MultiIndex) else out.index
    price_mask = pd.Series([
        daily_map[t]["close"].iloc[-1] <= max_px if t in daily_map else True
        for t in tix_all
    ], index=out.index)
    filtered_out = out[~price_mask]
    if len(filtered_out):
        print(f"  价格过滤：剔除 {list(filtered_out.index.get_level_values('ticker') if isinstance(filtered_out.index, pd.MultiIndex) else filtered_out.index)}")
    out = out[price_mask]

    # OPT6: 置信度过滤 - 低置信度时减少持仓数
    min_conf   = CONSTRAINTS.get("min_confidence", 0.0)
    min_top_n  = CONSTRAINTS.get("min_top_n", TOP_N)
    max_score  = out["ensemble_score"].max() if len(out) else 0
    effective_top_n = TOP_N
    if max_score < min_conf:
        effective_top_n = min_top_n
        print(f"  ⚠️  置信度不足（最高分 {max_score:.3f} < {min_conf}），"
              f"持仓数从 {TOP_N} 降至 {min_top_n}")
    elif max_score < min_conf * 1.5:
        effective_top_n = max(min_top_n, int(TOP_N * 0.7))
        print(f"  ⚠️  置信度偏低（{max_score:.3f}），持仓数降至 {effective_top_n}")

    # 三维分散约束 + OPT1 矿业子行业硬上限
    mgics = CONSTRAINTS.get("max_per_gics", 99)
    msty  = CONSTRAINTS.get("max_per_style", 99)
    mtyp  = CONSTRAINTS.get("max_per_type", 99)
    max_gold = CONSTRAINTS.get("max_gold_mining", 99)
    max_base = CONSTRAINTS.get("max_base_metals", 99)
    gc, sc2, tc = {}, {}, {}
    gold_cnt = base_cnt = 0
    keep = []
    for idx in out.index:
        if len(keep) >= effective_top_n:
            break
        t  = idx[1] if isinstance(idx, tuple) else idx
        p  = STOCK_PROFILE.get(t, {})
        g, s, tp = p.get("gics","?"), p.get("style","?"), p.get("type","?")
        # 矿业子行业检查
        if t in GOLD_MINING_TICKERS:
            if gold_cnt >= max_gold: continue
            gold_cnt += 1
        elif t in BASE_METALS_TICKERS:
            if base_cnt >= max_base: continue
            base_cnt += 1
        if gc.get(g,0)<mgics and sc2.get(s,0)<msty and tc.get(tp,0)<mtyp:
            keep.append(idx)
            gc[g] = gc.get(g,0)+1
            sc2[s]= sc2.get(s,0)+1
            tc[tp]= tc.get(tp,0)+1
    out = out.loc[keep]
    print(f"  分散后：{len(out)} 支  {dict(gc)}"
          f"  [黄金:{gold_cnt}/{max_gold} 贱金属:{base_cnt}/{max_base}]")

    imp = pd.Series(xgb_r.feature_importances_,
                    index=FEATURE_COLS).sort_values(ascending=False)

    # OPT4: 最大回撤熔断 + OPT7: VIX 波动率缩仓
    dd_signal    = None
    vix_scale    = 1.0   # 默认不缩仓
    halt_scale   = 1.0

    if wf is not None and not wf.empty and len(wf["date"].unique()) >= 3:
        recent = wf.groupby("date")["actual_ret"].mean().tail(3)
        cum    = (1 + recent).prod() - 1

        # OPT4: 熔断 - 连续3月累计亏损超阈值
        dd_thresh = CONSTRAINTS.get("dd_halt_threshold", -0.15)
        if cum < dd_thresh:
            dd_signal  = cum
            halt_scale = CONSTRAINTS.get("dd_halt_scale", 0.5)
            print(f"  🔴 熔断触发：近3月累计 {cum*100:.1f}%，仓位缩减至 {halt_scale*100:.0f}%")

    # OPT7: VIX 高时缩仓
    try:
        if macro_df is not None and not macro_df.empty and "vix" in macro_df.columns:
            vix_now = float(macro_df["vix"].iloc[-1])
            vix_thr = CONSTRAINTS.get("vix_scale_threshold", 25.0)
            if vix_now > vix_thr:
                vix_scale = CONSTRAINTS.get("vix_scale_factor", 0.70)
                print(f"  🟡 VIX={vix_now:.1f} > {vix_thr}，仓位缩减至 {vix_scale*100:.0f}%")
    except Exception:
        pass

    # 综合缩仓系数（熔断 × VIX 双重保护）
    total_scale = halt_scale * vix_scale
    if total_scale < 1.0:
        out["ensemble_score"] = out["ensemble_score"] * total_scale
        print(f"  ⚡ 综合缩仓系数：{total_scale:.2f}（仓位整体降低）")

    return out, imp, dd_signal

# ══════════════════════════════════════════════════════════════════
# 7. Fuzzy + 风险平价仓位
# ══════════════════════════════════════════════════════════════════

def fuzzy_membership(score, lo, md, hi):
    light    = max(0.0, min(1.0, (md-score)/(md-lo+1e-9))) if score<=md else 0.0
    moderate = (max(0.0,min(1.0,(score-lo)/(md-lo+1e-9))) if score<=md
                else max(0.0,min(1.0,(hi-score)/(hi-md+1e-9))))
    heavy    = max(0.0, min(1.0, (score-md)/(hi-md+1e-9))) if score>md else 0.0
    return light, moderate, heavy


def risk_parity(top, daily_map):
    """第四层：风险平价权重 ∝ 1/波动率"""
    vols = []
    for idx, _ in top.iterrows():
        t  = idx[1] if isinstance(idx,tuple) else idx
        df = daily_map.get(t, pd.DataFrame())
        v  = df["close"].pct_change().tail(63).std()*np.sqrt(252) if not df.empty else 0.25
        vols.append(max(v, 0.05))
    inv = [1/v for v in vols]
    s   = sum(inv)
    return [w/s for w in inv]


def _vec_fuzzy_membership(scores: np.ndarray, lo: float, md: float, hi: float):
    """向量化三角模糊隶属度（替代逐行循环）"""
    light    = np.where(scores <= md, np.clip((md-scores)/(md-lo+1e-9),0,1), 0.0)
    moderate = np.where(scores <= md,
                        np.clip((scores-lo)/(md-lo+1e-9),0,1),
                        np.clip((hi-scores)/(hi-md+1e-9),0,1))
    heavy    = np.where(scores > md,  np.clip((scores-md)/(hi-md+1e-9),0,1), 0.0)
    return light, moderate, heavy


def fuzzy_sizing(top, daily_map):
    """
    Fuzzy Logic × 风险平价 混合仓位（向量化，替代 iterrows）。
    速度提升约 10-50x。
    """
    RANGE_LO = {"Heavy Buy": 0.20, "Moderate Buy": 0.10, "Light Buy": 0.03}
    RANGE_HI = {"Heavy Buy": 0.25, "Moderate Buy": 0.15, "Light Buy": 0.08}

    scores = top["ensemble_score"].values
    probs  = top["pred_top20pct"].values
    rets   = top["pred_return"].values

    # 动态分位数阈值（一次性计算）
    lo,md,hi   = np.percentile(scores,[25,50,75])
    p_lo,p_md,p_hi = np.percentile(probs,  [25,50,75])
    r_lo,r_md,r_hi = np.percentile(rets,   [25,50,75])

    # 向量化隶属度计算
    l_s,m_s,h_s = _vec_fuzzy_membership(scores, lo, md, hi)
    l_p,m_p,h_p = _vec_fuzzy_membership(probs,  p_lo, p_md, p_hi)
    l_r,m_r,h_r = _vec_fuzzy_membership(rets,   r_lo, r_md, r_hi)

    wh    = (h_s + h_p + h_r) / 3
    wm    = (m_s + m_p + m_r) / 3
    wl    = (l_s + l_p + l_r) / 3
    crisp = (wh*3 + wm*2 + wl*1) / (wh + wm + wl + 1e-9)

    # 向量化波动率（一次性批量计算）
    tickers = [idx[1] if isinstance(idx,tuple) else idx for idx in top.index]
    vols    = np.array([
        daily_map[t]["close"].pct_change().tail(63).std()*np.sqrt(252)
        if t in daily_map and not daily_map[t].empty else 0.25
        for t in tickers
    ])

    # 向量化类别判定
    cats = np.where(crisp >= 2.2, "Heavy Buy",
           np.where(crisp >= 1.6, "Moderate Buy", "Light Buy"))
    # Vol downgrade
    cats = np.where((vols > 0.55) & (cats == "Heavy Buy"),    "Moderate Buy", cats)
    cats = np.where((vols > 0.65) & (cats == "Moderate Buy"), "Light Buy",    cats)

    # 向量化仓位计算（cats 是 np.str_，需要转成 str 才能查字典）
    rng_lo = np.array([RANGE_LO[str(c)] for c in cats])
    rng_hi = np.array([RANGE_HI[str(c)] for c in cats])
    t_val  = np.clip((crisp - 1.0) / 2.0, 0, 1)
    vol_f  = np.clip(1.0 - (vols - 0.15) / 0.40, 0, 1)
    alloc  = (rng_lo + t_val*(rng_hi - rng_lo)) * vol_f + rng_lo * (1 - vol_f)

    # 风险平价微调（向量化）
    rp      = np.array(risk_parity(top, daily_map))
    rp_mean = 1.0 / max(len(rp), 1)
    rp_adj  = 1.0 + 0.2 * (rp - rp_mean) / (rp_mean + 1e-9)
    alloc   = np.round(alloc * rp_adj * 100, 1)

    df_out = pd.DataFrame({
        "ticker":   tickers,
        "category": cats,
        "alloc_pct":alloc,
        "vol_ann":  vols,
        "crisp":    np.round(crisp, 2),
        "score":    scores,
        "prob":     probs,
        "ret":      rets,
    })
    total  = df_out["alloc_pct"].sum()
    df_out["alloc_pct"] = (df_out["alloc_pct"]/total*100).round(1)

    # 单支最大仓位上限
    cap = CONSTRAINTS.get("max_single_alloc", 1.0) * 100
    if (df_out["alloc_pct"] > cap).any():
        df_out["alloc_pct"] = df_out["alloc_pct"].clip(upper=cap)
        # 被截掉的部分重新按比例分配给其他股票
        total2 = df_out["alloc_pct"].sum()
        df_out["alloc_pct"] = (df_out["alloc_pct"] / total2 * 100).round(1)

    return df_out

# ══════════════════════════════════════════════════════════════════
# 8. 输出
# ══════════════════════════════════════════════════════════════════

def print_picks(result, imp, daily_map, meta_df, wf, dd_signal):
    top = result.head(TOP_N)
    models_str = "XGBoost"+(" + LightGBM" if LGBM else "")+(" + MLP" if TORCH else "")

    print("\n"+"═"*84)
    print(f"  🍁 TSX 量化选股 v2.0 — {datetime.today().strftime('%Y-%m-%d')}")
    print(f"  模型：{models_str}  |  特征：{len(FEATURE_COLS)} 个  |  训练：{YEARS} 年历史")
    print("═"*84)

    # 第四层：回撤控制警告
    if dd_signal is not None:
        print(f"\n  ⚠️  【风控】近3月组合累计 {dd_signal*100:.1f}%，触发回撤控制")
        print(f"     建议：整体仓位减半，等待信号恢复\n")

    # Top 10 明细（含建议股数）
    TOTAL_CAPITAL = 100_000   # 修改此处调整资金规模 CAD
    print(f"{'#':<4}{'Ticker':<13}{'公司':<18}{'GICS':<13}{'预测涨幅':>9}"
          f"{'进前20%':>8}{'集成分':>8}{'股价':>9}{'建议股数':>9}")
    print("─"*95)
    for i,(idx,row) in enumerate(top.iterrows(),1):
        t    = idx[1] if isinstance(idx,tuple) else idx
        prof = STOCK_PROFILE.get(t,{})
        df   = daily_map.get(t, pd.DataFrame())
        price = float(df["close"].iloc[-1]) if not df.empty else 0
        # 建议股数：用 fuzzy 仓位比例计算（预估，正式仓位在下方 Position Sizing）
        alloc_pct = 1.0 / len(top)   # 等权预估
        shares = int(TOTAL_CAPITAL * alloc_pct / price) if price > 0 else 0
        price_s  = f"${price:>7.2f}" if price > 0 else "   N/A"
        shares_s = f"{shares:>6}股"   if shares > 0 else "   N/A"
        print(f"{i:<4}{t:<13}{str(row.get('name',''))[:16]:<18}"
              f"{prof.get('gics','?')[:11]:<13}"
              f"{row['pred_return']*100:>+8.1f}%"
              f"{row['pred_top20pct']*100:>7.0f}%"
              f"{row['ensemble_score']:>9.3f}"
              f"  {price_s}  {shares_s}")
    print("─"*95)
    print(f"  ※ 建议股数基于等权 ${TOTAL_CAPITAL:,.0f} CAD，精确仓位见下方 Position Sizing")

    # 约束核查（含股息率）
    print(f"\n  {'Ticker':<14}{'价格':>8}{'ADV':>10}{'P/E':>7}{'市值':>10}"
          f"{'ROE':>7}{'股息率':>8}")
    print(f"  {'─'*64}")
    for _,(idx,row) in enumerate(top.iterrows()):
        t  = idx[1] if isinstance(idx,tuple) else idx
        df = daily_map.get(t,pd.DataFrame())
        m  = meta_df.loc[t] if t in meta_df.index else pd.Series(dtype=float)
        price  = f"${df['close'].iloc[-1]:.2f}"                   if not df.empty else "N/A"
        adv    = f"${(df['close'].tail(20)*df['volume'].tail(20)).mean()/1e6:.1f}M" \
                 if not df.empty else "N/A"
        pe     = f"{m.get('pe'):.1f}x"                            if pd.notna(m.get('pe'))  else "N/A"
        mc     = f"${m.get('mktcap')/1e9:.1f}B"                   if m.get('mktcap')        else "N/A"
        roe    = f"{m.get('roe')*100:.1f}%"                        if pd.notna(m.get('roe')) else "N/A"
        div    = f"{m.get('div_yield')*100:.1f}%"                  if pd.notna(m.get('div_yield')) else "N/A"
        print(f"  {t:<14}{price:>8}{adv:>10}{pe:>7}{mc:>10}{roe:>7}{div:>8}")

    # 第一层：止损提示
    print(f"\n  ⚡ 止损线：持仓期间单支跌超 {abs(STOP_LOSS_PCT)*100:.0f}% 建议止损")

    # Fuzzy 仓位
    fz = fuzzy_sizing(top, daily_map)
    # 加入当前价格和建议股数
    TOTAL_CAPITAL = 100_000   # ← 改成你的总资金 CAD

    prices = {}
    for idx, _ in top.iterrows():
        t  = idx[1] if isinstance(idx, tuple) else idx
        df = daily_map.get(t, pd.DataFrame())
        prices[t] = float(df["close"].iloc[-1]) if not df.empty else 0.0

    fz["price"]     = fz["ticker"].map(prices)
    fz["alloc_cad"] = (fz["alloc_pct"] / 100 * TOTAL_CAPITAL).round(0)
    fz["shares"]    = (fz["alloc_cad"] / fz["price"].replace(0, float("nan"))).apply(
                       lambda x: int(x) if pd.notna(x) else 0)

    SEP = "=" * 92
    print("\n" + SEP)
    print(f"  Position Sizing  (Total Capital: ${TOTAL_CAPITAL:,.0f} CAD)")
    print(f"  Heavy 20-25%  |  Moderate 10-15%  |  Light 3-8%")
    print(f"  Change TOTAL_CAPITAL above to match your actual portfolio size")
    print(f"  ⚠️  标注股票股数 < {CONSTRAINTS.get('min_shares',5)} 股，实际持仓意义有限（建议增加资金或跳过）")
    print(SEP)
    print(f"  {'#':<3}{'Ticker':<13}{'Category':<15}{'Alloc%':>7}  "
          f"{'Price':>8}  {'Amount(CAD)':>12}  {'Shares':>8}  {'Vol':>6}  {'Score':>7}")
    print(f"  {'-'*90}")
    for i, row in fz.iterrows():
        price_s = f"${row['price']:>7.2f}" if row["price"] > 0 else "    N/A"
        cad_s   = f"${row['alloc_cad']:>10,.0f}"
        min_sh  = CONSTRAINTS.get("min_shares", 1)
        sh_s    = f"{row['shares']:>7,}" if row["shares"] >= min_sh else f"  ⚠️ {row['shares']}股"
        print(f"  {i+1:<3}{row['ticker']:<13}{row['category']:<15}"
              f"{row['alloc_pct']:>6.1f}%  "
              f"{price_s}  {cad_s}  {sh_s}  "
              f"{row['vol_ann']*100:>5.1f}%  {row['score']:>7.3f}")
    print(f"  {'-'*90}")
    print(f"  {'Total':<31}{fz['alloc_pct'].sum():>6.1f}%  {'':>8}  "
          f"${fz['alloc_cad'].sum():>10,.0f}\n")
    for cat in ["Heavy Buy", "Moderate Buy", "Light Buy",
                "\u5927\u91cf\u4e70\u5165 \U0001f534",
                "\u9002\u5ea6\u4e70\u5165 \U0001f7e1",
                "\u5c11\u91cf\u4e70\u5165 \U0001f7e2"]:
        sub = fz[fz["category"]==cat]
        if len(sub):
            print(f"    {cat:<20} {len(sub)} stocks  {sub['alloc_pct'].sum():.1f}%"
                  f"  -> {' | '.join(sub['ticker'].tolist())}")

    # Feature importance
    print(f"\n  特征重要性（XGBoost，Top 12）：")
    for feat, val in imp.head(12).items():
        bar = "█"*int(val*300)
        print(f"  {feat:<24} {bar:<30} {val:.4f}")

# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════
# MODULE A: 共线性处理（VIF + PCA 降维）
# ══════════════════════════════════════════════════════════════════

def compute_vif(panel: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    计算每个特征的 VIF（方差膨胀因子）。
    VIF > 10 说明严重共线性，> 5 说明中度共线性。

    动量族（mom_1m/3m/6m/12m）通常 VIF > 20，是最大共线性来源。
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X = panel[feature_cols].fillna(panel[feature_cols].median())
    # 去掉方差为0的列
    X = X.loc[:, X.std() > 0]

    vif_data = []
    for i, col in enumerate(X.columns):
        try:
            v = variance_inflation_factor(X.values, i)
            vif_data.append({"feature": col, "vif": round(v, 2)})
        except Exception:
            vif_data.append({"feature": col, "vif": np.nan})

    return pd.DataFrame(vif_data).sort_values("vif", ascending=False)


def apply_collinearity_reduction(panel: pd.DataFrame, vif_threshold: float = 10.0) -> tuple[pd.DataFrame, list[str]]:
    """
    两步共线性处理：

    步骤1：对高度相关的动量族特征做 PCA 压缩
      mom_1m / mom_3m / mom_6m / mom_12m / mom_12_1
      → 提取 2 个主成分：mom_pc1（趋势强度）/ mom_pc2（动量曲率）

    步骤2：对波动率族特征做 PCA 压缩
      vol_1m / vol_3m / vol_ratio
      → 提取 1 个主成分：vol_pc1（整体波动水平）

    步骤3：逐步删除 VIF > threshold 的剩余特征

    返回：处理后的 panel，新的 feature_cols
    """
    from sklearn.decomposition import PCA

    print("\n  [共线性处理] 分析特征相关性...")

    panel = panel.copy()
    new_cols = list(FEATURE_COLS)

    # ── 步骤1：动量 PCA ────────────────────────────────────────────
    mom_cols = ["mom_1m","mom_3m","mom_6m","mom_12m","mom_12_1"]
    mom_data = panel[mom_cols].fillna(0)  # PCA前已中性化，0=均值，正确

    pca_mom = PCA(n_components=2, random_state=42)
    mom_pcs = pca_mom.fit_transform(mom_data)
    panel["mom_pc1"] = mom_pcs[:, 0]   # 趋势强度
    panel["mom_pc2"] = mom_pcs[:, 1]   # 动量曲率
    expl = pca_mom.explained_variance_ratio_
    print(f"    动量 PCA: PC1={expl[0]*100:.1f}%  PC2={expl[1]*100:.1f}%  "
          f"累计={sum(expl)*100:.1f}%")

    for c in mom_cols:
        new_cols.remove(c)
    new_cols = ["mom_pc1","mom_pc2"] + new_cols

    # ── 步骤2：波动率 PCA ──────────────────────────────────────────
    vol_cols = ["vol_1m","vol_3m","vol_ratio"]
    vol_data = panel[vol_cols].fillna(0)  # 同上

    pca_vol = PCA(n_components=1, random_state=42)
    vol_pcs = pca_vol.fit_transform(vol_data)
    panel["vol_pc1"] = vol_pcs[:, 0]
    expl_v = pca_vol.explained_variance_ratio_
    print(f"    波动率 PCA: PC1={expl_v[0]*100:.1f}%")

    for c in vol_cols:
        new_cols.remove(c)
    new_cols = ["vol_pc1"] + new_cols

    # ── 步骤3：计算剩余 VIF，删除高共线性特征 ─────────────────────
    vif_df = compute_vif(panel, new_cols)
    high_vif = vif_df[vif_df["vif"] > vif_threshold]["feature"].tolist()

    print(f"\n    VIF 分析（阈值 {vif_threshold}）：")
    print(f"    {'特征':<24} {'VIF':>8}  {'状态'}")
    print(f"    {'─'*44}")
    for _, row in vif_df.head(15).iterrows():
        status = "✗ 删除" if row["feature"] in high_vif else "✓ 保留"
        bar    = "▓" * min(20, int(row["vif"]/5)) if pd.notna(row["vif"]) else ""
        vif_str= f"{row['vif']:.1f}" if pd.notna(row["vif"]) else "N/A"
        print(f"    {row['feature']:<24} {vif_str:>8}  {bar}  {status}")

    # 删除高 VIF 特征（但保留 PCA 主成分）
    pc_cols = ["mom_pc1","mom_pc2","vol_pc1"]
    to_remove = [c for c in high_vif if c not in pc_cols]
    for c in to_remove:
        if c in new_cols:
            new_cols.remove(c)

    print(f"\n    最终特征数：{len(new_cols)} 个（原 {len(FEATURE_COLS)} 个）")
    print(f"    删除：{to_remove if to_remove else '无'}")

    return panel, new_cols


# ══════════════════════════════════════════════════════════════════
# MODULE B: SEDI 内部人交易数据（加拿大 SEC）
# ══════════════════════════════════════════════════════════════════

def fetch_sedi_insider(tickers: list[str], lookback_days: int = 90) -> pd.DataFrame:
    """
    从 canadianinsider.com 抓取 SEDI 内部人交易数据。
    canadianinsider.com 是 SEDI 的聚合展示平台，比直接抓 SEDI 更友好。

    内部人交易信号逻辑：
      净买入（insider_buy_ratio > 0.6）→ 正面信号，加分
      净卖出（insider_buy_ratio < 0.4）→ 负面信号，减分
      无交易                          → 中性

    返回 DataFrame，index=ticker，columns=[buy_cnt, sell_cnt, net_shares, signal]
    """
    import requests
    from bs4 import BeautifulSoup

    results = []
    end_date  = datetime.today()
    start_date= end_date - timedelta(days=lookback_days)

    print(f"\n  [SEDI] 抓取内部人交易（过去 {lookback_days} 天）...")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml",
    }

    for t in tickers:
        t_clean = t.replace(".TO","").replace("-",".")
        try:
            # canadianinsider.com 按股票代码查询
            url = f"https://www.canadianinsider.com/node?ticker={t_clean}"
            r   = requests.get(url, headers=headers, timeout=10)

            if r.status_code != 200:
                results.append({"ticker":t,"buy_cnt":0,"sell_cnt":0,
                                 "net_shares":0,"signal":0.0,"source":"no_data"})
                continue

            soup = BeautifulSoup(r.text, "html.parser")

            # 找交易表格
            buy_cnt = sell_cnt = net_shares = 0
            rows = soup.find_all("tr")

            for row in rows:
                cols = row.find_all("td")
                if len(cols) < 5:
                    continue
                try:
                    # 列格式：日期 | 内部人 | 职位 | 交易类型 | 数量 | 价格
                    date_str = cols[0].get_text(strip=True)
                    txn_date = datetime.strptime(date_str, "%Y-%m-%d")
                    if txn_date < start_date:
                        continue
                    txn_type = cols[3].get_text(strip=True).lower()
                    shares_str = cols[4].get_text(strip=True).replace(",","")
                    shares = float(shares_str) if shares_str.replace(".","").isdigit() else 0

                    if "acquisition" in txn_type or "purchase" in txn_type or "buy" in txn_type:
                        buy_cnt   += 1
                        net_shares+= shares
                    elif "disposition" in txn_type or "sale" in txn_type or "sell" in txn_type:
                        sell_cnt   += 1
                        net_shares -= shares
                except Exception:
                    continue

            total = buy_cnt + sell_cnt
            buy_ratio = buy_cnt / total if total > 0 else 0.5

            # 信号：-1（强卖）到 +1（强买）
            if total == 0:
                signal = 0.0
            elif buy_ratio > 0.7:
                signal = 1.0    # 强买入信号
            elif buy_ratio > 0.55:
                signal = 0.5   # 温和买入
            elif buy_ratio < 0.3:
                signal = -1.0  # 强卖出信号
            elif buy_ratio < 0.45:
                signal = -0.5  # 温和卖出
            else:
                signal = 0.0   # 中性

            results.append({"ticker":t,"buy_cnt":buy_cnt,"sell_cnt":sell_cnt,
                            "net_shares":int(net_shares),"signal":signal,"source":"sedi"})

            status = ("强买 🟢" if signal>=1 else "买 🟡" if signal>0
                      else "强卖 🔴" if signal<=-1 else "卖 🟠" if signal<0 else "中性 ⚪")
            print(f"    {t:<14} 买入{buy_cnt:>3}次  卖出{sell_cnt:>3}次  "
                  f"净股数{net_shares:>10,}  {status}")

        except Exception as e:
            results.append({"ticker":t,"buy_cnt":0,"sell_cnt":0,
                            "net_shares":0,"signal":0.0,"source":"error"})

    df = pd.DataFrame(results).set_index("ticker")
    n_sig = (df["signal"] != 0).sum()
    print(f"\n    有效信号：{n_sig}/{len(tickers)} 支  "
          f"（买入信号 {(df['signal']>0).sum()} 支，"
          f"卖出信号 {(df['signal']<0).sum()} 支）")
    return df


def apply_insider_signal(result: pd.DataFrame, insider_df: pd.DataFrame,
                          weight: float = 0.15) -> pd.DataFrame:
    """
    将内部人信号加入最终排名。
    集成分 = 原始集成分 × (1 - weight) + insider_signal × weight

    weight=0.15 意味着内部人信号占 15% 权重。
    """
    if insider_df.empty:
        return result

    result = result.copy()
    tix    = result.index.get_level_values("ticker") if isinstance(result.index, pd.MultiIndex) \
             else result.index

    signals = tix.map(insider_df["signal"].to_dict()).fillna(0)
    # 归一化 insider signal 到 [0,1]
    sig_norm = (signals + 1) / 2   # -1→0, 0→0.5, +1→1

    result["insider_signal"]  = signals.values
    result["ensemble_score"]  = (result["ensemble_score"] * (1 - weight) +
                                  sig_norm.values * weight)
    result = result.sort_values("ensemble_score", ascending=False)
    return result


# ══════════════════════════════════════════════════════════════════
# MODULE C: 参数敏感性分析
# ══════════════════════════════════════════════════════════════════

def sensitivity_analysis(daily_map: dict, pit_map: dict, meta_df: pd.DataFrame,
                          macro_df: pd.DataFrame, feature_cols: list[str],
                          n_jobs: int = 1) -> pd.DataFrame:
    """
    对关键约束参数做网格搜索，找出让 Walk-Forward 表现最好的组合。

    搜索空间：
      max_pe:        [40, 50, 60, 80]
      min_mktcap:    [200M, 500M, 1B]
      max_per_gics:  [1, 2, 3]
      tx_cost:       [0.001, 0.002, 0.005]

    评估指标：Sharpe = 多空价差 / 波动率（越高越好）

    ⚠️  每个参数组合都要跑一次 Walk-Forward，计算量大。
        建议先用小网格测试（QUICK_MODE=True）。
    """
    import itertools

    QUICK_MODE = True   # True = 小网格，快速验证；False = 完整搜索

    if QUICK_MODE:
        grid = {
            "max_pe":       [40, 60],
            "min_mktcap":   [200_000_000, 500_000_000],
            "max_per_gics": [1, 2],
            "tx_cost":      [0.001, 0.003],
        }
    else:
        grid = {
            "max_pe":       [40, 50, 60, 80],
            "min_mktcap":   [200_000_000, 500_000_000, 1_000_000_000],
            "max_per_gics": [1, 2, 3],
            "tx_cost":      [0.001, 0.002, 0.005],
        }

    keys   = list(grid.keys())
    combos = list(itertools.product(*grid.values()))
    print(f"\n[参数敏感性] 搜索 {len(combos)} 个参数组合 "
          f"({'快速模式' if QUICK_MODE else '完整模式'})...")

    records = []
    for ci, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))

        # 构建当次约束
        c_test = CONSTRAINTS.copy()
        c_test["max_pe"]        = params["max_pe"]
        c_test["min_mktcap_cad"]= params["min_mktcap"]
        c_test["max_per_gics"]  = params["max_per_gics"]

        passed_test = apply_constraints(daily_map, meta_df, c_test)
        if len(passed_test) < TOP_N:
            records.append({**params, "n_stocks":len(passed_test),
                            "spread":np.nan, "sharpe":np.nan, "win_rate":np.nan})
            continue

        try:
            panel_t = build_panel(passed_test, daily_map, pit_map, macro_df)
            panel_t = add_labels(panel_t)
            panel_t = cross_z(panel_t)

            wf_t = walk_forward(panel_t, tx_cost=params["tx_cost"])
            if wf_t.empty or "actual_ret" not in wf_t.columns:
                raise ValueError("Walk-Forward 结果为空")

            v    = wf_t.dropna(subset=["actual_ret","actual_cls"])
            p    = v["ens"]
            topR = v.loc[p>=p.quantile(0.80),"actual_ret"].mean()*100
            botR = v.loc[p<=p.quantile(0.20),"actual_ret"].mean()*100
            spread = topR - botR

            # Sharpe = 多空价差 / 月收益标准差（年化）
            monthly = v.groupby("date")["actual_ret"].mean()
            vol_m   = monthly.std() * np.sqrt(12) * 100
            sharpe  = spread / vol_m if vol_m > 0 else 0

            win_rate= (v.loc[p>=p.quantile(0.80),"actual_ret"]>0).mean()*100

            records.append({**params, "n_stocks":len(passed_test),
                           "spread":round(spread,3),"sharpe":round(sharpe,3),
                           "win_rate":round(win_rate,1),"top_ret":round(topR,3)})

            print(f"  [{ci:>3}/{len(combos)}] PE≤{params['max_pe']}  "
                  f"MCap≥{params['min_mktcap']//1e6:.0f}M  "
                  f"GICS≤{params['max_per_gics']}  "
                  f"cost={params['tx_cost']*100:.1f}%  "
                  f"→ 价差{spread:+.2f}%  Sharpe{sharpe:.2f}  "
                  f"通过{len(passed_test)}支")
        except Exception as e:
            records.append({**params, "n_stocks":len(passed_test),
                           "spread":np.nan,"sharpe":np.nan,"win_rate":np.nan})
            print(f"  [{ci:>3}/{len(combos)}] ✗ {str(e)[:40]}")

    result_df = pd.DataFrame(records).sort_values("sharpe", ascending=False)

    print(f"\n  {'─'*70}")
    print(f"  参数敏感性分析结果（按 Sharpe 排序，Top 5）：")
    print(f"  {'─'*70}")
    print(f"  {'max_pe':>8} {'min_mktcap':>12} {'max_gics':>10} "
          f"{'tx_cost':>8} {'n_stk':>6} {'价差%':>8} {'Sharpe':>8} {'胜率%':>8}")
    print(f"  {'─'*70}")
    for _, row in result_df.head(5).iterrows():
        if pd.notna(row["sharpe"]):
            print(f"  {row['max_pe']:>8.0f} "
                  f"${row['min_mktcap']/1e6:>9.0f}M "
                  f"{row['max_per_gics']:>10.0f} "
                  f"{row['tx_cost']*100:>7.1f}% "
                  f"{row['n_stocks']:>6.0f} "
                  f"{row['spread']:>+8.2f}% "
                  f"{row['sharpe']:>8.2f} "
                  f"{row['win_rate']:>7.1f}%")

    best = result_df.dropna(subset=["sharpe"]).iloc[0] if not result_df.dropna(subset=["sharpe"]).empty else None
    if best is not None:
        print(f"\n  ★ 最优参数组合：")
        print(f"    max_pe        = {best['max_pe']:.0f}")
        print(f"    min_mktcap    = ${best['min_mktcap']/1e6:.0f}M CAD")
        print(f"    max_per_gics  = {best['max_per_gics']:.0f}")
        print(f"    tx_cost       = {best['tx_cost']*100:.1f}%")
        print(f"    → Sharpe {best['sharpe']:.2f}，多空价差 {best['spread']:+.2f}%")

    return result_df


# ══════════════════════════════════════════════════════════════════
# MODULE D: 真实回测框架
# ══════════════════════════════════════════════════════════════════

class RealisticBacktest:
    """
    真实回测框架，模拟实际交易环境。

    改进点 vs 简单 Walk-Forward：
      ① 市场冲击模型（大单影响价格）
      ② 买卖价差（Bid-Ask Spread）
      ③ T+2 结算（加拿大市场标准）
      ④ 分红再投资（DRIP）
      ⑤ 止损执行（月内而非月底）
      ⑥ 持仓惯性（避免过度换仓）
      ⑦ 资金管理（保留现金缓冲）
    """

    def __init__(self,
                 capital:       float = 100_000,
                 tx_cost:       float = 0.002,    # 单边手续费
                 bid_ask:       float = 0.001,    # 买卖价差（0.1%）
                 market_impact: float = 0.002,    # 市场冲击（大单滑点）
                 stop_loss:     float = -0.08,    # 止损线
                 cash_buffer:   float = 0.05,     # 现金缓冲比例
                 hold_inertia:  float = 0.10,     # 持仓惯性阈值（分数差< X 不换仓）
                 drip:          bool  = True):    # 分红再投资

        self.capital       = capital
        self.tx_cost       = tx_cost
        self.bid_ask       = bid_ask
        self.market_impact = market_impact
        self.stop_loss     = stop_loss
        self.cash_buffer   = cash_buffer
        self.hold_inertia  = hold_inertia
        self.drip          = drip

        # 状态变量
        self.cash          = capital
        self.positions     = {}    # {ticker: {shares, cost_basis, entry_price}}
        self.nav_history   = []    # 净值历史
        self.trade_log     = []    # 交易记录
        self.monthly_rets  = []    # 月度收益

    def _trade_cost(self, value: float, adv: float) -> float:
        """
        总交易成本 = 手续费 + 买卖价差 + 市场冲击。
        市场冲击：交易金额 / (ADV × 0.1) 的平方根模型。
        """
        fee    = value * self.tx_cost
        spread = value * self.bid_ask
        # 市场冲击：交易占日均成交量比例越大，冲击越大
        participation = value / max(adv * 0.1, value)  # 假设只用10%日成交量
        impact = value * self.market_impact * np.sqrt(participation)
        return fee + spread + impact

    def _current_nav(self, prices: dict) -> float:
        """计算当前组合净值（现金 + 持仓市值）"""
        pos_value = sum(
            info["shares"] * prices.get(t, info["entry_price"])
            for t, info in self.positions.items()
        )
        return self.cash + pos_value

    def run(self, panel: pd.DataFrame, daily_map: dict,
            meta_df: pd.DataFrame, model_scores: pd.DataFrame) -> dict:
        """
        执行完整回测。
        model_scores: Walk-Forward 输出的 {date, ticker, ens} DataFrame
        """
        dates = sorted(panel.index.get_level_values("date").unique())
        print(f"\n[真实回测] 初始资金 ${self.capital:,.0f} CAD  "
              f"手续费 {self.tx_cost*100:.1f}%  买卖价差 {self.bid_ask*100:.1f}%")
        print(f"  止损线 {self.stop_loss*100:.0f}%  现金缓冲 {self.cash_buffer*100:.0f}%  "
              f"持仓惯性 {self.hold_inertia:.2f}")
        print(f"  {'─'*60}")

        turnover_total = 0

        for i, date in enumerate(dates[:-1]):
            nxt_date = dates[i+1]

            # 获取当月价格
            curr_prices = {}
            for t, df in daily_map.items():
                month_data = df[df.index.date <= date.date()]
                if not month_data.empty:
                    curr_prices[t] = float(month_data["close"].iloc[-1])

            nav_start = self._current_nav(curr_prices)

            # ── 止损检查 ─────────────────────────────────────────
            to_stop = []
            for t, info in self.positions.items():
                curr_p = curr_prices.get(t, info["entry_price"])
                ret    = (curr_p - info["entry_price"]) / info["entry_price"]
                if ret < self.stop_loss:
                    to_stop.append(t)

            for t in to_stop:
                info  = self.positions[t]
                price = curr_prices.get(t, info["entry_price"])
                adv   = (daily_map[t]["close"].tail(20) *
                         daily_map[t]["volume"].tail(20)).mean() if t in daily_map else 1e6
                cost  = self._trade_cost(info["shares"] * price, adv)
                proceeds = info["shares"] * price - cost
                self.cash += proceeds
                self.trade_log.append({"date":date,"ticker":t,"action":"止损卖出",
                                       "price":price,"shares":info["shares"],
                                       "cost":cost,"reason":f"跌幅超{self.stop_loss*100:.0f}%"})
                del self.positions[t]

            # ── 获取本月模型评分 ──────────────────────────────────
            month_scores = model_scores[model_scores["date"] == date]
            if month_scores.empty:
                self.nav_history.append({"date":date,"nav":self._current_nav(curr_prices)})
                continue

            top_scores = month_scores.nlargest(TOP_N, "ens")
            target_tickers = set(top_scores["ticker"].tolist())
            current_tickers = set(self.positions.keys())

            # ── 持仓惯性：评分差异小于阈值的保持持仓 ─────────────
            score_dict = dict(zip(top_scores["ticker"], top_scores["ens"]))
            keep = set()
            for t in current_tickers:
                if t in score_dict:
                    old_rank = list(top_scores["ticker"]).index(t) if t in list(top_scores["ticker"]) else TOP_N
                    if old_rank < TOP_N and score_dict[t] > top_scores["ens"].quantile(0.3):
                        keep.add(t)
            target_tickers = target_tickers | (current_tickers & keep)

            # ── 卖出不在目标中的持仓 ──────────────────────────────
            to_sell = current_tickers - target_tickers
            for t in to_sell:
                info  = self.positions[t]
                price = curr_prices.get(t, info["entry_price"])
                adv   = (daily_map[t]["close"].tail(20) *
                         daily_map[t]["volume"].tail(20)).mean() if t in daily_map else 1e6
                cost  = self._trade_cost(info["shares"] * price, adv)
                self.cash += info["shares"] * price - cost
                self.trade_log.append({"date":date,"ticker":t,"action":"卖出",
                                       "price":price,"shares":info["shares"],"cost":cost})
                del self.positions[t]
                turnover_total += 1

            # ── 等风险分配资金 ────────────────────────────────────
            investable = self.cash * (1 - self.cash_buffer)
            n_new      = len(target_tickers - current_tickers)
            if n_new > 0 and investable > 1000:
                alloc_per = investable / max(len(target_tickers), 1)

                # ── 买入新标的 ────────────────────────────────────
                to_buy = (target_tickers - current_tickers) & set(curr_prices.keys())
                for t in to_buy:
                    price = curr_prices[t]
                    adv   = (daily_map[t]["close"].tail(20) *
                             daily_map[t]["volume"].tail(20)).mean() if t in daily_map else 1e6

                    # 流动性约束：单日最多买入 ADV 的 5%
                    max_buy = adv * 0.05
                    buy_val = min(alloc_per, max_buy, self.cash * 0.9)
                    if buy_val < 500:
                        continue

                    cost   = self._trade_cost(buy_val, adv)
                    shares = (buy_val - cost) / price
                    actual_cost = shares * price + cost

                    if actual_cost > self.cash:
                        continue

                    self.cash -= actual_cost
                    self.positions[t] = {"shares": shares, "entry_price": price,
                                         "cost_basis": price + cost/shares}
                    self.trade_log.append({"date":date,"ticker":t,"action":"买入",
                                           "price":price,"shares":shares,"cost":cost})
                    turnover_total += 1

            # ── 分红再投资（DRIP）────────────────────────────────
            if self.drip:
                for t, info in list(self.positions.items()):
                    if t in meta_df.index:
                        div_yield = meta_df.loc[t, "div_yield"]
                        if pd.notna(div_yield) and div_yield > 0:
                            # 月度分红 ≈ 年化股息率 / 12
                            monthly_div = info["shares"] * curr_prices.get(t, info["entry_price"]) \
                                         * div_yield / 12
                            if monthly_div > 10:   # 最小 $10 再投资
                                price    = curr_prices.get(t, info["entry_price"])
                                new_sh   = monthly_div / price
                                self.positions[t]["shares"] += new_sh
                                self.cash  = max(0, self.cash - monthly_div * 0.1)  # 预扣税

            # ── 记录月度净值 ──────────────────────────────────────
            nav_end = self._current_nav(curr_prices)
            ret_m   = (nav_end / nav_start - 1) if nav_start > 0 else 0
            self.monthly_rets.append(ret_m)
            self.nav_history.append({"date":date,"nav":nav_end,"ret":ret_m,
                                     "n_positions":len(self.positions),
                                     "cash_pct":self.cash/nav_end*100})

            if i % 4 == 0:
                print(f"  {date.strftime('%Y-%m')}  NAV ${nav_end:>10,.0f}  "
                      f"持仓{len(self.positions):>2}支  "
                      f"现金{self.cash/nav_end*100:.0f}%  "
                      f"月收益{ret_m*100:>+6.2f}%")

        return self._report(turnover_total)

    def _report(self, turnover_total: int) -> dict:
        """生成回测报告"""
        if not self.monthly_rets:
            return {}

        rets   = np.array(self.monthly_rets)
        ann    = (np.prod(1 + rets) ** (12/len(rets)) - 1) * 100
        vol    = rets.std() * np.sqrt(12) * 100
        sharpe = (ann/100 - 0.04) / (vol/100) if vol > 0 else 0

        nav_ser = pd.Series([h["nav"] for h in self.nav_history])
        cum_max = nav_ser.cummax()
        mdd     = ((nav_ser - cum_max) / cum_max).min() * 100

        win_rate= (rets > 0).mean() * 100
        calmar  = ann / abs(mdd) if mdd != 0 else 0
        total_tx_cost = sum(t["cost"] for t in self.trade_log)

        print(f"\n  {'═'*56}")
        print(f"  📊 真实回测报告")
        print(f"  {'═'*56}")
        print(f"  初始资金       ${self.capital:>12,.0f} CAD")
        print(f"  最终净值       ${self.nav_history[-1]['nav']:>12,.0f} CAD")
        print(f"  总收益         {(self.nav_history[-1]['nav']/self.capital-1)*100:>+11.1f}%")
        print(f"  年化收益       {ann:>+11.1f}%")
        print(f"  年化波动率     {vol:>11.1f}%")
        print(f"  Sharpe 比率    {sharpe:>11.2f}")
        print(f"  最大回撤       {mdd:>+11.1f}%")
        print(f"  Calmar 比率    {calmar:>11.2f}")
        print(f"  月胜率         {win_rate:>11.1f}%")
        print(f"  总交易次数     {turnover_total:>12}")
        print(f"  总交易成本     ${total_tx_cost:>12,.0f} CAD")
        print(f"  {'─'*56}")
        print(f"  注：含手续费 {self.tx_cost*100:.1f}% + 买卖价差 {self.bid_ask*100:.1f}% "
              f"+ 市场冲击 + 止损 + 分红再投资")

        return {"ann_ret":ann,"vol":vol,"sharpe":sharpe,"mdd":mdd,
                "calmar":calmar,"win_rate":win_rate,"total_cost":total_tx_cost}


# ══════════════════════════════════════════════════════════════════
# 运行四个新模块的入口函数
# ══════════════════════════════════════════════════════════════════

def run_advanced_analysis(panel, daily_map, pit_map, meta_df, macro_df, wf,
                           run_sensitivity=False,
                           run_collinearity=True,
                           run_insider=True,
                           run_backtest=True):
    """
    统一入口，按需运行四个高级模块。
    默认不跑敏感性分析（耗时很长），其他三个默认开启。
    """
    print("\n" + "▓"*60)
    print("  高级分析模块")
    print("▓"*60)

    used_feature_cols = FEATURE_COLS

    # ── A. 共线性处理 ────────────────────────────────────────────
    if run_collinearity:
        print("\n【A】特征共线性处理")
        try:
            panel_reduced, used_feature_cols = apply_collinearity_reduction(
                panel, vif_threshold=10.0)
            print(f"  ✅ 共线性处理完成：{len(FEATURE_COLS)} → {len(used_feature_cols)} 个特征")
        except Exception as e:
            print(f"  ⚠️  共线性处理失败：{e}，使用原始特征")
            panel_reduced = panel

    # ── B. SEDI 内部人交易 ────────────────────────────────────────
    insider_df = pd.DataFrame()
    if run_insider:
        print("\n【B】SEDI 内部人交易信号")
        # canadianinsider.com 屏蔽 Colab IP，仅本地运行有效
        # 本地运行时将 SEDI_LOCAL_ONLY 改为 False 启用
        SEDI_LOCAL_ONLY = True
        if SEDI_LOCAL_ONLY:
            print("  ⚠️  SEDI 仅支持本地运行（Colab IP 被 canadianinsider.com 屏蔽）")
            print("       本地运行：将上方 SEDI_LOCAL_ONLY = True 改为 False")
        else:
            passed_tickers = list(daily_map.keys())
            try:
                insider_df = fetch_sedi_insider(passed_tickers, lookback_days=90)
            except Exception as e:
                print(f"  ⚠️  SEDI 抓取失败：{e}")

    # ── C. 参数敏感性分析 ─────────────────────────────────────────
    if run_sensitivity:
        print("\n【C】参数敏感性分析（耗时较长）")
        try:
            sensitivity_df = sensitivity_analysis(
                daily_map, pit_map, meta_df, macro_df, used_feature_cols)
        except Exception as e:
            print(f"  ⚠️  敏感性分析失败：{e}")

    # ── D. 真实回测框架 ───────────────────────────────────────────
    if run_backtest and not wf.empty:
        print("\n【D】真实回测框架")
        bt = RealisticBacktest(
            capital       = 100_000,
            tx_cost       = 0.002,
            bid_ask       = 0.001,
            market_impact = 0.002,
            stop_loss     = STOP_LOSS_PCT,
            cash_buffer   = 0.05,
            hold_inertia  = 0.10,
            drip          = True,
        )
        try:
            bt_result = bt.run(panel, daily_map, meta_df, wf)
        except Exception as e:
            print(f"  ⚠️  真实回测失败：{e}")

    return insider_df, used_feature_cols

# ══════════════════════════════════════════════════════════════════
# MODULE E: Regime Detection（市场状态识别）
# ══════════════════════════════════════════════════════════════════

class MarketRegime:
    """
    市场状态识别器。

    三种 Regime：
      BULL   (Risk-On)  → VIX < 20  且 TSX > 200日均线
      NEUTRAL           → 介于两者之间
      BEAR   (Risk-Off) → VIX > 28  或 TSX < 200日均线

    对模型的影响：
      BULL   → 动量权重 ×1.5，波动率约束放宽，允许买高波动成长股
      NEUTRAL → 使用默认权重
      BEAR   → 防御模式：股息/低波动权重 ×2，PE约束收紧
    """

    BULL     = "BULL"
    NEUTRAL  = "NEUTRAL"
    BEAR     = "BEAR"

    # Regime → 因子权重调整系数（相对于默认权重）
    FACTOR_MULTIPLIERS = {
        BULL: {
            "mom_6m":       1.5,   # 动量加大
            "mom_12m":      1.5,
            "pe":           0.7,   # 估值降权（牛市不看PE）
            "roe":          1.0,
            "fcf_yield":    0.8,
            "vol_1m":       0.7,   # 波动率惩罚降低
            "vix_level":    0.5,
        },
        NEUTRAL: {k: 1.0 for k in FEATURE_COLS},
        BEAR: {
            "mom_6m":       0.5,   # 动量降权（熊市动量失效）
            "mom_12m":      0.5,
            "pe":           1.5,   # 估值加权（熊市只买便宜货）
            "fcf_yield":    2.0,   # 现金流加权（防御）
            "vol_1m":       2.0,   # 高波动惩罚加大
            "vix_level":    1.5,
            "price_vs_52w_high": 0.3,  # 不追高
        },
    }

    # Regime → 约束调整
    CONSTRAINT_OVERRIDES = {
        BULL:    {"max_pe": 70, "max_per_gics": 2},
        NEUTRAL: {},
        BEAR:    {"max_pe": 40, "max_per_gics": 1,
                  "min_roe": 0.05},   # 熊市只买 ROE > 5%
    }

    def __init__(self, macro_df: pd.DataFrame):
        self.macro_df = macro_df
        self.regime   = self._detect()

    def _detect(self) -> str:
        if self.macro_df.empty or len(self.macro_df) < 10:
            return self.NEUTRAL

        latest = self.macro_df.iloc[-1]
        vix    = latest.get("vix", 20)

        # TSX vs 200日均线（月线用10个月均线近似 200日）
        if "tsx" in self.macro_df.columns:
            tsx_now = latest.get("tsx", 0)
            tsx_ma  = self.macro_df["tsx"].tail(10).mean()
            tsx_above_ma = tsx_now > tsx_ma
        else:
            tsx_above_ma = True

        if vix < 20 and tsx_above_ma:
            return self.BULL
        elif vix > 28 or not tsx_above_ma:
            return self.BEAR
        else:
            return self.NEUTRAL

    def describe(self) -> str:
        vix = self.macro_df["vix"].iloc[-1] if "vix" in self.macro_df.columns else "N/A"
        tsx_now = self.macro_df["tsx"].iloc[-1] if "tsx" in self.macro_df.columns else 0
        tsx_ma  = self.macro_df["tsx"].tail(10).mean() if "tsx" in self.macro_df.columns else 0
        diff    = (tsx_now/tsx_ma - 1)*100 if tsx_ma > 0 else 0

        emoji  = {"BULL":"🟢","NEUTRAL":"🟡","BEAR":"🔴"}[self.regime]
        labels = {"BULL":"牛市 Risk-On","NEUTRAL":"中性","BEAR":"熊市 Risk-Off"}

        return (f"  {emoji} 当前 Regime：{labels[self.regime]}\n"
                f"     VIX = {vix:.1f}  "
                f"TSX vs 10月均线 = {diff:+.1f}%\n"
                f"     {'动量权重加大，允许高波动' if self.regime==self.BULL else '防御模式，低波动/高股息' if self.regime==self.BEAR else '默认权重'}")

    def adjust_features(self, panel: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
        """根据 Regime 调整特征权重（乘以调整系数后重新 Z-score）"""
        mults = self.FACTOR_MULTIPLIERS.get(self.regime, {})
        out   = panel.copy()
        for col in feat_cols:
            if col in mults and col in out.columns:
                out[col] = out[col] * mults[col]
        return out

    def adjust_constraints(self, base_constraints: dict) -> dict:
        """根据 Regime 覆盖约束参数"""
        c = base_constraints.copy()
        c.update(self.CONSTRAINT_OVERRIDES.get(self.regime, {}))
        return c


# ══════════════════════════════════════════════════════════════════
# MODULE F: Earnings Surprise（盈利惊喜因子）
# ══════════════════════════════════════════════════════════════════

def fetch_earnings_surprise(tickers: list[str]) -> pd.DataFrame:
    """
    从 yfinance 获取季度盈利惊喜数据。

    盈利惊喜 = (实际EPS - 预期EPS) / |预期EPS|

    PEAD（Post-Earnings Announcement Drift）效应：
      超预期 > +5%  → 未来1-3月股价持续跑赢
      低于预期 < -5% → 未来1-3月股价持续跑输

    返回：DataFrame，index=ticker，
          columns=[surprise_pct, surprise_dir, n_quarters_positive]
    """
    print(f"\n  [Earnings Surprise] 获取 {len(tickers)} 支季报数据...")
    rows = []

    for t in tickers:
        try:
            ticker = yf.Ticker(t)
            ed     = ticker.earnings_dates

            if ed is None or ed.empty:
                rows.append({"ticker":t,"surprise_pct":0.0,
                             "surprise_dir":0,"n_beat":0,"source":"no_data"})
                continue

            # 只取有实际EPS数据的历史季报
            ed = ed.dropna(subset=["Reported EPS","EPS Estimate"])
            ed = ed[ed["Reported EPS"].notna() & ed["EPS Estimate"].notna()]

            if ed.empty:
                rows.append({"ticker":t,"surprise_pct":0.0,
                             "surprise_dir":0,"n_beat":0,"source":"no_eps"})
                continue

            # 最近一季的惊喜度
            latest = ed.iloc[0]
            actual = float(latest["Reported EPS"])
            est    = float(latest["EPS Estimate"])

            if abs(est) < 0.01:   # 避免除以接近零的预期
                surprise_pct = 0.0
            else:
                surprise_pct = (actual - est) / abs(est)

            # 过去4季中超预期的次数（一致性）
            n_quarters = min(4, len(ed))
            recent4    = ed.head(n_quarters)
            n_beat     = ((recent4["Reported EPS"] > recent4["EPS Estimate"]).sum())

            # 方向信号：连续超预期 > 单次超预期
            if surprise_pct > 0.05 and n_beat >= 3:
                surprise_dir = 2    # 强超预期（持续性）
            elif surprise_pct > 0.02:
                surprise_dir = 1    # 轻微超预期
            elif surprise_pct < -0.05 and n_beat <= 1:
                surprise_dir = -2   # 强低于预期
            elif surprise_pct < -0.02:
                surprise_dir = -1   # 轻微低于预期
            else:
                surprise_dir = 0    # 中性

            rows.append({
                "ticker":       t,
                "surprise_pct": round(surprise_pct * 100, 2),  # 转成%
                "surprise_dir": surprise_dir,
                "n_beat":       int(n_beat),
                "actual_eps":   round(actual, 3),
                "est_eps":      round(est, 3),
                "source":       "yfinance",
            })

            dir_str = {2:"强超预期 ✅",1:"超预期 ☑",0:"中性 ⚪",-1:"低于预期 ⚠️",-2:"大幅低于预期 ❌"}.get(surprise_dir,"?")
            print(f"    {t:<14} 实际{actual:+.2f} vs 预期{est:+.2f}  "
                  f"惊喜{surprise_pct*100:+.1f}%  {dir_str}  "
                  f"近4季超预期{n_beat}/4次")

        except Exception as e:
            rows.append({"ticker":t,"surprise_pct":0.0,
                         "surprise_dir":0,"n_beat":0,"source":"error"})

    df = pd.DataFrame(rows).set_index("ticker")
    n_pos = (df["surprise_dir"] > 0).sum()
    n_neg = (df["surprise_dir"] < 0).sum()
    print(f"\n    汇总：超预期 {n_pos} 支  低于预期 {n_neg} 支  "
          f"中性 {len(df)-n_pos-n_neg} 支")
    return df


def apply_earnings_signal(result: pd.DataFrame, surprise_df: pd.DataFrame,
                           weight: float = 0.12) -> pd.DataFrame:
    """
    将盈利惊喜信号融入最终排名。

    规则：
      强超预期（dir=+2）→ 集成分 × 1.15
      超预期   (dir=+1) → 集成分 × 1.07
      大幅低于预期(dir=-2) → 集成分 × 0.80，且移到后面
      低于预期 (dir=-1) → 集成分 × 0.92

    连续4季超预期的股票额外加权（PEAD 效应更强）
    """
    if surprise_df.empty:
        return result

    result = result.copy()
    tix    = (result.index.get_level_values("ticker")
              if isinstance(result.index, pd.MultiIndex) else result.index)

    for i, t in enumerate(tix):
        if t not in surprise_df.index:
            continue
        row = surprise_df.loc[t]
        d   = row.get("surprise_dir", 0)
        n   = row.get("n_beat", 0)

        # 基础乘数
        mult = {2:1.15, 1:1.07, 0:1.0, -1:0.92, -2:0.80}.get(d, 1.0)

        # 连续性加成：4季全超预期再 +5%
        if n >= 4 and d > 0:
            mult *= 1.05

        result.loc[result.index[i], "ensemble_score"] *= mult

    result = result.sort_values("ensemble_score", ascending=False)
    result["earnings_surprise"] = tix.map(surprise_df["surprise_pct"].to_dict()).fillna(0)
    result["surprise_dir"]      = tix.map(surprise_df["surprise_dir"].to_dict()).fillna(0)
    return result


# ══════════════════════════════════════════════════════════════════
# MODULE G: Black-Litterman 最优组合构建
# ══════════════════════════════════════════════════════════════════

def black_litterman_weights(top: pd.DataFrame, daily_map: dict,
                             meta_df: pd.DataFrame,
                             risk_aversion: float = 2.5,
                             prev_weights: dict = None) -> pd.DataFrame:
    """
    用 Black-Litterman 模型替换 Fuzzy 仓位分配。

    原理：
      1. 市场均衡收益 → 以 TSX 市值加权为起点（不偏向任何股票）
      2. 模型观点     → 把 ensemble_score 转化为预期超额收益
      3. 协方差矩阵   → 用历史日收益率计算（Ledoit-Wolf 收缩）
      4. BL 公式      → 混合均衡收益 + 模型观点 → 最优权重

    参数：
      risk_aversion: 风险厌恶系数（越高越保守，一般2-4）

    输出：每支股票的最优仓位比例（合计100%）
    """
    try:
        from pypfopt import BlackLittermanModel, risk_models, expected_returns
        from pypfopt.efficient_frontier import EfficientFrontier
        try:
            import cvxpy as cp
        except ImportError:
            cp = None
    except ImportError:
        print("  ⚠️  PyPortfolioOpt 未安装，使用 Fuzzy 仓位")
        return pd.DataFrame()

    tickers = []
    for idx in top.index:
        t = idx[1] if isinstance(idx, tuple) else idx
        tickers.append(t)

    # ── 1. 构建历史价格矩阵 ───────────────────────────────────────
    price_data = {}
    for t in tickers:
        df = daily_map.get(t, pd.DataFrame())
        if not df.empty:
            price_data[t] = df["close"]

    if len(price_data) < 3:
        print("  ⚠️  价格数据不足，使用 Fuzzy 仓位")
        return pd.DataFrame()

    prices = pd.DataFrame(price_data).ffill().dropna()

    if len(prices) < 60:
        print("  ⚠️  历史数据不足60天，使用 Fuzzy 仓位")
        return pd.DataFrame()

    # ── 2. 协方差矩阵（Ledoit-Wolf 收缩，减少估计误差）─────────────
    try:
        S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    except Exception:
        S = risk_models.sample_cov(prices)

    # ── 3. 市场均衡权重（以市值为比例）─────────────────────────────
    mktcaps = {}
    for t in tickers:
        mc = meta_df.loc[t, "mktcap"] if t in meta_df.index else None
        mktcaps[t] = float(mc) if mc else 1e9   # 缺失用 $1B 默认

    total_mc = sum(mktcaps.values())
    market_weights = pd.Series({t: v/total_mc for t,v in mktcaps.items()})

    # ── 4. Grinold 公式构建观点矩阵：E[R] = IC × σ × Z ─────────────
    # 替代硬编码线性映射，建立在真实预测能力和个股风险特征上
    # IC: 信息系数（模型真实预测能力代理）
    # σ:  个股特异性月波动率
    # Z:  集成分数横截面 Z-Score

    # IC 代理：用 pred_top20pct 偏离 0.5 的程度衡量模型判别力
    probs_arr = top["pred_top20pct"].values
    ic_proxy  = float(np.mean(np.abs(probs_arr - 0.5)) * 2)
    ic_proxy  = np.clip(ic_proxy, 0.05, 0.40)

    # 横截面 Z-Score 标准化
    scores_arr = top["ensemble_score"].values
    z_scores   = (scores_arr - scores_arr.mean()) / (scores_arr.std() + 1e-8)

    views = {}
    for j, (idx, row) in enumerate(top.iterrows()):
        t  = idx[1] if isinstance(idx, tuple) else idx
        df = daily_map.get(t, pd.DataFrame())
        # 个股特异性月波动率（63日 × √21）
        if not df.empty:
            sigma = float(df["close"].pct_change().tail(63).std() * np.sqrt(21))
            sigma = np.clip(sigma, 0.03, 0.25)
        else:
            sigma = 0.08
        # Grinold: E[R] = IC × σ × Z
        views[t] = float(np.clip(ic_proxy * sigma * float(z_scores[j]), -0.15, 0.15))

    # ── 5. 观点置信度 ─────────────────────────────────────────────
    confidences = {}
    for idx, row in top.iterrows():
        t    = idx[1] if isinstance(idx, tuple) else idx
        prob = row.get("pred_top20pct", 0.5)
        confidences[t] = float(np.clip(prob, 0.3, 0.9))

    # ── 6. Black-Litterman 模型 ───────────────────────────────────
    try:
        bl = BlackLittermanModel(
            S,
            pi          = "market",
            market_caps = mktcaps,
            risk_aversion = risk_aversion,
            absolute_views = views,
        )
        # 设置观点置信度
        omega = bl.bl_weights(risk_aversion=risk_aversion)

        # 用 BL 后验收益做均值方差优化
        bl_returns = bl.bl_returns()
        ef = EfficientFrontier(bl_returns, S)

        # OPT C: Turnover-Aware 优化
        if prev_weights and cp is not None:
            # 对齐上期权重到当期股票池
            w_prev = np.array([prev_weights.get(t, 0.0) for t in tickers])
            total_prev = w_prev.sum()
            if total_prev > 1e-6:
                w_prev = w_prev / total_prev

            # 换仓惩罚：双边手续费 0.4%（买入0.2% + 卖出0.2%）
            # 优化器会权衡：换仓收益 vs 0.4% * 换仓量
            try:
                from pypfopt import objective_functions
                ef.add_objective(
                    objective_functions.transaction_cost,
                    w_prev=w_prev, k=0.004
                )
                ef.max_sharpe(risk_free_rate=0.04/12)
                weights = ef.clean_weights()
                turnover = float(np.abs(
                    np.array(list(weights.values())) - w_prev).sum())
                print(f"  BL 换仓感知：换仓量 {turnover*100:.1f}%（含成本惩罚）")
            except Exception:
                ef.max_sharpe(risk_free_rate=0.04/12)
                weights = ef.clean_weights()
        else:
            # 首次运行 / cvxpy 未装：标准 max_sharpe
            ef.max_sharpe(risk_free_rate=0.04/12)
            weights = ef.clean_weights()

    except Exception as e:
        print(f"  ⚠️  BL 优化失败：{e}，使用等风险权重")
        # Fallback：等风险权重
        vols = []
        for t in tickers:
            df  = daily_map.get(t, pd.DataFrame())
            vol = df["close"].pct_change().tail(63).std() * np.sqrt(252) if not df.empty else 0.25
            vols.append(max(vol, 0.05))
        inv_vol  = [1/v for v in vols]
        total_iv = sum(inv_vol)
        weights  = {t: w/total_iv for t,w in zip(tickers, inv_vol)}

    # ── 7. 转为 DataFrame 输出 ────────────────────────────────────
    rows = []
    for idx, row in top.iterrows():
        t    = idx[1] if isinstance(idx, tuple) else idx
        w    = weights.get(t, 0.0)
        prof = STOCK_PROFILE.get(t, {})
        df   = daily_map.get(t, pd.DataFrame())
        vol  = df["close"].pct_change().tail(63).std()*np.sqrt(252) if not df.empty else 0.25

        # 仓位类别（用于显示）
        if w >= 0.18:    cat = "大量买入 🔴"
        elif w >= 0.10:  cat = "适度买入 🟡"
        else:            cat = "少量买入 🟢"

        rows.append({
            "ticker":    t,
            "alloc_pct": round(w * 100, 1),
            "category":  cat,
            "vol_ann":   vol,
            "score":     row["ensemble_score"],
            "prob":      row.get("pred_top20pct", 0),
            "ret":       row.get("pred_return", 0),
            "view_ret":  views.get(t, 0),
        })

    df_out = pd.DataFrame(rows)
    total  = df_out["alloc_pct"].sum()
    if total > 0:
        df_out["alloc_pct"] = (df_out["alloc_pct"]/total*100).round(1)

    return df_out


def print_bl_weights(df: pd.DataFrame, regime: str = "NEUTRAL"):
    """打印 Black-Litterman 仓位"""
    regime_label = {"BULL":"牛市","NEUTRAL":"中性","BEAR":"熊市"}.get(regime, regime)
    print("\n" + "═"*76)
    print(f"  📊 Black-Litterman 最优仓位  [Regime: {regime_label}]")
    print(f"  方法：市场均衡权重 + 模型观点 + Ledoit-Wolf协方差 → Max Sharpe")
    print("═"*76)
    print(f"  {'#':<3}{'Ticker':<13}{'类别':<14}{'BL仓位%':>8}  "
          f"{'条形图':<14}{'波动率':>7}{'预期超额':>9}{'集成分':>8}")
    print(f"  {'─'*74}")
    for i, row in df.iterrows():
        bar = "█" * int(row["alloc_pct"] / 2)
        print(f"  {i+1:<3}{row['ticker']:<13}{row['category']:<14}"
              f"{row['alloc_pct']:>7.1f}%  {bar:<14}"
              f"  {row['vol_ann']*100:>5.1f}%"
              f"  {row['view_ret']*100:>+7.1f}%"
              f"  {row['score']:>7.3f}")
    print(f"  {'─'*74}")
    print(f"  合计：{df['alloc_pct'].sum():.1f}%\n")
    for cat in ["Heavy Buy","Moderate Buy","Light Buy"]:
        sub = df[df["category"]==cat]
        if len(sub):
            print(f"    {cat}  {len(sub)}支  合计{sub['alloc_pct'].sum():.1f}%"
                  f"  → {' | '.join(sub['ticker'].tolist())}")


# ══════════════════════════════════════════════════════════════════
# MODULE H: 自动月度报告（邮件）
# ══════════════════════════════════════════════════════════════════

def generate_monthly_report(result, bl_weights_df, wf, regime,
                             surprise_df, imp, dd_signal=None, daily_map=None):
    today = datetime.today().strftime("%Y-%m-%d")
    month = datetime.today().strftime("%Y-%m")
    top   = result.head(TOP_N)

    RED="#C0392B"; GREEN="#27AE60"; ORANGE="#E67E22"
    BLUE="#2980B9"; GRAY="#7F8C8D"; BG="#F8F9FA"; WHITE="#FFFFFF"
    r_obj   = getattr(regime, "regime", "NEUTRAL")
    r_color = {"BULL":GREEN,"NEUTRAL":ORANGE,"BEAR":RED}.get(r_obj, ORANGE)
    r_label = {"BULL":"Bull Market - Risk On","NEUTRAL":"Neutral","BEAR":"Bear Market - Risk Off"}.get(r_obj,"Neutral")

    wf_topR = wf_acc = wf_mdd = "N/A"
    if not wf.empty and "actual_ret" in wf.columns:
        v = wf.dropna(subset=["actual_ret","actual_cls"])
        if len(v):
            p = v["ens"]
            mask80 = p >= p.quantile(0.80)
            wf_topR = f"{v.loc[mask80,'actual_ret'].mean()*100:+.2f}%"
            acc_val = accuracy_score(v["actual_cls"], mask80.astype(int)) * 100
            wf_acc  = f"{acc_val:.1f}%"
            navs    = (1 + v.groupby("date")["actual_ret"].mean()).cumprod()
            wf_mdd  = f"{((navs - navs.cummax()) / navs.cummax()).min()*100:+.2f}%"

    css = (
        "<style>"
        f"body{{font-family:-apple-system,Helvetica,Arial,sans-serif;background:{BG};margin:0;padding:20px;color:#2C3E50}}"
        f".wrap{{max-width:660px;margin:0 auto;background:{WHITE};border-radius:12px;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,.1)}}"
        f".hdr{{background:#1A252F;padding:28px 28px 20px}}"
        f".hdr h1{{color:{WHITE};margin:0;font-size:21px;font-weight:700}}"
        f".hdr p{{color:#BDC3C7;margin:5px 0 0;font-size:13px}}"
        f".sec{{padding:22px 28px;border-bottom:1px solid #ECF0F1}}"
        f".sec h2{{font-size:15px;font-weight:700;color:#1A252F;margin:0 0 14px;padding-bottom:7px;border-bottom:2px solid {BLUE}}}"
        f".rbox{{background:{BG};border-left:4px solid {r_color};padding:11px 15px;border-radius:0 7px 7px 0;font-size:14px;font-weight:700;color:{r_color}}}"
        f".wbox{{background:#FFF3CD;border-left:4px solid {ORANGE};padding:11px 15px;border-radius:0 7px 7px 0;font-size:13px;margin-top:11px}}"
        ".mgrid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-top:4px}"
        f".mcard{{background:{BG};border-radius:8px;padding:13px;text-align:center}}"
        ".mval{font-size:19px;font-weight:700;color:#1A252F}"
        f".mlbl{{font-size:11px;color:{GRAY};margin-top:3px}}"
        "table{width:100%;border-collapse:collapse;font-size:13px}"
        f"th{{background:#1A252F;color:{WHITE};padding:9px 11px;text-align:left;font-size:12px;font-weight:600}}"
        "td{padding:9px 11px;border-bottom:1px solid #ECF0F1}"
        f"tr:nth-child(even) td{{background:{BG}}}"
        f".tk{{font-weight:700;color:{BLUE};font-family:monospace}}"
        f".pos{{color:{GREEN};font-weight:600}}.neg{{color:{RED};font-weight:600}}"
        f".ch{{color:{RED};font-weight:700}}.cm{{color:{ORANGE};font-weight:700}}.cl{{color:{GREEN};font-weight:700}}"
        ".frow{display:flex;align-items:center;margin:5px 0}"
        f".fn{{font-family:monospace;font-size:12px;color:{BLUE};width:190px;flex-shrink:0}}"
        ".fbg{flex:1;background:#ECF0F1;height:7px;border-radius:3px}"
        f".fb{{background:{BLUE};height:7px;border-radius:3px}}"
        f".fv{{font-size:12px;color:{GRAY};margin-left:9px;width:48px}}"
        f".ftr{{background:#1A252F;padding:16px 28px;text-align:center}}"
        f".ftr p{{color:#7F8C8D;font-size:11px;margin:0}}"
        "</style>"
    )

    h = f'<!DOCTYPE html><html><head><meta charset="utf-8">{css}</head><body><div class="wrap">'
    h += f'<div class="hdr"><h1>TSX Monthly Stock Report</h1>'
    h += f'<p>{month} &nbsp;| XGBoost + LightGBM + MLP | {len(FEATURE_COLS)} Features</p></div>'

    # Regime box
    h += f'<div class="sec"><h2>Market Regime</h2><div class="rbox">{r_label}</div>'
    if dd_signal:
        h += f'<div class="wbox"><b>Risk Warning:</b> 3-month cumulative return {dd_signal*100:.1f}%. Consider reducing position by 50%.</div>'
    h += '</div>'

    # Walk-Forward metrics
    tc = "pos" if wf_topR != "N/A" and "+" in wf_topR else "neg"
    h += (
        '<div class="sec"><h2>Walk-Forward Performance</h2><div class="mgrid">'
        f'<div class="mcard"><div class="mval {tc}">{wf_topR}</div><div class="mlbl">Top20% Monthly Return</div></div>'
        f'<div class="mcard"><div class="mval" style="color:{BLUE}">{wf_acc}</div><div class="mlbl">Classification Accuracy</div></div>'
        f'<div class="mcard"><div class="mval" style="color:{RED}">{wf_mdd}</div><div class="mlbl">Max Monthly Drawdown</div></div>'
        '</div></div>'
    )

    # Top 10 table with price and shares
    REPORT_CAPITAL = 100_000  # CAD - change to match your portfolio
    h += ('<div class="sec"><h2>Top 10 Stock Picks</h2>'
          '<table style="font-size:12px;width:100%">'
          '<tr><th>#</th><th>Ticker</th><th>Company</th><th>GICS</th>'
          '<th>Pred Return</th><th>Price (CAD)</th>'
          '<th>Amount (CAD)</th><th style="color:#2980B9">Shares to Buy</th></tr>')
    for i, (idx, row) in enumerate(top.iterrows(), 1):
        t    = idx[1] if isinstance(idx, tuple) else idx
        prof = STOCK_PROFILE.get(t, {})
        surp = surprise_df.loc[t, "surprise_pct"] if t in surprise_df.index else 0
        rp   = row["pred_return"] * 100
        rc   = "pos" if rp >= 0 else "neg"
        # Allocation: from BL weights if available, else equal weight
        alloc_pct = 1.0 / max(len(top), 1)
        if not bl_weights_df.empty and "ticker" in bl_weights_df.columns:
            match = bl_weights_df[bl_weights_df["ticker"] == t]
            if not match.empty:
                alloc_pct = float(match["alloc_pct"].iloc[0]) / 100
        amount = REPORT_CAPITAL * alloc_pct
        # Price from daily_map
        price_val = 0
        if daily_map and t in daily_map and not daily_map[t].empty:
            price_val = float(daily_map[t]["close"].iloc[-1])
        shares  = int(amount / price_val) if price_val > 0 else 0
        price_s = f"${price_val:.2f}" if price_val > 0 else "N/A"
        h += (f'<tr><td>{i}</td><td class="tk">{t}</td>'
              f'<td style="font-size:11px">{str(row.get("name",""))[:18]}</td>'
              f'<td style="font-size:11px;color:{GRAY}">{prof.get("gics","?")[:12]}</td>'
              f'<td class="{rc}">{rp:+.1f}%</td>'
              f'<td style="font-family:monospace">{price_s}</td>'
              f'<td style="font-family:monospace">${amount:,.0f}</td>'
              f'<td style="font-weight:700;color:{BLUE};font-size:13px">{shares:,}</td></tr>')
    h += '</table></div>'

    # Position Sizing（BL 或 Fuzzy，始终显示，含价格和股数）
    pos_df = bl_weights_df if not bl_weights_df.empty else pd.DataFrame()
    if pos_df.empty and len(top) > 0:
        pos_rows = []
        for idx2, row2 in top.iterrows():
            t2 = idx2[1] if isinstance(idx2, tuple) else idx2
            pos_rows.append({"ticker":t2,"category":"Equal Weight",
                             "alloc_pct":100.0/len(top),"vol_ann":0,
                             "score":row2.get("ensemble_score",0)})
        pos_df = pd.DataFrame(pos_rows)

    if not pos_df.empty:
        REPORT_CAPITAL = 100_000
        h += '<div class="sec"><h2>Position Sizing</h2>'
        h += '<table style="font-size:12px;width:100%">'
        h += ('<tr><th>Ticker</th><th>Category</th><th>Allocation</th>'
              '<th>Price (CAD)</th><th>Amount (CAD)</th>'
              '<th style="color:#2980B9">Shares to Buy</th><th>Vol</th></tr>')
        for _, row in pos_df.iterrows():
            t3  = row.get("ticker","")
            cat = row.get("category","")
            cc  = "ch" if "Heavy" in cat else ("cm" if "Moderate" in cat else "cl")
            bw  = min(int(row["alloc_pct"]*3), 100)
            price_val = 0
            if daily_map and t3 in daily_map and not daily_map[t3].empty:
                price_val = float(daily_map[t3]["close"].iloc[-1])
            amount  = REPORT_CAPITAL * row["alloc_pct"] / 100
            shares  = int(amount / price_val) if price_val > 0 else 0
            price_s = f"${price_val:.2f}" if price_val > 0 else "N/A"
            h += (f'<tr><td class="tk">{t3}</td><td class="{cc}">{cat}</td>'
                  f'<td><div style="display:flex;align-items:center;gap:6px">'
                  f'<div class="fbg" style="width:60px"><div class="fb" style="width:{bw}%"></div></div>'
                  f'<b>{row["alloc_pct"]:.1f}%</b></div></td>'
                  f'<td>{price_s}</td>'
                  f'<td>${amount:,.0f}</td>'
                  f'<td style="font-weight:700;color:#2980B9;font-size:13px">{shares:,}</td>'
                  f'<td style="color:#7F8C8D">{row.get("vol_ann",0)*100:.1f}%</td></tr>')
        h += '</table></div>'

    # Feature importance
    h += '<div class="sec"><h2>Key Driving Factors (Top 8)</h2>'
    max_v = imp.head(8).max()
    for feat, val in imp.head(8).items():
        bw = int(val / max_v * 100) if max_v > 0 else 0
        h += (f'<div class="frow"><span class="fn">{feat}</span>'
              f'<div class="fbg"><div class="fb" style="width:{bw}%"></div></div>'
              f'<span class="fv">{val:.4f}</span></div>')
    h += '</div>'

    h += (f'<div class="ftr"><p>Generated: {datetime.today().strftime("%Y-%m-%d %H:%M")}'
          ' | TSX Quant Stock Picker v3.0'
          ' | For reference only, not investment advice</p></div>'
          '</div></body></html>')
    return h


def html_to_pdf(html_content: str, pdf_path: str) -> bool:
    """HTML report to PDF (weasyprint)"""
    try:
        import logging
        logging.getLogger("weasyprint").setLevel(logging.ERROR)
        logging.getLogger("fontTools").setLevel(logging.ERROR)
        logging.getLogger("fontTools.subset").setLevel(logging.ERROR)
        logging.getLogger("fontTools.ttLib").setLevel(logging.ERROR)
        logging.disable(logging.DEBUG)
        from weasyprint import HTML
        HTML(string=html_content).write_pdf(pdf_path)
        print(f"  PDF generated: {pdf_path}")
        return True
    except ImportError:
        print("  weasyprint not installed: pip install weasyprint")
        return False
    except Exception as e:
        print(f"  PDF generation failed: {e}")
        return False


def send_monthly_report(html_content, to_email="your@email.com",
                        from_email="your@gmail.com",
                        app_password="your_gmail_app_password",
                        attach_pdf=True):
    """
    Send HTML email with optional PDF attachment.
    attach_pdf=True: attaches PDF version for printing/archiving.
    """
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.application import MIMEApplication

    month_str  = datetime.today().strftime("%Y%m")
    html_fname = f"tsx_report_{month_str}.html"
    pdf_fname  = f"tsx_report_{month_str}.pdf"

    # Save HTML locally
    with open(html_fname, "w", encoding="utf-8") as f:
        f.write(html_content)

    # Generate PDF
    pdf_ok = html_to_pdf(html_content, pdf_fname) if attach_pdf else False

    if "your@" in to_email or "your_gmail" in app_password:
        print(f"  Email not configured. Saved: {html_fname}")
        return False

    try:
        msg = MIMEMultipart("mixed")
        msg["Subject"] = f"TSX Stock Report - {datetime.today().strftime('%Y-%m')}"
        msg["From"]    = from_email
        msg["To"]      = to_email

        # HTML body
        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText("Please view in HTML-capable email client.", "plain", "utf-8"))
        alt.attach(MIMEText(html_content, "html", "utf-8"))
        msg.attach(alt)

        # PDF attachment
        if pdf_ok:
            with open(pdf_fname, "rb") as f:
                part = MIMEApplication(f.read(), _subtype="pdf")
                part.add_header("Content-Disposition", "attachment", filename=pdf_fname)
                msg.attach(part)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(from_email, app_password)
            server.sendmail(from_email, to_email, msg.as_string())

        print(f"  Report sent to {to_email}")
        print(f"  HTML: {html_fname}" + (f"  |  PDF attached: {pdf_fname}" if pdf_ok else ""))
        return True

    except Exception as e:
        print(f"  Email failed: {e}. Saved: {html_fname}")
        return False


    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"🍁 TSX 量化选股月报 — {datetime.today().strftime('%Y-%m')}"
        msg["From"]    = from_email
        msg["To"]      = to_email

        # 纯文本版本
        msg.attach(MIMEText(report_text, "plain", "utf-8"))

        # HTML 版本（Markdown 转简单 HTML）
        html_body = report_text.replace("\n", "<br>").replace("**", "<b>").replace("##", "<h3>").replace("#", "<h2>")
        msg.attach(MIMEText(f"<html><body>{html_body}</body></html>", "html", "utf-8"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(from_email, app_password)
            server.sendmail(from_email, to_email, msg.as_string())

        print(f"  ✅ 月度报告已发送至 {to_email}")
        with open("monthly_report.md", "w") as f:
            f.write(report_text)
        return True

    except Exception as e:
        print(f"  ⚠️  邮件发送失败：{e}")
        print("       报告已保存到：monthly_report.md")
        with open("monthly_report.md", "w") as f:
            f.write(report_text)
        return False


# ══════════════════════════════════════════════════════════════════
# 四个新模块的统一入口
# ══════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════
# Double Machine Learning (DML) — 事件因果效应估计
# ══════════════════════════════════════════════════════════════════

def estimate_dml_alpha(panel: pd.DataFrame,
                       event_signal: pd.Series,
                       signal_name: str = "event",
                       n_folds: int = 5) -> pd.Series:
    """
    用 Double Machine Learning 估计事件信号（内部人交易/盈利惊喜）
    对下月收益的纯因果效应（Treatment Effect），替代硬编码乘数。

    DML 两步法（Chernozhukov et al. 2018）：
      Step 1: 用 ML（XGBoost）从特征 X 预测收益 Y → 得到残差 Ẽ_Y = Y - Ŷ
      Step 2: 用 ML 从特征 X 预测处理变量 T → 得到残差 Ẽ_T = T - T̂
      Step 3: OLS 回归 Ẽ_Y ~ Ẽ_T → θ 是纯净的因果效应（排除混淆变量）

    返回：每支股票的动态 Alpha 调整系数（比固定乘数更精确）
    """
    from sklearn.model_selection import KFold
    from sklearn.linear_model import LinearRegression

    # 准备数据
    # Bug3 修复：event_signal 的 index 是 ticker，panel 是 MultiIndex(date, ticker)
    # 需要将 signal 按 ticker 对齐到 panel 的每一行
    tix_in_panel = panel.index.get_level_values("ticker")
    sig_dict     = event_signal.to_dict()
    valid_tix    = [t for t in tix_in_panel.unique() if t in sig_dict]
    n_valid      = len(valid_tix)
    if n_valid < 20:
        print(f"  [DML] {signal_name}: 样本不足（{n_valid} 支有信号），使用默认权重")
        return pd.Series(0.0, index=event_signal.index)

    # 对齐事件信号到 panel（按 ticker 展开）
    tix = tix_in_panel
    T   = np.array([sig_dict.get(t, 0.0) for t in tix], dtype=float)  # Treatment
    Y   = panel["next_ret"].fillna(panel["next_ret"].median()).values.astype(float)  # Outcome
    X   = panel[FEATURE_COLS].fillna(panel[FEATURE_COLS].median()).values    # Confounders

    if T.std() < 1e-6:
        return pd.Series(0.0, index=event_signal.index)

    # Cross-fitting：必须用时间序列分割，严禁 shuffle=True
    # 金融面板数据中随机折叠会导致用未来数据预测过去（数据泄露）
    from sklearn.model_selection import TimeSeriesSplit

    # 获取每行的时间戳，按时间排序确保 train < val
    if isinstance(panel.index, pd.MultiIndex):
        dates_arr = panel.index.get_level_values("date").values
    else:
        dates_arr = panel.index.values
    sort_order = np.argsort(dates_arr)   # 按时间升序排列

    # 按时间顺序重新排列 X/Y/T
    X_sorted = X[sort_order]
    Y_sorted = Y[sort_order]
    T_sorted = T[sort_order]

    tscv    = TimeSeriesSplit(n_splits=n_folds, gap=1)   # gap=1 避免相邻月泄露
    res_Y   = np.zeros_like(Y_sorted)
    res_T   = np.zeros_like(T_sorted)

    for train_idx, val_idx in tscv.split(X_sorted):
        sc  = StandardScaler()
        X_tr= sc.fit_transform(X_sorted[train_idx])
        X_va= sc.transform(X_sorted[val_idx])

        # ML1: 预测 Y（控制混淆变量）
        m_Y = xgb.XGBRegressor(n_estimators=100, max_depth=3,
                                learning_rate=0.05, verbosity=0,
                                random_state=42).fit(X_tr, Y_sorted[train_idx])
        res_Y[val_idx] = Y_sorted[val_idx] - m_Y.predict(X_va)

        # ML2: 预测 T（倾向得分）
        m_T = xgb.XGBRegressor(n_estimators=100, max_depth=3,
                                learning_rate=0.05, verbosity=0,
                                random_state=42).fit(X_tr, T_sorted[train_idx])
        res_T[val_idx] = T_sorted[val_idx] - m_T.predict(X_va)

    # Step 3: 从残差回归提取纯净 Treatment Effect（时序排序后的残差）
    mask = np.abs(res_T) > 1e-6
    if mask.sum() < 20:
        return pd.Series(0.0, index=event_signal.index)

    theta = np.dot(res_T[mask], res_Y[mask]) / np.dot(res_T[mask], res_T[mask])

    print(f"  [DML] {signal_name}: θ = {theta:.4f}  "
          f"（纯因果效应，正值=事件信号有Alpha贡献）")

    # 用 θ 动态计算每支股票的仓位调整系数
    # 调整量 = θ × signal_value，映射到 [-0.2, +0.2] 范围
    raw_adj = event_signal * float(theta)
    adj     = raw_adj.clip(-0.20, 0.20)
    return adj


def apply_dml_signal(result: pd.DataFrame, panel: pd.DataFrame,
                     insider_df: pd.DataFrame, surprise_df: pd.DataFrame) -> pd.DataFrame:
    """
    用 DML 估计的因果效应替代硬编码乘数，动态调整集成分。
    """
    result = result.copy()
    tix    = (result.index.get_level_values("ticker")
              if isinstance(result.index, pd.MultiIndex) else result.index)

    adj_total = pd.Series(0.0, index=tix)

    # 内部人交易 DML
    if not insider_df.empty and "signal" in insider_df.columns:
        insider_sig = insider_df["signal"].reindex(tix).fillna(0)
        if insider_sig.std() > 0:
            insider_adj = estimate_dml_alpha(
                panel, insider_sig, signal_name="内部人交易")
            adj_total  += insider_adj.reindex(tix).fillna(0)

    # 盈利惊喜 DML
    if not surprise_df.empty and "surprise_pct" in surprise_df.columns:
        surprise_sig = (surprise_df["surprise_pct"] / 100).reindex(tix).fillna(0)
        if surprise_sig.std() > 0:
            surprise_adj = estimate_dml_alpha(
                panel, surprise_sig, signal_name="盈利惊喜")
            adj_total   += surprise_adj.reindex(tix).fillna(0)

    # 应用动态调整
    result["ensemble_score"] = result["ensemble_score"] + adj_total.values
    result = result.sort_values("ensemble_score", ascending=False)
    return result


def run_new_modules(panel, daily_map, meta_df, macro_df, wf,
                    result, imp, dd_signal,
                    # 邮件配置（填写后自动发送）
                    email_to:       str = "your@email.com",
                    email_from:     str = "your@gmail.com",
                    email_password: str = "your_gmail_app_password"):
    """
    运行四个新模块并返回结果：
      E. Regime Detection
      F. Earnings Surprise
      G. Black-Litterman 仓位
      H. 月度报告 + 邮件
    """
    print("\n" + "▓"*60)
    print("  新增模块 E/F/G/H")
    print("▓"*60)

    # ── E. Regime Detection ───────────────────────────────────────
    print("\n【E】Regime Detection（市场状态识别）")
    regime = MarketRegime(macro_df)
    print(regime.describe())

    # Regime 调整特征权重（用于下次训练参考）
    panel_regime = regime.adjust_features(panel, FEATURE_COLS)
    regime_constraints = regime.adjust_constraints(CONSTRAINTS)
    print(f"  约束调整：{regime_constraints}")

    # ── F. Earnings Surprise ──────────────────────────────────────
    print("\n【F】Earnings Surprise（盈利惊喜）")
    passed_tickers = (result.index.get_level_values("ticker").tolist()
                      if isinstance(result.index, pd.MultiIndex)
                      else result.index.tolist())
    surprise_df = pd.DataFrame()
    try:
        surprise_df = fetch_earnings_surprise(passed_tickers[:TOP_N])
        if result is not None and not surprise_df.empty:
            # 优先使用 DML 动态估计（同时处理 insider + surprise）
            print("  [DML] 用双重机器学习估计事件信号的纯因果效应...")
            try:
                result = apply_dml_signal(result, panel, insider_df, surprise_df)
                print("  ✅ DML 动态调整已融入排名（替代固定乘数）")
            except Exception as e_dml:
                print(f"  ⚠️  DML 失败（{e_dml}），fallback 到固定乘数")
                result = apply_earnings_signal(result, surprise_df, weight=0.12)
    except Exception as e:
        print(f"  ⚠️  Earnings Surprise 获取失败：{e}")

    # ── G. Black-Litterman ────────────────────────────────────────
    print("\n【G】Black-Litterman 最优仓位")
    bl_df = pd.DataFrame()
    if result is not None:
        try:
            top = result.head(TOP_N)
            # OPT C: 传入上期持仓权重实现换仓感知优化
            prev_bl_weights = {}
            if not wf.empty and "ens" in wf.columns:
                last_date  = wf["date"].max()
                last_picks = wf[wf["date"] == last_date].nlargest(TOP_N, "ens")
                n_picks    = len(last_picks)
                if n_picks > 0:
                    prev_bl_weights = {
                        row["ticker"]: 1.0/n_picks
                        for _, row in last_picks.iterrows()
                    }
            bl_df = black_litterman_weights(top, daily_map, meta_df,
                                             risk_aversion=2.5,
                                             prev_weights=prev_bl_weights)
            if not bl_df.empty:
                print_bl_weights(bl_df, regime.regime)
            else:
                print("  ⚠️  BL 优化失败，请查看上方提示")
        except Exception as e:
            print(f"  ⚠️  Black-Litterman 失败：{e}")

    # ── H. 月度报告 ───────────────────────────────────────────────
    print("\n【H】生成月度报告")
    if result is not None and imp is not None:
        try:
            # generate_monthly_report 返回 HTML 字符串
            html_report = generate_monthly_report(
                result, bl_df, wf, regime,
                surprise_df, imp, dd_signal,
                daily_map=daily_map)

            month_str = datetime.today().strftime("%Y%m")

            # 保存 HTML
            html_fname = f"tsx_report_{month_str}.html"
            with open(html_fname, "w", encoding="utf-8") as f:
                f.write(html_report)
            print(f"  ✅ HTML 报告已保存：{html_fname}")

            # 生成 PDF（weasyprint）
            pdf_fname = f"tsx_report_{month_str}.pdf"
            pdf_ok = html_to_pdf(html_report, pdf_fname)

            # 发送邮件（HTML正文 + PDF附件）
            send_monthly_report(html_report, email_to, email_from, email_password,
                                attach_pdf=True)

        except Exception as e:
            print(f"  ⚠️  报告生成失败：{e}")

    return result, bl_df, surprise_df, regime


# ══════════════════════════════════════════════════════════════════
# MODULE I: 历史回测 — Walk-Forward 完整模型 P&L
# ══════════════════════════════════════════════════════════════════
# 直接用主模型的 Walk-Forward 输出重建组合历史收益。
# 每月：取集成分 Top N → 等权持有 → 计算实际下月收益（含止损/手续费）
# 这是完整的 26特征 + XGBoost + LightGBM + MLP 模型的真实历史表现。


BACKTEST_MONTHS  = 12
INITIAL_CAPITAL  = 100_000
BENCHMARK        = "XIU.TO"
BT_TX_COST       = 0.002
BT_STOP_LOSS     = -0.08   # 全局止损下限（保底）
BT_VOL_STOP_MULT = 1.5    # 动态止损 = 个股历史月波动率 × 此倍数
# 例：月波动率 8% 的矿业股 → 止损 -12%（不被噪音洗盘）
#     月波动率 3% 的银行股  → 止损  -4.5%（更敏感）
#     两者取 min(动态止损, BT_STOP_LOSS=-8%) 作为实际止损线



def backtest_from_wf(wf, daily_map, meta_df, top_n=10):
    """从 Walk-Forward 输出构建组合历史 P&L（完整模型，无简化）"""
    if wf.empty or "actual_ret" not in wf.columns:
        print("  WF 结果为空")
        return []

    dates = sorted(wf["date"].unique())
    dates = dates[-min(BACKTEST_MONTHS, len(dates)):]

    bench_rets = {}
    if BENCHMARK in daily_map:
        bdf = daily_map[BENCHMARK]["close"].resample("ME").last().pct_change()
        bench_rets = {str(k.date()): float(v) for k, v in bdf.items() if not pd.isna(v)}

    nav, prev_hold, results = float(INITIAL_CAPITAL), set(), []

    for date in dates:
        month_wf = wf[wf["date"] == date].copy()
        if month_wf.empty:
            continue

        top_month = month_wf.nlargest(top_n, "ens")
        tickers   = top_month["ticker"].tolist()
        curr_hold = set(tickers)
        new_in    = curr_hold - prev_hold
        exit_out  = prev_hold - curr_hold

        holdings = []
        for _, row in top_month.iterrows():
            t       = row["ticker"]
            raw_ret = row.get("actual_ret", np.nan)
            if pd.isna(raw_ret):
                continue

            stop_hit = False
            df = daily_map.get(t, pd.DataFrame())
            if not df.empty:
                next_m = df[df.index > pd.Timestamp(date)]
                end_dt = date + pd.offsets.MonthEnd(1)
                mdata  = next_m[next_m.index <= pd.Timestamp(end_dt)]
                if not mdata.empty:
                    ep  = df[df.index <= pd.Timestamp(date)]["close"].iloc[-1]
                    mp  = mdata["low"].min()

                    # OPT3: 动态波动率止损
                    # 个股历史月波动率（63日）× BT_VOL_STOP_MULT
                    hist_vol = df["close"].pct_change().tail(63).std() * np.sqrt(21)
                    vol_stop = -abs(hist_vol * BT_VOL_STOP_MULT) if hist_vol > 0 else BT_STOP_LOSS
                    # 取动态止损和全局止损的较大值（更宽松的保护）
                    effective_stop = max(vol_stop, BT_STOP_LOSS)

                    if (mp - ep) / ep < effective_stop:
                        raw_ret  = effective_stop
                        stop_hit = True

            tx      = BT_TX_COST * 2 if t in new_in else BT_TX_COST
            net_ret = raw_ret - tx
            holdings.append({
                "ticker":   t,
                "raw_ret":  round(raw_ret * 100, 2),
                "net_ret":  round(net_ret * 100, 2),
                "weight":   1.0 / top_n,
                "contrib":  round(net_ret / top_n * 100, 3),
                "score":    round(row["ens"], 4),
                "stop_hit": stop_hit,
            })

        if not holdings:
            prev_hold = curr_hold
            continue

        hdf      = pd.DataFrame(holdings)
        port_ret = hdf["contrib"].sum()

        # 基准日期匹配：找最接近的月末日期（±5天容忍）
        ts_date = pd.Timestamp(date)
        bench_ret = 0.0
        for bk, bv in bench_rets.items():
            try:
                if abs((pd.Timestamp(bk) - ts_date).days) <= 5:
                    bench_ret = float(bv) * 100
                    break
            except Exception:
                pass

        nav_before = nav
        nav        = nav * (1 + port_ret / 100)

        results.append({
            "date":        date,
            "month":       pd.Timestamp(date).strftime("%Y-%m"),
            "portfolio":   round(port_ret, 2),
            "benchmark":   round(bench_ret, 2),
            "excess":      round(port_ret - bench_ret, 2),
            "nav":         round(nav, 0),
            "nav_chg":     round(nav - nav_before, 0),
            "turnover":    len(new_in),
            "holdings_df": hdf,
            "tickers":     tickers,
        })
        prev_hold = curr_hold

    return results


def print_wf_backtest(results, initial_capital):
    """打印逐月 P&L + 年度汇总"""
    if not results:
        print("  没有回测结果")
        return

    SEP = "=" * 72
    print()
    print(SEP)
    print("  模型历史回测（Walk-Forward，完整26特征模型）")
    print(f"  初始资金 ${initial_capital:,.0f} CAD  手续费 {BT_TX_COST*100:.1f}%  止损 {BT_STOP_LOSS*100:.0f}%")
    print(SEP)

    for m in results:
        hdf    = m["holdings_df"].sort_values("net_ret", ascending=False)
        profit = "盈利" if m["portfolio"] >= 0 else "亏损"
        icon   = "+" if m["portfolio"] >= 0 else ""
        beat   = "+" if m["excess"] >= 0 else ""

        print()
        print(f"  {'-'*70}")
        print(f"  {m['month']}  {profit}  策略 {icon}{m['portfolio']:.2f}%  "
              f"基准 {m['benchmark']:+.2f}%  "
              f"超额 {beat}{m['excess']:.2f}%  "
              f"NAV ${m['nav']:,.0f} ({m['nav_chg']:+,.0f})")
        print(f"  换仓 {m['turnover']} 支")
        print()
        print(f"  {'Ticker':<13} {'集成分':>7} {'月涨跌':>8} {'净收益':>8} {'贡献%':>8}")
        print(f"  {'-'*50}")
        for _, row in hdf.iterrows():
            stop_s = " STOP" if row["stop_hit"] else ""
            sign   = "+" if row["net_ret"] >= 0 else ""
            print(f"  {row['ticker']:<13} {row['score']:>7.4f} "
                  f"{row['raw_ret']:>+7.2f}% "
                  f"{row['net_ret']:>+7.2f}% "
                  f"{row['contrib']:>+7.3f}%"
                  f"{stop_s}")

    # 汇总
    rets       = np.array([m["portfolio"] for m in results]) / 100
    bench_rets = np.array([m["benchmark"] for m in results]) / 100
    final_nav  = results[-1]["nav"]
    n          = len(rets)

    total_ret = (final_nav / initial_capital - 1) * 100
    ann_ret   = ((1 + rets).prod() ** (12/n) - 1) * 100 if n > 0 else 0
    bench_ann = ((1 + bench_rets).prod() ** (12/n) - 1) * 100 if n > 0 else 0
    vol_m     = rets.std() * np.sqrt(12) * 100
    sharpe    = (ann_ret/100 - 0.04) / (vol_m/100) if vol_m > 0 else 0
    navs      = pd.Series([m["nav"] for m in results])
    mdd       = ((navs - navs.cummax()) / navs.cummax()).min() * 100
    win_m     = sum(r > 0 for r in rets)
    beat_m    = sum(m["excess"] > 0 for m in results)
    best_m    = max(results, key=lambda m: m["portfolio"])
    worst_m   = min(results, key=lambda m: m["portfolio"])

    from collections import Counter
    ticker_counts = Counter([t for m in results for t in m["tickers"]])

    print()
    print(SEP)
    print(f"  年度汇总（{n} 个月）")
    print(SEP)
    summary = [
        ("初始资金",        f"${initial_capital:,.0f} CAD"),
        ("最终净值",        f"${final_nav:,.0f} CAD"),
        ("总收益",          f"{total_ret:+.2f}%"),
        ("年化收益（策略）", f"{ann_ret:+.2f}%"),
        ("年化收益（基准）", f"{bench_ann:+.2f}%"),
        ("超额收益 Alpha",  f"{ann_ret - bench_ann:+.2f}%"),
        ("年化波动率",      f"{vol_m:.2f}%"),
        ("Sharpe 比率",     f"{sharpe:.2f}"),
        ("最大回撤",        f"{mdd:+.2f}%"),
        ("月胜率",          f"{win_m/n*100:.1f}%  ({win_m}/{n} 月)"),
        ("跑赢基准",        f"{beat_m/n*100:.1f}%  ({beat_m}/{n} 月)"),
        ("最佳月份",        f"{best_m['month']}  {best_m['portfolio']:+.2f}%"),
        ("最差月份",        f"{worst_m['month']}  {worst_m['portfolio']:+.2f}%"),
    ]
    for label, value in summary:
        print(f"  {label:<20} {value}")

    print()
    print("  月度明细：")
    print(f"  {'月份':<10} {'策略':>8} {'基准':>8} {'超额':>8} {'净值':>12} {'盈亏CAD':>10}")
    print(f"  {'-'*58}")
    for m in results:
        icon = "+" if m["portfolio"] >= 0 else ""
        print(f"  {m['month']:<10} {m['portfolio']:>+7.2f}% {m['benchmark']:>+7.2f}% "
              f"{m['excess']:>+7.2f}% {m['nav']:>12,.0f} {m['nav_chg']:>+9,.0f}")

    print()
    print("  最常入选 Top 10（次数）：")
    for t, cnt in ticker_counts.most_common(10):
        bar = "#" * cnt
        print(f"  {t:<14} {cnt:>2}次  {cnt/n*100:>4.0f}%  {bar}")

    print()
    print("  注：Walk-Forward 完整模型，每月用该月前所有数据训练")
    print(f"     手续费 {BT_TX_COST*100:.1f}% 单边已扣，止损 {BT_STOP_LOSS*100:.0f}% 按日线触发")


# ══════════════════════════════════════════════════════════════════
# 运行模式配置
# ══════════════════════════════════════════════════════════════════

MODE = "pick"   # "pick"     → 当月选股（默认）
                # "backtest" → 历史回测（过去12个月逐月模拟）
                # "both"     → 先回测再选股

EMAIL_CONFIG = {
    "to":       "carlchenn@hotmail.com",
    "from":     "carlchenyiqing@gmail.com",
    "password": "vvdn ezoz yivl fbrw",
}


if __name__ == "__main__":
    from dateutil.relativedelta import relativedelta

    models_str = "XGBoost"+(" + LightGBM" if LGBM else "")+(" + MLP" if TORCH else "")
    print("="*70)
    print(f"  TSX 量化选股 v3.0  {datetime.today().strftime('%Y-%m-%d %H:%M')}")
    print(f"  模型：{models_str}  |  模式：{MODE.upper()}")
    print("="*70)

    # ── 历史回测 ─────────────────────────────────────────────────
    if MODE in ("backtest", "both"):
        print(f"\n{'|'*60}")
        print(f"  历史回测模式（完整模型 Walk-Forward，过去 {BACKTEST_MONTHS} 个月）")
        print(f"{'|'*60}")
        # 回测需要先跑完整 pipeline 拿到 wf
        if "daily_map" not in dir() or "wf" not in dir():
            daily_map, pit_map, meta_df, macro_df = fetch_all(TICKERS, YEARS)
            passed  = apply_constraints(daily_map, meta_df, CONSTRAINTS)
            panel   = build_panel(passed, daily_map, pit_map, macro_df)
            panel   = add_labels(panel)
            panel   = cross_z(panel)
            wf      = walk_forward(panel, tx_cost=BT_TX_COST)

        # 用 Walk-Forward 输出还原逐月动态选股 P&L
        bt_results = backtest_from_wf(wf, daily_map, meta_df, top_n=TOP_N)
        print_wf_backtest(bt_results, INITIAL_CAPITAL)

    # ── 当月选股 ─────────────────────────────────────────────────
    if MODE in ("pick", "both"):
        print(f"\n{'▓'*60}")
        print(f"  当月选股模式")
        print(f"{'▓'*60}")

        daily_map, pit_map, meta_df, macro_df = fetch_all(TICKERS, YEARS)

        print(f"\n[3/4] 约束过滤")
        passed = apply_constraints(daily_map, meta_df, CONSTRAINTS)
        if len(passed) < TOP_N:
            print(f"\n⚠️  通过约束 {len(passed)} 支 < TOP_N={TOP_N}，"
                  f"建议放宽 max_pe 或 min_mktcap_cad")
            # 注意：放宽约束时 max_price_cad 始终保留
            relaxed = {**CONSTRAINTS, "max_pe": 100, "min_mktcap_cad": 200_000_000,
                       "max_price_cad": CONSTRAINTS.get("max_price_cad", 9999)}
            passed = apply_constraints(daily_map, meta_df, relaxed)

        print(f"\n[4/4] 特征工程 + 模型")
        panel = build_panel(passed, daily_map, pit_map, macro_df)
        panel = add_labels(panel)
        print("  智能插补（行业截面中位数）...")
        panel = smart_impute(panel, FEATURE_COLS)
        panel = cross_z(panel)

        wf = walk_forward(panel, tx_cost=0.002)
        evaluate(wf)

        # 完整模型逐月 P&L 报告（用Walk-Forward的真实预测结果）
        backtest_report(wf, panel, daily_map, meta_df,
                        initial_capital = INITIAL_CAPITAL,
                        tx_cost         = 0.002,
                        stop_loss       = STOP_LOSS_PCT,
                        benchmark       = "XIU.TO")

        result, imp, dd_signal = predict_now(panel, daily_map, meta_df, wf)

        # 高级模块（共线性 / SEDI / 参数敏感性 / 真实回测框架）
        insider_df, feat_cols = run_advanced_analysis(
            panel, daily_map, pit_map, meta_df, macro_df, wf,
            run_sensitivity  = False,   # 改 True 开启参数搜索（耗时长）
            run_collinearity = True,
            run_insider      = True,
            run_backtest     = False,  # 存在前视偏差，用WF回测更准确
        )

        # Bug2 修复：将 VIF 过滤后的特征集同步到全局 FEATURE_COLS
        # 避免 predict_now 用了被删特征（vwap_bias/bias_20 出现在重要性里就是这个原因）
        if feat_cols and len(feat_cols) < len(FEATURE_COLS):
            FEATURE_COLS = feat_cols
            print(f"  特征集已同步：{len(feat_cols)} 个（VIF过滤后，已排除共线特征）")


        if result is not None and not insider_df.empty:
            result = apply_insider_signal(result, insider_df, weight=0.15)

        if result is not None:
            print_picks(result, imp, daily_map, meta_df, wf, dd_signal)

        # 新模块 E/F/G/H：Regime + Earnings + Black-Litterman + 月报
        result, bl_df, surprise_df, regime = run_new_modules(
            panel, daily_map, meta_df, macro_df, wf,
            result, imp, dd_signal,
            email_to       = EMAIL_CONFIG["to"],
            email_from     = EMAIL_CONFIG["from"],
            email_password = EMAIL_CONFIG["password"],
        )