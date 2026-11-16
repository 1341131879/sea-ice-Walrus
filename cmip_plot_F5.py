# -*- coding: utf-8 -*-

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

# ============================================================
# 1. 路径与核心参数配置
# ============================================================

GRAND_DAILY_CSV = Path(
    r"F:\CMIP6_SIday\regional_sic_results\GRAND_TOTAL_DAILY_both_regions_1995_2100.csv"
)
GRAND_CLIM_CSV = Path(
    r"F:\CMIP6_SIday\regional_sic_results\GRAND_TOTAL_CLIMATOLOGY_both_regions_1995_2100.csv"
)

# 实测海冰密集度逐日数据路径与叠加年份范围
OBS_DAILY_CSV = Path(r"G:\E\2024seaice\daily_mean_sic_1979_2024.csv")
OBS_YEAR_RANGE = (1995, 2024)  # 若仅需严格画在 Historical 分界线内，可改为 (1995, 2014)

OUT_DIR = Path(r"F:\CMIP6_SIday\regional_sic_results\figures_full_year_1995_2100")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 无冰阈值（自动识别 0~1 或 0~100）
SIC_FREE_PERCENT = 1.0

# 1–12 月完整全年窗口 (共 365 天)
TARGET_MONTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
MAX_POSSIBLE_DAYS = 365

# 是否规避“上一年秋冬季延迟结冰留在1月初的残余无冰日”
SKIP_WINTER_RESIDUAL_WATER = True

# 阴影区间类型: "iqr" (25%~75%), "p10_p90" (10%~90%), "std" (Mean ± 1 SD)
SPREAD_TYPE = "iqr"

# 滑动平均窗口（年）
ROLLING_WINDOW = 5

# 至少有几个模式在该年出现无冰才计算集合平均开始日期
MIN_MODELS_FOR_ONSET = 2

# 实测曲线样式（亮紫色，置于顶层以便与历史时期形成清晰对比）
OBS_STYLE = {
    "label": f"Observed ({OBS_YEAR_RANGE[0]}–{OBS_YEAR_RANGE[1]})",
    "color": "#9333EA",       # 亮紫色
    "zorder": 9,
}

# 配色方案（主线高对比深色 + 柔和阴影填充）
SCENARIO_STYLES = {
    "historical": {
        "label": "Historical (1995–2014)",
        "color": "#555A5E",       # 深岩石灰
        "fill_color": "#9EA4A8",
        "zorder": 6,
    },
    "ssp126": {
        "label": "SSP1-2.6",
        "color": "#18648C",       # 极地钢蓝
        "fill_color": "#74ADD1",
        "zorder": 3,
    },
    "ssp245": {
        "label": "SSP2-4.5",
        "color": "#147864",       # 松石青绿
        "fill_color": "#66C2A5",
        "zorder": 4,
    },
    "ssp370": {
        "label": "SSP3-7.0",
        "color": "#C67915",       # 暖赭石金
        "fill_color": "#EEAC4B",
        "zorder": 5,
    },
    "ssp585": {
        "label": "SSP5-8.5",
        "color": "#A3322B",       # 赤陶砖红
        "fill_color": "#E27D76",
        "zorder": 6,
    },
}

SCENARIO_ORDER = ["historical", "ssp126", "ssp245", "ssp370", "ssp585"]

# 出版级全局字体、高清预览与刻度样式
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 600


# ============================================================
# 2. 核心统计函数（1–12月全年窗口，0 = Jan 01）
# ============================================================

META_AND_STAT_COLS = {
    "region", "scenario", "period", "year", "month", "day", "month_day",
    "model_count", "ensemble_mean", "ensemble_std", "ensemble_p10", "ensemble_p90",
    "Date", "date", "plot_date",
}


def get_model_columns(df):
    return [c for c in df.columns if c not in META_AND_STAT_COLS]


def detect_threshold(df, model_cols):
    sample_max = np.nanmax(df[model_cols].to_numpy(dtype=np.float64))
    if sample_max <= 1.5:
        return SIC_FREE_PERCENT / 100.0
    return SIC_FREE_PERCENT


def offset_to_mmdd_jan_base(offset_val):
    if not np.isfinite(offset_val):
        return None
    clamped = int(np.clip(round(offset_val), 0, 364))
    dt = pd.Timestamp(2001, 1, 1) + pd.Timedelta(days=clamped)
    return dt.strftime("%m-%d")


def find_first_ice_free_day_full_year(df_sub, col_name, threshold):
    s = df_sub[col_name]
    if s.notna().sum() < 30:
        return np.nan

    is_free = s < threshold
    cond_3days = (
        is_free
        & is_free.shift(-1, fill_value=False)
        & is_free.shift(-2, fill_value=False)
    )

    if not cond_3days.any():
        return np.nan

    if SKIP_WINTER_RESIDUAL_WATER:
        winter_spring_mask = df_sub["month"].isin([1, 2, 3, 4, 5])
        has_ice_in_winter = (~is_free) & winter_spring_mask & s.notna()
        if has_ice_in_winter.any():
            winter_sic = s.where(winter_spring_mask, np.nan)
            peak_pos = int(np.nanargmax(winter_sic.to_numpy()))
            search_cond = cond_3days.copy()
            search_cond.iloc[:peak_pos] = False
            if search_cond.any():
                cond_3days = search_cond
            else:
                return np.nan

    first_pos = int(np.flatnonzero(cond_3days.to_numpy())[0])
    m = int(df_sub.iloc[first_pos]["month"])
    d = int(df_sub.iloc[first_pos]["day"])

    if m == 2 and d == 29:
        d = 28

    offset_days = (pd.Timestamp(2001, m, d) - pd.Timestamp(2001, 1, 1)).days
    return float(offset_days)


def compute_obs_annual_metrics(obs_csv_path):
    """读取实测海冰密集度数据并计算逐年无冰天数与首次无冰日（与 CMIP6 坐标基准一致）"""
    if not obs_csv_path.exists():
        print(f"[WARN] 未找到实测数据文件，将跳过实测曲线绘制: {obs_csv_path}")
        return pd.DataFrame()

    print(f"正在读取实测海冰密集度数据:\n  {obs_csv_path}")
    df = pd.read_csv(obs_csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    full_date_index = pd.date_range(start="1979-01-01", end="2024-12-31", freq="D")
    df = df.set_index("Date").reindex(full_date_index).rename_axis("Date").reset_index()

    obs_cols = ["MeanSIC_West", "MeanSIC_East"]
    df[obs_cols] = df[obs_cols].interpolate(method="linear", limit=3)
    threshold = detect_threshold(df, obs_cols)

    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    df = df[df["month"].isin(TARGET_MONTHS)].copy()

    region_map = {
        "usiceshp_west": "MeanSIC_West",
        "useiceshp2": "MeanSIC_East",
    }

    records = []
    for reg_key, col_name in region_map.items():
        for yr, grp in df.groupby("year", sort=True):
            grp = grp.sort_values("Date").reset_index(drop=True)
            s = grp[col_name]
            if s.notna().sum() >= 100:
                days_val = min(float((s < threshold).sum()), float(MAX_POSSIBLE_DAYS))
            else:
                days_val = np.nan

            onset_val = find_first_ice_free_day_full_year(grp, col_name, threshold)
            records.append({
                "region": reg_key,
                "year": int(yr),
                "days_mean": days_val,
                "onset_mean": onset_val,
            })

    obs_df = pd.DataFrame(records)
    # 先在完整序列上计算滑动平均，避免 1995 起始端边界截断偏差，再截取目标年份段
    for reg_key in region_map:
        mask = obs_df["region"] == reg_key
        obs_df.loc[mask, "days_smooth"] = (
            obs_df.loc[mask, "days_mean"]
            .rolling(ROLLING_WINDOW, center=True, min_periods=1)
            .mean()
            .to_numpy()
        )
        obs_df.loc[mask, "onset_smooth"] = (
            obs_df.loc[mask, "onset_mean"]
            .rolling(ROLLING_WINDOW, center=True, min_periods=1)
            .mean()
            .to_numpy()
        )

    obs_df = obs_df[
        (obs_df["year"] >= OBS_YEAR_RANGE[0]) & (obs_df["year"] <= OBS_YEAR_RANGE[1])
    ].reset_index(drop=True)
    return obs_df


def compute_annual_metrics_from_grand_daily(grand_csv_path):
    if not grand_csv_path.exists():
        raise FileNotFoundError(f"未找到逐日总表: {grand_csv_path}")

    print(f"正在读取双区域合并逐日总表:\n  {grand_csv_path}")
    df = pd.read_csv(grand_csv_path)

    model_cols = get_model_columns(df)
    threshold = detect_threshold(df, model_cols)
    print(f"  -> 识别到 {len(model_cols)} 个模式: {model_cols}")
    print(f"  -> 自动匹配 {SIC_FREE_PERCENT}% 无冰阈值为: {threshold}")

    df = df[df["month"].isin(TARGET_MONTHS)].copy()
    df = df.sort_values(["region", "scenario", "year", "month", "day"]).reset_index(drop=True)

    df[model_cols] = (
        df.groupby(["region", "scenario", "year"])[model_cols]
        .transform(lambda x: x.interpolate(method="linear", limit=3))
    )

    records = []

    for (reg, scen, yr), grp in df.groupby(["region", "scenario", "year"], sort=True):
        if scen not in SCENARIO_STYLES:
            continue

        yr = int(yr)
        if scen == "historical" and not (1995 <= yr <= 2014):
            continue
        if scen != "historical" and not (2015 <= yr <= 2100):
            continue

        grp = grp.reset_index(drop=True)

        days_per_model = {}
        onset_per_model = {}

        for m in model_cols:
            s_m = grp[m]
            if s_m.notna().sum() >= 100:
                raw_days = float((s_m < threshold).sum())
                days_per_model[m] = min(raw_days, float(MAX_POSSIBLE_DAYS))
            else:
                days_per_model[m] = np.nan

            onset_per_model[m] = find_first_ice_free_day_full_year(grp, m, threshold)

        days_arr = np.array(list(days_per_model.values()), dtype=np.float64)
        onset_arr = np.array(list(onset_per_model.values()), dtype=np.float64)

        valid_days = days_arr[np.isfinite(days_arr)]
        valid_onset = onset_arr[np.isfinite(onset_arr)]

        onset_mean_val = (
            np.nanmean(valid_onset) if len(valid_onset) >= MIN_MODELS_FOR_ONSET else np.nan
        )

        rec = {
            "region": reg,
            "scenario": scen,
            "year": yr,
            "days_model_count": len(valid_days),
            "days_mean": np.nanmean(valid_days) if len(valid_days) > 0 else np.nan,
            "days_std": np.nanstd(valid_days, ddof=1) if len(valid_days) > 1 else 0.0,
            "days_p10": np.nanpercentile(valid_days, 10) if len(valid_days) > 0 else np.nan,
            "days_p25": np.nanpercentile(valid_days, 25) if len(valid_days) > 0 else np.nan,
            "days_p75": np.nanpercentile(valid_days, 75) if len(valid_days) > 0 else np.nan,
            "days_p90": np.nanpercentile(valid_days, 90) if len(valid_days) > 0 else np.nan,
            "onset_model_count": len(valid_onset),
            "onset_mean": onset_mean_val,
            "onset_mean_date": offset_to_mmdd_jan_base(onset_mean_val),
            "onset_std": np.nanstd(valid_onset, ddof=1) if len(valid_onset) > 1 else 0.0,
            "onset_p10": np.nanpercentile(valid_onset, 10) if len(valid_onset) >= MIN_MODELS_FOR_ONSET else np.nan,
            "onset_p25": np.nanpercentile(valid_onset, 25) if len(valid_onset) >= MIN_MODELS_FOR_ONSET else np.nan,
            "onset_p75": np.nanpercentile(valid_onset, 75) if len(valid_onset) >= MIN_MODELS_FOR_ONSET else np.nan,
            "onset_p90": np.nanpercentile(valid_onset, 90) if len(valid_onset) >= MIN_MODELS_FOR_ONSET else np.nan,
        }

        for m in model_cols:
            rec[f"{m}_IceFreeDays"] = days_per_model[m]
            rec[f"{m}_OnsetIndex"] = onset_per_model[m]
            rec[f"{m}_OnsetDate"] = offset_to_mmdd_jan_base(onset_per_model[m])

        records.append(rec)

    return pd.DataFrame(records)


def compute_climatology_metrics_from_grand_clim(grand_clim_path):
    if not grand_clim_path.exists():
        return pd.DataFrame()

    print(f"正在读取双区域合并气候态总表:\n  {grand_clim_path}")
    df_c = pd.read_csv(grand_clim_path)
    model_cols = get_model_columns(df_c)
    threshold = detect_threshold(df_c, model_cols)

    df_c = df_c[df_c["month"].isin(TARGET_MONTHS)].copy()
    df_c = df_c.sort_values(["region", "scenario", "period", "month", "day"]).reset_index(drop=True)

    clim_records = []
    for (reg, scen, prd), grp in df_c.groupby(["region", "scenario", "period"], sort=False):
        grp = grp.reset_index(drop=True)
        days_list = []
        onset_list = []

        for m in model_cols:
            s_m = grp[m]
            if s_m.notna().sum() >= 100:
                days_list.append(min(float((s_m < threshold).sum()), float(MAX_POSSIBLE_DAYS)))
            onset_val = find_first_ice_free_day_full_year(grp, m, threshold)
            if np.isfinite(onset_val):
                onset_list.append(onset_val)

        days_arr = np.array(days_list, dtype=np.float64)
        onset_arr = np.array(onset_list, dtype=np.float64)
        onset_mean_val = np.nanmean(onset_arr) if len(onset_arr) > 0 else np.nan

        clim_records.append({
            "region": reg,
            "scenario": scen,
            "period": prd,
            "days_mean": np.nanmean(days_arr) if len(days_arr) > 0 else np.nan,
            "days_p25": np.nanpercentile(days_arr, 25) if len(days_arr) > 0 else np.nan,
            "days_p75": np.nanpercentile(days_arr, 75) if len(days_arr) > 0 else np.nan,
            "onset_mean": onset_mean_val,
            "onset_mean_date": offset_to_mmdd_jan_base(onset_mean_val),
            "onset_p25": np.nanpercentile(onset_arr, 25) if len(onset_arr) > 0 else np.nan,
            "onset_p75": np.nanpercentile(onset_arr, 75) if len(onset_arr) > 0 else np.nan,
        })

    return pd.DataFrame(clim_records)


def get_spread_bounds(df_scen, prefix):
    mean = df_scen[f"{prefix}_mean"].to_numpy(dtype=np.float64)
    if SPREAD_TYPE == "iqr":
        return (
            df_scen[f"{prefix}_p25"].to_numpy(dtype=np.float64),
            df_scen[f"{prefix}_p75"].to_numpy(dtype=np.float64),
        )
    elif SPREAD_TYPE == "p10_p90":
        return (
            df_scen[f"{prefix}_p10"].to_numpy(dtype=np.float64),
            df_scen[f"{prefix}_p90"].to_numpy(dtype=np.float64),
        )
    else:
        std = df_scen[f"{prefix}_std"].to_numpy(dtype=np.float64)
        return mean - std, mean + std


# ============================================================
# 3. 绘制单个区域的上下分层双 Y 轴子图（主图级清晰度优化）
# ============================================================

def plot_region_panel(
    ax1,
    df_reg,
    panel_letter,
    panel_title,
    accent_color,
    show_top_period_labels=False,
    df_obs_reg=None,
):
    ax2 = ax1.twinx()

    hist_df = (
        df_reg[df_reg["scenario"] == "historical"]
        .sort_values("year")
        .reset_index(drop=True)
    )
    last_hist_row = hist_df.iloc[[-1]] if not hist_df.empty else pd.DataFrame()

    # 历史时期 (1995–2014) 淡淡的背景底色衬托，替代原本杂乱的柱状图
    ax1.axvspan(1994.2, 2014.5, color="#F3F4F6", alpha=0.65, zorder=0)

    # 循环绘制 Historical + 4 个 SSP 情景
    for idx, scen in enumerate(SCENARIO_ORDER):
        sub = (
            df_reg[df_reg["scenario"] == scen]
            .sort_values("year")
            .reset_index(drop=True)
        )
        if sub.empty:
            continue

        # 衔接 2014->2015
        if scen != "historical" and not last_hist_row.empty:
            sub = pd.concat([last_hist_row, sub], ignore_index=True)

        years = sub["year"].to_numpy(dtype=np.float64)
        color = SCENARIO_STYLES[scen]["color"]
        fill_color = SCENARIO_STYLES[scen]["fill_color"]
        z = SCENARIO_STYLES[scen]["zorder"]

        # ----------------------------------------------------
        # A. 左轴 (下半区): 1–12月无冰总天数 (0 ~ 365 d)
        # ----------------------------------------------------
        days_mean = sub["days_mean"].to_numpy(dtype=np.float64)
        days_low, days_high = get_spread_bounds(sub, "days")

        days_low = np.clip(days_low, 0.0, float(MAX_POSSIBLE_DAYS))
        days_high = np.clip(days_high, 0.0, float(MAX_POSSIBLE_DAYS))

        days_mean_smooth = (
            pd.Series(days_mean)
            .rolling(ROLLING_WINDOW, center=True, min_periods=1)
            .mean()
            .to_numpy()
        )
        days_low_smooth = (
            pd.Series(days_low)
            .rolling(ROLLING_WINDOW, center=True, min_periods=1)
            .mean()
            .to_numpy()
        )
        days_high_smooth = (
            pd.Series(days_high)
            .rolling(ROLLING_WINDOW, center=True, min_periods=1)
            .mean()
            .to_numpy()
        )

        # 纯净半透明区间填充（去掉多余边界点线和未平滑噪点细线）
        ax1.fill_between(
            years,
            days_low_smooth,
            days_high_smooth,
            facecolor=fill_color,
            alpha=0.20 if scen != "historical" else 0.25,
            edgecolor="none",
            zorder=z - 1,
        )

        # 滑动平均主实线
        ax1.plot(
            years,
            days_mean_smooth,
            color=color,
            linewidth=2.2 if scen != "historical" else 2.5,
            linestyle="-",
            zorder=z + 1,
        )

        # ----------------------------------------------------
        # B. 右轴 (上半区): 1–12月首次连续3天无冰日期
        # ----------------------------------------------------
        onset_mean = sub["onset_mean"].to_numpy(dtype=np.float64)
        onset_low, onset_high = get_spread_bounds(sub, "onset")

        onset_low = np.clip(onset_low, 0.0, 364.0)
        onset_high = np.clip(onset_high, 0.0, 364.0)

        onset_mean_smooth = (
            pd.Series(onset_mean)
            .rolling(ROLLING_WINDOW, center=True, min_periods=1)
            .mean()
            .to_numpy()
        )
        onset_low_smooth = (
            pd.Series(onset_low)
            .rolling(ROLLING_WINDOW, center=True, min_periods=1)
            .mean()
            .to_numpy()
        )
        onset_high_smooth = (
            pd.Series(onset_high)
            .rolling(ROLLING_WINDOW, center=True, min_periods=1)
            .mean()
            .to_numpy()
        )

        ax2.fill_between(
            years,
            onset_low_smooth,
            onset_high_smooth,
            facecolor=fill_color,
            alpha=0.18 if scen != "historical" else 0.24,
            edgecolor="none",
            zorder=z - 1,
        )

        # 错峰打空心圆点（每隔8年打一个点，不同情景错开2年，防止重合期圆点叠在一起）
        ax2.plot(
            years,
            onset_mean_smooth,
            color=color,
            linewidth=2.0 if scen != "historical" else 2.3,
            linestyle="--",
            marker="o",
            markevery=(idx * 2, 8),
            markersize=4.5,
            markerfacecolor="white",
            markeredgewidth=1.3,
            markeredgecolor=color,
            zorder=z + 2,
        )

    # --------------------------------------------------------
    # C. 叠加实测海冰数据平均曲线（亮紫色）
    # --------------------------------------------------------
    if df_obs_reg is not None and not df_obs_reg.empty:
        obs_sub = df_obs_reg.sort_values("year").reset_index(drop=True)
        obs_years = obs_sub["year"].to_numpy(dtype=np.float64)
        obs_days_smooth = np.clip(
            obs_sub["days_smooth"].to_numpy(dtype=np.float64),
            0.0,
            float(MAX_POSSIBLE_DAYS),
        )
        obs_onset_smooth = np.clip(
            obs_sub["onset_smooth"].to_numpy(dtype=np.float64),
            0.0,
            364.0,
        )

        # 下半轨（左轴）：实测无冰总天数实线
        ax1.plot(
            obs_years,
            obs_days_smooth,
            color=OBS_STYLE["color"],
            linewidth=2.5,
            linestyle="-",
            zorder=OBS_STYLE["zorder"],
        )

        # 上半轨（右轴）：实测首次无冰日期虚线 + 空心圆点
        ax2.plot(
            obs_years,
            obs_onset_smooth,
            color=OBS_STYLE["color"],
            linewidth=2.3,
            linestyle="--",
            marker="o",
            markevery=(1, 6),
            markersize=4.5,
            markerfacecolor="white",
            markeredgewidth=1.3,
            markeredgecolor=OBS_STYLE["color"],
            zorder=OBS_STYLE["zorder"] + 1,
        )

    # ========================================================
    # 3. 拓宽中央安全隔离带的双 Y 轴设置 (总跨度均为 830 天)
    #    - 左轴下半区: 0 ~ 365 d (占高度 0% ~ 44%)
    #    - 中央隔离带: 左轴 365 ~ 465 (占高度 44% ~ 56%，曲线绝不相交)
    #    - 右轴上半区: Jan 01 (0) ~ Nov 01 (304) (占高度 56% ~ 100%)
    # ========================================================

    # 左轴 (下半区: 0 ~ 365 天)
    ax1.set_ylim(0, 830)
    left_yticks = [0, 90, 180, 270, 365]
    ax1.set_yticks(left_yticks)
    ax1.set_yticklabels([str(v) for v in left_yticks], fontsize=10.5, color="#1f2937")
    ax1.set_ylabel(
        "Ice-Free Season Length (d)",
        fontsize=11.5,
        fontweight="bold",
        color="#1f2937",
        labelpad=6,
    )

    # 右轴 (上半区: 精简为双月刻度，消除 12 个月份挤在一起的问题)
    ax2.set_ylim(-465, 365)
    right_yticks = [0, 59, 120, 181, 243, 304]
    right_yticklabels = ["Jan 01", "Mar 01", "May 01", "Jul 01", "Sep 01", "Nov 01"]
    ax2.set_yticks(right_yticks)
    ax2.set_yticklabels(right_yticklabels, fontsize=10.5, color="#374151")
    ax2.set_ylabel(
        "First Ice-Free Date",
        fontsize=11.5,
        fontweight="bold",
        color="#374151",
        labelpad=6,
    )

    # 中央隔离带与上下轨道边界线
    ax1.axhspan(380, 450, color="#F8FAFC", alpha=0.9, zorder=2)
    ax1.axhline(y=415, color="#64748B", linestyle="--", linewidth=1.0, alpha=0.8, zorder=3)

    # 水平辅助网格线（减淡且与精简后的刻度严格对齐）
    for y_val in [90, 180, 270, 365]:
        ax1.axhline(y=y_val, color="#CBD5E1", linestyle=":", linewidth=0.6, alpha=0.6, zorder=0)
    for y_val in [0, 59, 120, 181, 243, 304]:
        ax2.axhline(y=y_val, color="#CBD5E1", linestyle=":", linewidth=0.6, alpha=0.6, zorder=0)

    # ========================================================
    # 4. 2014/2015 分界线及 20 年时期分界线
    # ========================================================
    ax1.axvline(x=2014.5, color="#333333", linestyle="--", linewidth=1.2, alpha=0.85, zorder=7)
    for p_bound in [2034.5, 2054.5, 2074.5]:
        ax1.axvline(x=p_bound, color="#94A3B8", linestyle=":", linewidth=0.8, alpha=0.65, zorder=1)

    if show_top_period_labels:
        period_boxes = [
            (2004.5, "Historical\n(1995–2014)"),
            (2024.5, "2015–2034"),
            (2044.5, "2035–2054"),
            (2064.5, "2055–2074"),
            (2087.5, "2075–2100"),
        ]
        for x_mid, p_text in period_boxes:
            ax1.text(
                x_mid,
                842,
                p_text,
                ha="center",
                va="bottom",
                fontsize=10.0,
                fontweight="bold",
                color="#1f2937",
            )

    # ========================================================
    # 5. 主图规范子图编号 (a, b)、区域标签与轨道引导文字
    # ========================================================
    ax1.text(
        -0.055, 1.03, panel_letter,
        transform=ax1.transAxes,
        fontsize=15, fontweight="bold", color="black", va="bottom", ha="left"
    )

    bbox_props = dict(
        boxstyle="round,pad=0.30",
        fc="white",
        ec=accent_color,
        lw=1.8,
        alpha=0.95,
    )
    ax1.text(
        0.012,
        0.90,
        panel_title,
        transform=ax1.transAxes,
        fontsize=11.5,
        fontweight="bold",
        color=accent_color,
        bbox=bbox_props,
        zorder=10,
    )

    # 轨道引导文字放置在左侧 1995–2025 空白隔离带上下缘，加半透明白底防干扰
    track_bbox = dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.85)
    ax1.text(
        0.012,
        0.525,
        "▲ Upper Track: First Ice-Free Date (Right Axis, Dashed)",
        transform=ax1.transAxes,
        fontsize=9.0,
        fontweight="bold",
        color="#374151",
        bbox=track_bbox,
        zorder=8,
    )
    ax1.text(
        0.012,
        0.445,
        "▼ Lower Track: Ice-Free Season Length (Left Axis, Solid)",
        transform=ax1.transAxes,
        fontsize=9.0,
        fontweight="bold",
        color="#1f2937",
        bbox=track_bbox,
        zorder=8,
    )

    ax1.tick_params(axis="both", direction="in", length=4, width=0.9)
    ax2.tick_params(axis="y", direction="in", length=4, width=0.9)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)


# ============================================================
# 4. 主程序
# ============================================================

def main():
    all_stat_df = compute_annual_metrics_from_grand_daily(GRAND_DAILY_CSV)
    obs_stat_df = compute_obs_annual_metrics(OBS_DAILY_CSV)

    stat_csv_path = OUT_DIR / "CMIP6_FullYear_ice_free_days_and_onset_1995_2100.csv"
    all_stat_df.to_csv(stat_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] 1995–2100 (1–12月全年) 逐年无冰天数与开始日期统计表已保存至:\n  {stat_csv_path}")

    clim_stat_df = compute_climatology_metrics_from_grand_clim(GRAND_CLIM_CSV)
    if not clim_stat_df.empty:
        clim_csv_path = OUT_DIR / "CMIP6_FullYear_period_climatology_summary.csv"
        clim_stat_df.to_csv(clim_csv_path, index=False, encoding="utf-8-sig")
        print(f"[SAVE] 5个时期气候态汇总表已保存至:\n  {clim_csv_path}")

    df_west_stat = all_stat_df[all_stat_df["region"] == "usiceshp_west"].copy()
    df_east_stat = all_stat_df[all_stat_df["region"] == "useiceshp2"].copy()

    df_west_obs = (
        obs_stat_df[obs_stat_df["region"] == "usiceshp_west"].copy()
        if not obs_stat_df.empty else pd.DataFrame()
    )
    df_east_obs = (
        obs_stat_df[obs_stat_df["region"] == "useiceshp2"].copy()
        if not obs_stat_df.empty else pd.DataFrame()
    )

    # 紧凑高清晰度主图尺寸 (12.0 x 7.4)，确保缩印到期刊页面后字号依然饱满清晰
    print("\n正在绘制主图级 1995–2100 双区域四情景统计图...")
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 7.4), sharex=True)
    plt.subplots_adjust(hspace=0.10, top=0.85, bottom=0.09, left=0.07, right=0.92)

    # 上图 (a): West Region (统一使用代表 West 的暖橙色边框 #E67E22)
    plot_region_panel(
        ax1=axes[0],
        df_reg=df_west_stat,
        panel_letter="a",
        panel_title="West foraging ground",
        accent_color="#E67E22",
        show_top_period_labels=True,
        df_obs_reg=df_west_obs,
    )

    # 下图 (b): East Region (统一使用代表 East 的深蓝色边框 #2b6a99)
    plot_region_panel(
        ax1=axes[1],
        df_reg=df_east_stat,
        panel_letter="b",
        panel_title="East foraging ground",
        accent_color="#2b6a99",
        show_top_period_labels=False,
        df_obs_reg=df_east_obs,
    )

    # 顶部全局共享图例
    spread_label_map = {
        "iqr": "Shading: Interquartile Range (25th–75th)",
        "p10_p90": "Shading: 10th–90th Percentile",
        "std": "Shading: Ensemble Mean ± 1 SD",
    }

    legend_handles = [
        Line2D([0], [0], color= OBS_STYLE["color"], lw=2.5, label=OBS_STYLE["label"])
    ] + [
        Line2D([0], [0], color=SCENARIO_STYLES[s]["color"], lw=2.5, label=SCENARIO_STYLES[s]["label"])
        for s in SCENARIO_ORDER
    ]

    legend_handles.extend([
        Line2D(
            [0], [0],
            color="#333333", lw=2.2, linestyle="-",
            label="Ice-Free Season Length (5-yr smooth)",
        ),
        Line2D(
            [0], [0],
            color="#333333", lw=2.0, linestyle="--",
            marker="o", markersize=5.0, markerfacecolor="white",
            markeredgewidth=1.4, markeredgecolor="#333333",
            label="First Ice-Free Date (5-yr smooth)",
        ),
        Patch(
            facecolor="#9EA4A8", alpha=0.35, edgecolor="none",
            label=spread_label_map.get(SPREAD_TYPE, "Inter-Model Spread"),
        ),
    ])

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=5,
        frameon=False,
        fontsize=10.0,
        columnspacing=1.2,
        handlelength=2.2,
    )

    # X 轴刻度精简：每 10 年一个主刻度 + 起止年份 1995，每 5 年一个次刻度
    axes[-1].set_xlabel("Year", fontsize=12, fontweight="bold", labelpad=6)
    axes[-1].set_xlim(1994.2, 2100.8)

    major_years = [1995, 2000, 2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100]
    axes[-1].set_xticks(major_years)
    axes[-1].set_xticklabels([str(y) for y in major_years], fontsize=10.5)

    minor_years = list(range(1995, 2101, 5))
    axes[-1].set_xticks(minor_years, minor=True)
    axes[-1].tick_params(axis="x", which="minor", direction="in", length=2.5, width=0.8)
    axes[-1].tick_params(axis="x", which="major", direction="in", length=4.5, width=0.9)

    # 导出 600 DPI 超清 PNG 与 PDF 矢量图
    fig_png = OUT_DIR / f"CMIP6_FullYear_IceFreeDays_and_Onset_1995_2100_{SPREAD_TYPE}_600dpi.png"
    fig_pdf = OUT_DIR / f"CMIP6_FullYear_IceFreeDays_and_Onset_1995_2100_{SPREAD_TYPE}_vector.pdf"
    plt.savefig(fig_png, dpi=600, bbox_inches="tight")
    plt.savefig(fig_pdf, bbox_inches="tight")
    print(f"\n[SAVE] 600 DPI 主图已保存至:\n  {fig_png}\n  {fig_pdf}")

    plt.show()


if __name__ == "__main__":
    main()