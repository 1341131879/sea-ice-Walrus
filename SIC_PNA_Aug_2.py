
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from pathlib import Path
from tqdm import tqdm
import scipy.stats as stats
import pyproj

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']

# ============================================================
# 1. 参数与路径设置
# ============================================================

# ---------- 数据路径 ----------
ice_dir = Path(r"G:\E\2024seaice\osimonthlytif")
pna_file = Path(r"G:\E\2024seaice\2025\SST\PNA_monthly_normals.xlsx")

# ---------- 时间 ----------
TARGET_MONTH = 8
YEAR_MIN = 2002
YEAR_MAX = 2024
ORIGINAL_YEAR_MIN = 1979
BASELINE_START = 1980
BASELINE_END = 2010

# ---------- 白框目标区域 ----------
LAT_MIN = 71.0
LAT_MAX = 74.0
LON_MIN = 180.0
LON_MAX = 203.0

# ============================================================
# 2. 读取与处理 PNA 数据
# ============================================================

print("=" * 60)
print("1. 处理 PNA 数据")
print("=" * 60)

pna_df = pd.read_excel(pna_file)

month_map = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
    'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
    'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}

# 子图1需要1979–2024年PNA
pna_aug = (
    pna_df.melt(id_vars="Year", var_name="Month", value_name="PNA")
    .assign(Month=lambda d: d["Month"].map(month_map))
    .query("@ORIGINAL_YEAR_MIN <= Year <= @YEAR_MAX and Month == @TARGET_MONTH")
    .set_index("Year")["PNA"]
)

# 计算PNA基准期平均值及异常值
pna_baseline_mean = pna_aug.loc[BASELINE_START:BASELINE_END].mean()
pna_anomaly = pna_aug - pna_baseline_mean

# ============================================================
# 3. 提取目标区域的海冰平均值 (SIC)
# ============================================================

print("\n" + "=" * 60)
print("2. 提取白框区域 August SIC 平均值")
print("=" * 60)

# 1) 获取栅格网格坐标并生成经纬度掩码
sample_file = next(ice_dir.glob("osi_monthly_*.tif"))

with rasterio.open(sample_file) as src:
    width, height = src.width, src.height
    transform = src.transform

    # 保留原始坐标提取方式
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    X, Y = rasterio.transform.xy(transform, rows, cols)
    X, Y = np.array(X), np.array(Y)

# 坐标转换：EPSG:3413 -> EPSG:4326
print("正在计算研究区域的地理掩码...")

# 保留原始代码，不使用src.crs替代
transformer = pyproj.Transformer.from_crs(3413, 4326, always_xy=True)
lons, lats = transformer.transform(X, Y)

# 将经度转为0–360范围
lons_360 = np.mod(lons, 360)

# 生成区域掩码
region_mask = (
    (lats >= LAT_MIN) & (lats <= LAT_MAX) &
    (lons_360 >= LON_MIN) & (lons_360 <= LON_MAX)
)

print(f"目标区域内共包含 {np.sum(region_mask)} 个像元")

# 2) 循环读取1979–2024年8月SIC
tif_re = re.compile(r"osi_monthly_(\d{4})(\d{2})\.tif$", re.I)
sic_records = {}

for tif_path in tqdm(sorted(ice_dir.glob("osi_monthly_*.tif"))):
    m = tif_re.search(tif_path.name)
    if not m:
        continue

    year, month = int(m.group(1)), int(m.group(2))

    # 只修改起始年份，读取1979–2024年
    if not (ORIGINAL_YEAR_MIN <= year <= YEAR_MAX and month == TARGET_MONTH):
        continue

    with rasterio.open(tif_path) as src:
        data = src.read(1, masked=True)

        # 保留原始有效数据提取逻辑
        valid_region_data = data[region_mask & ~data.mask]

        if len(valid_region_data) > 0:
            sic_records[year] = np.mean(valid_region_data)
        else:
            sic_records[year] = np.nan

sic_series = pd.Series(sic_records).dropna()

# ============================================================
# 4. 数据对齐与异常值计算
# ============================================================

print("\n" + "=" * 60)
print("3. 计算异常值与去趋势 (Detrending)")
print("=" * 60)

# SIC基准期平均值：使用1980–2010年数据
sic_baseline_mean = sic_series.loc[
    BASELINE_START:BASELINE_END
].mean()

# ------------------------------------------------------------
# 子图1：1979–2024年原始异常值
# ------------------------------------------------------------

common_years_original = sorted(
    set(sic_series.loc[ORIGINAL_YEAR_MIN:YEAR_MAX].index)
    & set(pna_anomaly.index)
)

df_original = pd.DataFrame({
    'SIC': sic_series.loc[common_years_original],
    'PNA_Anom': pna_anomaly.loc[common_years_original]
}, index=common_years_original)

df_original['SIC_Anom'] = df_original['SIC'] - sic_baseline_mean

# 原始异常值相关性
r_orig, p_orig = stats.pearsonr(
    df_original['SIC_Anom'],
    df_original['PNA_Anom']
)

print(
    f"1979–2024年不去趋势相关系数: "
    f"r = {r_orig:.3f}, p = {p_orig:.4f}"
)

# ------------------------------------------------------------
# 子图2：2001–2024年去趋势异常值
# 保留原来的分析时间范围
# ------------------------------------------------------------

common_years = sorted(
    set(sic_series.loc[YEAR_MIN:YEAR_MAX].index)
    & set(pna_anomaly.loc[YEAR_MIN:YEAR_MAX].index)
)

df = pd.DataFrame({
    'SIC': sic_series.loc[common_years],
    'PNA_Anom': pna_anomaly.loc[common_years]
}, index=common_years)

# SIC异常值（1980–2010年基准期）
df['SIC_Anom'] = df['SIC'] - sic_baseline_mean

# 去趋势
years_array = df.index.values.astype(float)

# SIC去趋势
slope_sic, intercept_sic, _, _, _ = stats.linregress(
    years_array, df['SIC_Anom']
)

df['SIC_Anom_Detrended'] = (
    df['SIC_Anom'] - (slope_sic * years_array + intercept_sic)
)

# PNA去趋势
slope_pna, intercept_pna, _, _, _ = stats.linregress(
    years_array, df['PNA_Anom']
)

df['PNA_Anom_Detrended'] = (
    df['PNA_Anom'] - (slope_pna * years_array + intercept_pna)
)

print(f"SIC线性趋势: {slope_sic:.5f} / year")
print(f"PNA线性趋势: {slope_pna:.5f} / year")

# 计算去趋势前后的相关系数
r_det, p_det = stats.pearsonr(
    df['SIC_Anom_Detrended'],
    df['PNA_Anom_Detrended']
)

print(f"去趋势后相关系数: r = {r_det:.3f}, p = {p_det:.4f}")

# ============================================================
# 5. 开始绘图
# ============================================================

print("\n" + "=" * 60)
print("4. 绘制时间序列图")
print("=" * 60)

# 仅将sharex改为False，使两个子图的横轴独立
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(7, 7), sharex=False
)

import matplotlib.ticker as ticker

# ----------------- 子图1：1979–2024年原始异常值 -----------------

bar1 = ax1.bar(
    df_original.index,
    df_original['SIC_Anom'],
    color='gray',
    alpha=0.85,
    width=0.6,
    label='SIC Anomaly (left)'
)

ax1.set_ylabel(
    'SIC Anomaly',
    fontsize=12,
    color='gray',
    fontweight='bold'
)

ax1.tick_params(axis='y', labelcolor='gray', labelsize=13)
ax1.axhline(0, color='gray', linewidth=1, linestyle='--')

# PNA黑线，双Y轴
ax1_twin = ax1.twinx()

line1, = ax1_twin.plot(
    df_original.index,
    df_original['PNA_Anom'],
    color='#D94841',
    linewidth=1.5,
    label='PNA Anomaly (right)'
)

ax1_twin.set_ylabel(
    'PNA Anomaly',
    fontsize=12,
    color='#D94841',
    fontweight='bold'
)

ax1_twin.tick_params(axis='y', labelcolor='black', labelsize=13)
ax1_twin.invert_yaxis()

ax1.set_title(
    f'r = {r_orig:.2f}, p = {p_orig:.5f}',
    fontsize=16,
    pad=8
)

ax1.grid(axis='x', linestyle=':', alpha=0.6)

# 子图1独立横轴
ax1.set_xlim(1978, 2025)
ax1.set_xticks(np.arange(1979, 2025, 5))

# ----------------- 子图2：2001–2024年去趋势异常值 -----------------

bar2 = ax2.bar(
    df.index,
    df['SIC_Anom_Detrended'],
    color='#4B8BBE',
    alpha=0.85,
    width=0.6,
    label='Detrended SIC'
)

ax2.set_ylabel(
    'Detrended SIC Anomaly',
    fontsize=12,
    color='#2B5B84',
    fontweight='bold'
)

ax2.tick_params(axis='y', labelcolor='#2B5B84', labelsize=13)
ax2.axhline(0, color='gray', linewidth=1, linestyle='--')
ax2.tick_params(axis='x', labelcolor='black', labelsize=13)

# PNA黑线，双Y轴
ax2_twin = ax2.twinx()

line2, = ax2_twin.plot(
    df.index,
    df['PNA_Anom_Detrended'],
    color='black',
    linewidth=1.5,
    label='Detrended PNA'
)

ax2_twin.set_ylabel(
    'Detrended PNA Anomaly',
    fontsize=12,
    color='black',
    fontweight='bold'
)

ax2_twin.tick_params(axis='y', labelcolor='black', labelsize=13)
ax2_twin.invert_yaxis()

ax2.set_title(
    f'r = {r_det:.2f}, p = {p_det:.5f}',
    fontsize=16,
    pad=8
)

ax2.grid(axis='x', linestyle=':', alpha=0.6)

# 子图2独立横轴，保持2001–2024年
ax2.set_xlim(YEAR_MIN - 1, YEAR_MAX + 1)
ax2.set_xticks(np.arange(YEAR_MIN, YEAR_MAX + 1, 3))

# ----------------- 整体布局优化 -----------------

# 保留原来的图例设置
lines = [bar1, line1]
labels = [l.get_label() for l in lines]

fig.legend(
    lines,
    labels,
    loc='upper center',
    ncol=2,
    fontsize=11,
    bbox_to_anchor=(0.5, 0.98),
    frameon=False
)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()
