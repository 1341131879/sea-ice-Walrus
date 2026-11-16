import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import rasterio
from pathlib import Path

from tqdm import tqdm
import scipy.stats as stats

# ================= 1. 全局图表设置 =================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']
# ============================================================
# 0. 参数设置
# ============================================================

# ---------- 数据路径 ----------
ice_dir = Path(r"G:\E\2024seaice\osimonthlytif")
pna_file = Path(r"G:\E\2024seaice\2025\SST\PNA_monthly_normals.xlsx")


# ---------- 时间 ----------
TARGET_MONTH = 8

YEAR_MIN = 2002
YEAR_MAX = 2024


# ---------- 气候基准期 ----------
BASELINE_START = 1980
BASELINE_END = 2010


# ---------- 显著性 ----------
SIG_LEVEL = 0.05


# ============================================================
# 1. 读取 PNA 数据
# ============================================================

print("=" * 60)
print("1. 读取 PNA 数据")
print("=" * 60)
pna_df = pd.read_excel(pna_file)
# ------------------------------------------------------------
# 将月份转换成数字
# ------------------------------------------------------------

month_map = {
    'Jan': 1,
    'Feb': 2,
    'Mar': 3,
    'Apr': 4,
    'May': 5,
    'Jun': 6,
    'Jul': 7,
    'Aug': 8,
    'Sep': 9,
    'Oct': 10,
    'Nov': 11,
    'Dec': 12
}

pna_aug = (
    pna_df
    .melt(
        id_vars="Year",
        var_name="Month",
        value_name="PNA"
    )
    .assign(
        Month=lambda d: d["Month"].map(month_map)
    )
    .query(
        "@YEAR_MIN <= Year <= @YEAR_MAX "
        "and Month == @TARGET_MONTH"
    )
    .set_index("Year")["PNA"]
)


print(
    f"PNA 数据年份："
    f"{pna_aug.index.min()}–{pna_aug.index.max()}"
)

print(
    f"August PNA 样本数："
    f"{len(pna_aug)}"
)

# ============================================================
# 2. 计算 PNA anomaly
#    基准期：1980–2010
# ============================================================

print("\n" + "=" * 60)
print("2. 计算 PNA anomaly")
print("=" * 60)

pna_baseline = pna_aug.loc[
    BASELINE_START:BASELINE_END
]

pna_climatology = pna_baseline.mean()

pna_anomaly = (
    pna_aug
    - pna_climatology
)

print(
    f"{BASELINE_START}–{BASELINE_END} "
    f"August PNA climatology = "
    f"{pna_climatology:.4f}"
)

# ============================================================
# 3. 读取 August SIC
# ============================================================

print("\n" + "=" * 60)
print("3. 读取 August SIC 数据")
print("=" * 60)


tif_re = re.compile(
    r"osi_monthly_(\d{4})(\d{2})\.tif$",
    re.I
)

all_ice_data = []
years_list = []
# ------------------------------------------------------------
# 找到一个文件作为空间模板
# ------------------------------------------------------------

sample_file = next(
    ice_dir.glob("osi_monthly_*.tif")
)

with rasterio.open(sample_file) as src:

    transform = src.transform
    height = src.height
    width = src.width
    raster_crs = src.crs
    bounds = src.bounds
    resolution = src.res

print(f"CRS        = {raster_crs}")
print(f"Shape      = {height} × {width}")
print(f"Resolution = {resolution}")
print(f"Bounds     = {bounds}")


# ------------------------------------------------------------
# 读取 August 文件
# ------------------------------------------------------------

for tif_path in tqdm(
    sorted(
        ice_dir.glob("osi_monthly_*.tif")
    )
):

    m = tif_re.search(
        tif_path.name
    )

    if m is None:
        continue


    year = int(m.group(1))
    month = int(m.group(2))


    # 只保留 1979–2024 年 August
    if not (
        YEAR_MIN <= year <= YEAR_MAX
        and month == TARGET_MONTH
    ):
        continue


    with rasterio.open(tif_path) as src:

        data = src.read(
            1,
            masked=True
        )

        all_ice_data.append(data)
        years_list.append(year)


# ------------------------------------------------------------
# 堆叠
# ------------------------------------------------------------

ice_cube = np.ma.stack(
    all_ice_data
)

years_array = np.array(
    years_list
)


print(
    f"\n读取到 "
    f"{len(years_array)} 个 August SIC 文件"
)


# ============================================================
# 4. 匹配 SIC 与 PNA 的共同年份
# ============================================================

print("\n" + "=" * 60)
print("4. 匹配共同年份")
print("=" * 60)


common_years = sorted(
    set(years_array)
    &
    set(pna_aug.index)
)


print(
    f"共同年份数量："
    f"{len(common_years)}"
)

print(
    f"共同年份范围："
    f"{min(common_years)}–{max(common_years)}"
)


# ------------------------------------------------------------
# 按年份提取 SIC
# ------------------------------------------------------------

year_indices = [
    np.where(
        years_array == year
    )[0][0]
    for year in common_years
]


ice_cube_common = ice_cube[
    year_indices
]


common_years = np.array(
    common_years,
    dtype=int
)


# ============================================================
# 5. 计算 SIC anomaly
#    基准期：1980–2010
# ============================================================

print("\n" + "=" * 60)
print("5. 计算 SIC anomaly")
print("=" * 60)


baseline_indices = np.where(
    (
        common_years
        >= BASELINE_START
    )
    &
    (
        common_years
        <= BASELINE_END
    )
)[0]


print(
    f"基准期共同年份数量："
    f"{len(baseline_indices)}"
)


# ------------------------------------------------------------
# 1980–2010 August SIC climatology
# ------------------------------------------------------------

sic_climatology = np.ma.mean(
    ice_cube_common[
        baseline_indices
    ],
    axis=0
)


# ------------------------------------------------------------
# SIC anomaly
# ------------------------------------------------------------

sic_anomaly = (
    ice_cube_common
    - sic_climatology
)


# ============================================================
# 6. 获取 PNA anomaly
# ============================================================

pna_values = pna_anomaly.loc[
    common_years
].values.astype(float)


# ============================================================
# 7. 对 PNA anomaly 去趋势
# ============================================================

print("\n" + "=" * 60)
print("6. PNA anomaly 去趋势")
print("=" * 60)


years_float = (
    common_years.astype(float)
)


# ------------------------------------------------------------
# 线性趋势
# ------------------------------------------------------------

pna_slope, pna_intercept, _, _, _ = (
    stats.linregress(
        years_float,
        pna_values
    )
)


pna_trend = (
    pna_slope
    * years_float
    +
    pna_intercept
)


pna_detrended = (
    pna_values
    -
    pna_trend
)


print(
    f"PNA anomaly trend = "
    f"{pna_slope:.6f} / year"
)


# ============================================================
# 8. 计算每个格点 SIC–PNA 去趋势相关
# ============================================================

print("\n" + "=" * 60)
print("7. 计算 SIC–PNA 去趋势相关")
print("=" * 60)


correlation_map = np.full(
    (height, width),
    np.nan,
    dtype=np.float32
)


pvalue_map = np.full(
    (height, width),
    np.nan,
    dtype=np.float32
)


# ------------------------------------------------------------
# 循环格点
# ------------------------------------------------------------

for i in tqdm(
    range(height)
):

    for j in range(width):

        # ----------------------------------------------------
        # SIC 时间序列
        # ----------------------------------------------------

        ice_series = (
            sic_anomaly[
                :,
                i,
                j
            ]
        )


        # ----------------------------------------------------
        # Mask 处理
        # ----------------------------------------------------

        if np.ma.isMaskedArray(
            ice_series
        ):

            mask = np.ma.getmaskarray(
                ice_series
            )

            valid_mask = ~mask

        else:

            valid_mask = np.isfinite(
                ice_series
            )


        # ----------------------------------------------------
        # 加入 PNA 的有效性
        # ----------------------------------------------------

        valid_mask = (
            valid_mask
            &
            np.isfinite(
                pna_detrended
            )
        )


        # 至少 10 年
        if np.sum(valid_mask) < 10:
            continue


        valid_sic = np.asarray(
            ice_series[
                valid_mask
            ],
            dtype=float
        )


        valid_pna = np.asarray(
            pna_detrended[
                valid_mask
            ],
            dtype=float
        )


        valid_years = (
            years_float[
                valid_mask
            ]
        )


        # ----------------------------------------------------
        # 再次去除 NaN / Inf
        # ----------------------------------------------------

        finite_mask = (
            np.isfinite(valid_sic)
            &
            np.isfinite(valid_pna)
        )


        valid_sic = (
            valid_sic[
                finite_mask
            ]
        )

        valid_pna = (
            valid_pna[
                finite_mask
            ]
        )

        valid_years = (
            valid_years[
                finite_mask
            ]
        )


        if len(valid_sic) < 10:
            continue


        # ----------------------------------------------------
        # 常数序列检查
        # ----------------------------------------------------

        if (
            np.std(valid_sic) == 0
            or
            np.std(valid_pna) == 0
        ):
            continue


        # ====================================================
        # SIC anomaly 去趋势
        # ====================================================

        try:

            sic_slope, sic_intercept, _, _, _ = (
                stats.linregress(
                    valid_years,
                    valid_sic
                )
            )


            sic_trend = (
                sic_slope
                * valid_years
                +
                sic_intercept
            )


            sic_detrended = (
                valid_sic
                -
                sic_trend
            )


            # ------------------------------------------------
            # 去趋势后检查
            # ------------------------------------------------

            if np.std(
                sic_detrended
            ) == 0:

                continue


            # ------------------------------------------------
            # Pearson correlation
            # ------------------------------------------------

            r, p = stats.pearsonr(
                sic_detrended,
                valid_pna
            )


            correlation_map[
                i,
                j
            ] = r


            pvalue_map[
                i,
                j
            ] = p


        except Exception:

            continue

# ============================================================
# 9. 显著性筛选
# ============================================================

print("\n" + "=" * 60)
print("8. 显著性筛选（保留全图，提取显著性掩码）")
print("=" * 60)

# 保留完整的相关系数矩阵，这样全图都有颜色
sig_corr_map = correlation_map.copy()

# 仅提取一个布尔型掩码，用于稍后寻找打黑点的坐标
is_significant = (pvalue_map < SIG_LEVEL)

# 提取显著性掩码（用于提取打点的坐标）
is_significant = (pvalue_map < SIG_LEVEL)
# ============================================================
# 10. 统计结果
# ============================================================

valid_mask_stat = (
    np.isfinite(
        correlation_map
    )
)


valid_corr = (
    correlation_map[
        valid_mask_stat
    ]
)


valid_pvalue = (
    pvalue_map[
        valid_mask_stat
    ]
)


sig_mask_stat = (
    valid_pvalue
    <
    SIG_LEVEL
)


print(
    f"总格点数："
    f"{height * width}"
)


print(
    f"有效相关格点数："
    f"{len(valid_corr)}"
)


print(
    f"显著相关格点数 "
    f"(p < {SIG_LEVEL})："
    f"{np.sum(sig_mask_stat)}"
)


if len(valid_corr) > 0:

    print(
        f"平均相关系数："
        f"{np.mean(valid_corr):.3f}"
    )

    print(
        f"最小相关系数："
        f"{np.min(valid_corr):.3f}"
    )

    print(
        f"最大相关系数："
        f"{np.max(valid_corr):.3f}"
    )


if np.sum(sig_mask_stat) > 0:

    sig_values = (
        valid_corr[
            sig_mask_stat
        ]
    )

    print(
        f"显著区域平均相关系数："
        f"{np.mean(sig_values):.3f}"
    )


# ============================================================
# 11. 获取 EPSG:3413 像元中心坐标
# ============================================================

print("\n" + "=" * 60)
print("9. 生成栅格坐标")
print("=" * 60)


# ------------------------------------------------------------
# 注意：
# 这里与您原来能够正常出图的代码保持一致
# ------------------------------------------------------------

x = np.linspace(
    bounds.left,
    bounds.right,
    width
)


y = np.linspace(
    bounds.top,
    bounds.bottom,
    height
)


X, Y = np.meshgrid(
    x,
    y
)


print(
    f"X range: "
    f"{X.min()} ~ {X.max()}"
)


print(
    f"Y range: "
    f"{Y.min()} ~ {Y.max()}"
)


print(
    f"Significant pixels: "
    f"{np.sum(np.isfinite(sig_corr_map))}"
)


# ============================================================
# 12. 投影设置
# ============================================================

# ------------------------------------------------------------
# 原始海冰数据 CRS
# EPSG:3413
# ------------------------------------------------------------

ice_crs = ccrs.Stereographic(
    central_latitude=90,
    central_longitude=-45,
    true_scale_latitude=70
)


# ------------------------------------------------------------
# 绘图投影
#
# 与你已经正常工作的海冰边缘图完全一致
# ------------------------------------------------------------

plot_crs = ccrs.NorthPolarStereo(
    central_longitude=-169
)


# ------------------------------------------------------------
# 经纬度坐标系
# ------------------------------------------------------------

data_crs = ccrs.PlateCarree()


# ============================================================
# 13. 开始绘图
# ============================================================

print("\n" + "=" * 60)
print("10. 开始绘图")
print("=" * 60)


fig = plt.figure(
    figsize=(10, 9)
)


ax = plt.axes(
    projection=plot_crs
)


# ============================================================
# 14. 设置研究区域
# ============================================================

ax.set_extent(
    [
        175,
        205,
        63.3,
        76.7
    ],
    crs=data_crs
)


# ============================================================
# 15. 添加陆地
# ============================================================

ax.add_feature(
    cfeature.LAND,
    facecolor='#E6E6E6',
    zorder=2
)


# ============================================================
# 16. 添加海岸线
# ============================================================

ax.add_feature(
    cfeature.COASTLINE,
    linewidth=0.5,
    edgecolor='black',
    zorder=4
)
import matplotlib.colors as mcolors

import matplotlib.colors as mcolors
import matplotlib as mpl
import matplotlib.colors as mcolors

import matplotlib as mpl
import matplotlib.colors as mcolors

# ============================================================
# 17. 绘制相关系数及显著性斜线 (黑线白边双层叠加法)
# ============================================================

# ------------------------------------------------------------
# 0. 自定义色表
# ------------------------------------------------------------
# 自定义色表
custom_colors = [
    '#0000FF', '#0040FF', '#0080FF', '#00BFFF',
    '#00FFFF', '#7AFFFF', '#CFFFFF', '#FFFFFF',
    '#FFE0E0', '#FFB3B3', '#FF8080', '#FF4D4D',
    '#FF0000', '#CC0000', '#990033',
]

# 色阶边界：-0.8 到 0.8，共 30 个区间
levels = np.linspace(-0.8, 0.8, 31)

custom_cmap = mcolors.LinearSegmentedColormap.from_list(
    'custom_ncl',
    custom_colors,
    N=256
)

levels = np.linspace(-0.8, 0.8, 31)

pcm = ax.contourf(
    X,
    Y,
    sig_corr_map,
    levels=levels,
    cmap=custom_cmap,
    vmin=-0.8,
    vmax=0.8,
    transform=ice_crs,
    extend='neither',
    zorder=1
)
# ------------------------------------------------------------
# 2. 在显著区域绘制斜线 (双层叠加实现“黑线白边”)
# ------------------------------------------------------------
sig_matrix = is_significant.astype(int)

if np.sum(sig_matrix) > 0:
    # 控制斜线的密度：'/' 最稀疏, '//' 适中, '////' 很密
    hatch_density = '//'

    # ================== 第一层：粗的白色斜线 (充当白边) ==================
    mpl.rcParams['hatch.linewidth'] = 3.0  # 粗白线
    mpl.rcParams['hatch.color'] = 'white'

    ax.contourf(
        X,
        Y,
        sig_matrix,

        levels=[0.5, 1.5],
        colors='none',
        hatches=[hatch_density],

        transform=ice_crs,
        zorder=3
    )

    # ================== 第二层：细的黑色斜线 (充当主体) ==================
    mpl.rcParams['hatch.linewidth'] = 1.0  # 细黑线，叠在白线上
    mpl.rcParams['hatch.color'] = 'black'

    ax.contourf(
        X,
        Y,
        sig_matrix,

        levels=[0.5, 1.5],
        colors='none',
        hatches=[hatch_density],

        transform=ice_crs,
        zorder=4
    )

# ============================================================
# 18. 添加 74°N 参考线
# ============================================================

line_lons = np.linspace(
    170,
    210,
    200
)


line_lats = np.full_like(
    line_lons,
    74.0
)


ax.plot(
    line_lons,
    line_lats,

    transform=data_crs,

    color='#D8A31A',

    linewidth=4,

    linestyle='--',

    alpha=0.85,

    zorder=5
)
# ============================================================
# 18. 绘制目标研究区域的白色经纬度框
# ============================================================

# 定义纬度范围
box_lat_min, box_lat_max = 71.0, 74.0

# 定义经度范围
# 提示：203 代表 157°W (在 180° 的右边，且在你当前地图范围内)
# 如果你确定是 157°E，请将 203 改为 157，并扩大第 14 步的地图边界
box_lon_min, box_lon_max = 180.0, 203.0

# 为了在极地投影中画出完美的弧线，我们在四条边上分别生成 100 个密集的点
lons_bottom = np.linspace(box_lon_min, box_lon_max, 100)
lats_bottom = np.full_like(lons_bottom, box_lat_min)

lons_top = np.linspace(box_lon_min, box_lon_max, 100)
lats_top = np.full_like(lons_top, box_lat_max)

lats_left = np.linspace(box_lat_min, box_lat_max, 100)
lons_left = np.full_like(lats_left, box_lon_min)

lats_right = np.linspace(box_lat_min, box_lat_max, 100)
lons_right = np.full_like(lats_right, box_lon_max)

# 按顺序把四条边拼接起来，形成一个闭合的环：
# 下边缘(左->右) -> 右边缘(下->上) -> 上边缘(右->左) -> 左边缘(上->下)
box_lons = np.concatenate([
    lons_bottom,
    lons_right,
    lons_top[::-1],
    lons_left[::-1],
    [lons_bottom[0]]
])
box_lats = np.concatenate([
    lats_bottom,
    lats_right,
    lats_top[::-1],
    lats_left[::-1],
    [lats_bottom[0]]
])

# 绘制白框
ax.plot(
    box_lons,
    box_lats,
    transform=data_crs,  # data_crs 是 ccrs.PlateCarree()
    color='#333333',       # 框线颜色
    linewidth=4,       # 框线粗细 (觉得太粗可以调成 1.5 或 2.0)
    linestyle='-',       # 实线
    zorder=4            # 放在 10 这个极高的层级，确保它不会被数据、斜线或者海岸线遮挡
)
# ============================================================
# 19. 色条
# ============================================================
tick_values = np.linspace(-0.8, 0.8, 9)
tick_labels = [
    '-0.8', '-0.6', '-0.4', '-0.2',
    '0.0',
    '0.2', '0.4', '0.6', '0.8'
]

cbar = plt.colorbar(
    pcm,
    ax=ax,
    orientation='horizontal',
    shrink=0.75,
    pad=0.06,
    ticks=tick_values,
    spacing='uniform',
    extend='neither'
)

cbar.set_ticks(tick_values)
cbar.set_ticklabels(tick_labels)

cbar.outline.set_edgecolor('black')
cbar.outline.set_linewidth(0.8)

cbar.set_label(
    'Correlation coefficient (r)',
    fontsize=13
)

cbar.ax.tick_params(
    axis='x',
    labelsize=13,
    length=4,
    width=0.8,
    direction='out'
)
# 刻度数字大小
cbar.ax.tick_params(labelsize=13)
# cbar.ax.tick_params(
#     labelsize=11,
#     width=0.8,
#     length=4,
#     direction='in',
#     color='white'
#
# )

# ============================================================
# 20. 标题
# ============================================================

ax.set_title(
    'August SIC–PNA Correlation',
    fontsize=17,
    pad=15
)



# ============================================================
# 22. 显示
# ============================================================
print("Colorbar limits:", cbar.mappable.get_clim())
print("Colorbar ticks:", cbar.get_ticks())
print("Colorbar axis limits:", cbar.ax.get_xlim())
plt.tight_layout()

plt.show()