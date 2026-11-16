import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time

warnings.filterwarnings('ignore')

# ================= 1. 核心计算函数 =================
def haversine_vectorized_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# ================= 2. 基础设置与数据加载 =================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
sns.set_theme(style="whitegrid")

input_file = r'G:\E\Walrus\movebank\pacific walurs\pacificWalrus_USGS_ASC_argosGPS.csv'
shp_path = r'G:\E\Walrus\landshp\usedland.shp'
region_path = r'G:\E\Walrus\landshp\useiceshp2.shp'
freq = '6H'

print(f"[{time.strftime('%H:%M:%S')}] 1. 正在加载基础数据 (剔除指定年份)...")
df = pd.read_csv(input_file, low_memory=False)
id_col = 'Animal_ID' if 'Animal_ID' in df.columns else 'individual-local-identifier'
lat_col = 'location-lat' if 'location-lat' in df.columns else 'Latitude'
lon_col = 'location-long' if 'location-long' in df.columns else 'Longitude'

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['Year'] = df['timestamp'].dt.year

# 限制只保留 7月、8月 和 9月 (夏季)
df = df[df['timestamp'].dt.month.isin([7, 8, 9])].copy()

# 💡 核心修改：统一剔除 1988, 1991, 2002, 2004
exclude_years = []
df = df[~df['Year'].isin(exclude_years)].copy()

df[lon_col] = np.where(df[lon_col] < 0, df[lon_col] + 360, df[lon_col])

# ================= 3. 批量插值与速度计算 =================
grouped = df.groupby(['Year', id_col])
total_ids = len(grouped)
all_interpolated_data = []

print(f"[{time.strftime('%H:%M:%S')}] 2. 开始按 6H 频率插值并计算速度...")
for (year, aid), group in grouped:
    df_aid = group.drop_duplicates('timestamp').set_index('timestamp').sort_index()
    if len(df_aid) < 2: continue

    global_start = df_aid.index.min().floor('D')
    global_end = df_aid.index.max().ceil('D')
    time_grid = pd.date_range(start=global_start, end=global_end, freq=freq)

    combined_index = df_aid.index.union(time_grid).sort_values()
    df_interp = df_aid.reindex(combined_index)
    df_interp[lat_col] = df_interp[lat_col].interpolate(method='pchip')
    df_interp[lon_col] = df_interp[lon_col].interpolate(method='pchip')

    final_df = df_interp.loc[time_grid].copy()
    final_df['speed_kmh'] = haversine_vectorized_km(
        final_df[lat_col].shift(1), final_df[lon_col].shift(1),
        final_df[lat_col], final_df[lon_col]
    ) / 6.0

    actual_min, actual_max = df_aid.index.min(), df_aid.index.max()
    final_df = final_df[(final_df.index >= actual_min) & (final_df.index <= actual_max)]

    final_df['Year'] = year
    final_df['Animal_ID'] = aid
    all_interpolated_data.append(final_df)

master_df = pd.concat(all_interpolated_data).reset_index().rename(columns={'index': 'timestamp'})
master_df = master_df.dropna(subset=[lat_col, lon_col, 'speed_kmh'])
master_df = master_df[master_df['speed_kmh'] < 15]

# ================= 4. 空间相交与状态判定 =================
print(f"[{time.strftime('%H:%M:%S')}] 3. 正在加载 Shapefiles 并判定海象的空间位置...")
gdf_walrus = gpd.GeoDataFrame(master_df, geometry=gpd.points_from_xy(master_df[lon_col], master_df[lat_col]),
                              crs="EPSG:4326")
gdf_walrus = gdf_walrus.to_crs("EPSG:3413")

# A. 沿岸陆地区域（usedland）
gdf_area = gpd.read_file(shp_path)
if gdf_area.crs is None:
    gdf_area.set_crs("EPSG:3413", inplace=True)
elif gdf_area.crs.to_string() != "EPSG:3413":
    gdf_area = gdf_area.to_crs("EPSG:3413")

area_polygon = gdf_area.unary_union
gdf_walrus["In_Area"] = gdf_walrus.geometry.intersects(area_polygon)

# B. 美国侧研究/捕食区（useiceshp2）
gdf_region = gpd.read_file(region_path)
if gdf_region.crs is None:
    gdf_region.set_crs("EPSG:3413", inplace=True)
elif gdf_region.crs.to_string() != "EPSG:3413":
    gdf_region = gdf_region.to_crs("EPSG:3413")

region_polygon = gdf_region.unary_union
gdf_walrus["In_Region"] = gdf_walrus.geometry.intersects(region_polygon)

# 筛选出整个夏天至少有 1 次踏入美国侧捕食区的海象
valid_animals = (
    gdf_walrus
    .groupby(["Year", "Animal_ID"])["In_Region"]
    .any()
    .reset_index()
)
valid_animals = valid_animals[valid_animals["In_Region"]][["Year", "Animal_ID"]]
print(f"排除了仅去俄罗斯的海象，到达过美国侧捕食区的有效海象数量为：{len(valid_animals)} 只")

gdf_walrus_valid = pd.merge(gdf_walrus, valid_animals, on=["Year", "Animal_ID"], how="inner")

# ================= 5. 计算空间边界穿越次数 (抗抖动平滑) =================
print(f"[{time.strftime('%H:%M:%S')}] 4. 正在计算沿岸边界 (usedland) 穿越次数...")

gdf_walrus_valid = gdf_walrus_valid.sort_values(['Animal_ID', 'timestamp'])
transition_records = []

for (year, aid), grp in gdf_walrus_valid.groupby(['Year', 'Animal_ID']):
    grp = grp.reset_index(drop=True)
    raw_state = grp['In_Area']

    smoothed_state = (
        raw_state
        .rolling(window=3, center=True, min_periods=1)
        .median()
        .astype(bool)
        .values
    )

    if len(smoothed_state) > 1:
        transitions = np.sum(smoothed_state[:-1] != smoothed_state[1:])
    else:
        transitions = 0

    transition_records.append({
        'Year': year,
        'Animal_ID': aid,
        'cross_count': transitions
    })

animal_transitions = pd.DataFrame(transition_records)

animal_annual = pd.merge(
    valid_animals,
    animal_transitions,
    on=["Year", "Animal_ID"],
    how="left"
)
animal_annual["cross_count"] = animal_annual["cross_count"].fillna(0)

print(f"总计识别到 {animal_annual['cross_count'].sum()} 次有效边界穿越")
# ================= 6. 综合统计与可视化 =================
print(f"[{time.strftime('%H:%M:%S')}] 5. 正在绘制空间利用频率演变图 (横向版本)...")

fig, ax = plt.subplots(figsize=(9.6, 7))

all_years = sorted(animal_annual['Year'].unique())

# ============================================================
# 保持原来的统计方式：
# seaborn barplot 原本计算的就是每年 cross_count 的 mean
# 这里仅改变“显示形式”，不改变数据和统计结果

# 每年的均值
year_mean = (
    animal_annual
    .groupby('Year')['cross_count']
    .mean()
    .reindex(all_years)
)

# ------------------------------------------------------------
# 计算每年的原始数据，用于保持与原 barplot 相同的误差统计
# ------------------------------------------------------------
year_values = [
    animal_annual.loc[
        animal_annual['Year'] == year,
        'cross_count'
    ].dropna().values
    for year in all_years
]

# ------------------------------------------------------------
# 用 seaborn 原来的统计逻辑获取均值和 95% CI
# ------------------------------------------------------------
import scipy.stats as stats

ci_low = []
ci_high = []

for values in year_values:

    n = len(values)

    if n <= 1:
        ci_low.append(np.nan)
        ci_high.append(np.nan)
        continue

    mean = np.mean(values)
    sem = stats.sem(values)

    # seaborn barplot 默认 errorbar='ci' 时的 bootstrap CI
    # 为了不改变你原来的统计结果，这里直接使用 bootstrap
    bootstrap_means = []

    rng = np.random.default_rng(0)

    for _ in range(1000):
        sample = rng.choice(
            values,
            size=n,
            replace=True
        )
        bootstrap_means.append(np.mean(sample))

    ci_low.append(
        np.percentile(bootstrap_means, 2.5)
    )

    ci_high.append(
        np.percentile(bootstrap_means, 97.5)
    )

ci_low = np.array(ci_low)
ci_high = np.array(ci_high)

# ============================================================
# 绘制：均值点 + 误差线
# ============================================================

point_color = "#657A86"

# ============================================================
# 年份从上到下递增
# ============================================================
y_pos = np.arange(len(all_years))[::-1]

for i, year in enumerate(all_years):

    mean_val = year_mean.loc[year]

    if np.isfinite(ci_low[i]) and np.isfinite(ci_high[i]):

        # 误差线
        ax.errorbar(
            mean_val,
            y_pos[i],
            xerr=[
                [mean_val - ci_low[i]],
                [ci_high[i] - mean_val]
            ],
            fmt='none',
            ecolor=point_color,
            elinewidth=1.8,
            capsize=4,
            capthick=1.8,
            zorder=3
        )

    # 均值点
    ax.scatter(
        mean_val,
        y_pos[i],
        s=95,
        color=point_color,
        edgecolor='white',
        linewidth=1.5,
        zorder=5
    )

# ============================================================
# 坐标轴
# ============================================================

ax.set_yticks(y_pos)

ax.set_yticklabels(
    all_years,
    fontsize=18
)

ax.set_xlabel(
    "Number of Crossings",
    fontsize=17,
    fontweight='bold',
    labelpad=10
)

ax.set_ylabel(
    "Year",
    fontsize=17,
    fontweight='bold',
    labelpad=10
)

# ============================================================
# 保持原来的 X 轴范围逻辑
# ============================================================

x_max = ax.get_xlim()[1]

# ============================================================
# 在右侧整齐排列 n 值
# ============================================================

n_dict = (
    animal_annual
    .groupby('Year')['Animal_ID']
    .nunique()
    .to_dict()
)

x_max = max(
    x_max,
    np.nanmax(ci_high) * 1.15
)

ax.set_xlim(
    ax.get_xlim()[0],
    x_max
)
# ============================================================
# 背景区域
# ============================================================


# ============================================================
# 2次 crossing 参考线
# ============================================================

ax.axvline(
    2,
    color="#DCEAF4",
    linewidth=1.3,
    linestyle="--",
    zorder=1
)
for i, year in enumerate(all_years):

    if year in n_dict:

        n = n_dict[year]

        ax.text(
            x_max * 0.99,
            y_pos[i],
            f"$n$ = {n}",
            ha='right',
            va='center',
            fontsize=14,
            fontweight='bold',
            color='black',
            zorder=10
        )

# ============================================================
# 只保留左、右边框
# 去掉上边框和下边框
# ============================================================
# ============================================================
# 边框：保留左、下；去掉上、右
# ============================================================

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)

ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')

ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# ============================================================
# 刻度朝内
# ============================================================

ax.tick_params(
    axis='both',
    which='major',
    direction='in',
    length=6,
    width=1.8
)

ax.tick_params(
    axis='x',
    labelsize=18,
    colors='black'
)

ax.tick_params(
    axis='y',
    labelsize=18,
    colors='black'
)

# ============================================================
# 不要网格
# ============================================================

ax.grid(False)

plt.tight_layout()

save_path = (
    r'G:\E\Walrus\movebank\pacific walurs'
    r'\Walrus_Climate_Shift_Horizontal_Publication.png'
)

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches='tight'
)

print(
    f"[{time.strftime('%H:%M:%S')}] "
    f"✅ 最终学术级横向图表已保存至: {save_path}"
)

plt.show()