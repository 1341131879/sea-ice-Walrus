import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import os
import rasterio
from pyproj import Transformer
import matplotlib.ticker as ticker
from scipy import stats
from adjustText import adjust_text
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Arial'
sns.set_theme(style="whitegrid")

# ================= 1. 基础设置与文件路径 =================
gps_file = r'G:\E\Walrus\movebank\pacific walurs\pacificWalrus_USGS_ASC_argosGPS.csv'
sensor_file = r'G:\E\Walrus\USGS_GPS\pacificWalrus_USGS_ASC_argosGPS_processedData\pacificWalrus_USGS_ASC_argosGPS_decodedSensorBehavior.csv'
region_path = r'G:\E\Walrus\landshp\useiceshp2.shp'
ice_dir = r'G:\E\2024seaice\osidailytif'
freq = '6H'


# 定义统一的时间范围过滤函数（7月1日 到 9月15日）
def filter_date_range(df, time_col='timestamp'):
    month = df[time_col].dt.month
    day = df[time_col].dt.day
    # 条件：(7月) 或 (8月) 或 (9月且日期 <= 15日)
    mask = (month.isin([7, 8])) | ((month == 9) & (day <= 10))
    return df[mask].copy()


# ================= 2. 加载与处理传感器 (Forage) 数据 =================
print(f"[{time.strftime('%H:%M:%S')}] 1. 正在加载高频传感器数据，提取真实捕食状态...")
df_sensor = pd.read_csv(sensor_file, low_memory=False)

sensor_time_col = 'sensorInterval_Start_Timestamp_UTC'
df_sensor['timestamp'] = pd.to_datetime(df_sensor[sensor_time_col]).dt.tz_localize(None)

# 将传感器数据限制在 7月1日 - 9月15日
df_sensor = filter_date_range(df_sensor, 'timestamp')

# 终极过滤修复：将捕食状态转化为小写字符串，清洗前后的空格
status_str = df_sensor['sensorInterval_Forage_Status'].astype(str).str.strip().str.lower()

# 白名单过滤机制：只保留明确知道是状态的值，把 NaN、"nan"、空格、错误代码 全部剔除！
valid_mask = status_str.isin(['0', '1', 'true', 'false', '0.0', '1.0', 'yes', 'no', 'y', 'n'])
df_sensor = df_sensor[valid_mask].copy()

# 重新获取过滤后的干净文本，再提取 1 为捕食，0 为非捕食
clean_status = df_sensor['sensorInterval_Forage_Status'].astype(str).str.lower()
df_sensor['is_foraging'] = clean_status.str.contains('1|true|yes|y').astype(int)

# 按 6H 重采样，计算每个时间窗口内的平均捕食时间占比
sensor_resampled = (
    df_sensor.set_index('timestamp')
    .groupby('Animal_ID')
    .resample(freq)['is_foraging']
    .mean()
    .reset_index()
    .rename(columns={'is_foraging': 'Forage_Proportion'})
)

# ================= 3. 加载与插值 GPS 轨迹数据 =================
print(f"[{time.strftime('%H:%M:%S')}] 2. 正在加载轨迹数据并过滤异常点...")
df_gps = pd.read_csv(gps_file, low_memory=False)
id_col = 'Animal_ID' if 'Animal_ID' in df_gps.columns else 'individual-local-identifier'
lat_col = 'location-lat' if 'location-lat' in df_gps.columns else 'Latitude'
lon_col = 'location-long' if 'location-long' in df_gps.columns else 'Longitude'

df_gps['timestamp'] = pd.to_datetime(df_gps['timestamp']).dt.tz_localize(None)
df_gps['Year'] = df_gps['timestamp'].dt.year

# 将轨迹时间限制在 7月1日 - 9月15日
df_gps = filter_date_range(df_gps, 'timestamp')
df_gps[lon_col] = np.where(df_gps[lon_col] < 0, df_gps[lon_col] + 360, df_gps[lon_col])


def haversine_vectorized_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


grouped = df_gps.groupby(['Year', id_col])
all_interpolated_data = []

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

# ================= 4. 数据合并与纯海上空间判定 =================
print(f"[{time.strftime('%H:%M:%S')}] 3. 正在融合捕食状态并执行离岸区域筛选...")

gps_years = set(master_df['Year'].unique())

merged_df = pd.merge(master_df, sensor_resampled, on=['Animal_ID', 'timestamp'], how='left')

# 完美销毁完全没有行为数据的年份（如04, 07年）
merged_df = merged_df.dropna(subset=['Forage_Proportion'])

gdf_walrus = gpd.GeoDataFrame(
    merged_df, geometry=gpd.points_from_xy(merged_df[lon_col], merged_df[lat_col]),
    crs="EPSG:4326"
).to_crs("EPSG:3413")

gdf_region = gpd.read_file(region_path).to_crs("EPSG:3413")
gdf_walrus["In_Region"] = gdf_walrus.geometry.intersects(gdf_region.unary_union)

gdf_target = gdf_walrus[gdf_walrus["In_Region"] == True].copy()
gdf_target['Date_Str'] = gdf_target['timestamp'].dt.strftime('%Y%m%d')

if len(gdf_target) == 0: raise ValueError("严重错误：筛选后捕食区内无数据！")

# ================= 5. 海冰数据精准提取 =================
print(f"[{time.strftime('%H:%M:%S')}] 4. 开始提取每日海冰密集度 (SIC)... (这可能需要几分钟)")
gdf_target['SIC'] = np.nan
unique_dates = pd.to_datetime(gdf_target['Date_Str'], format='%Y%m%d').dt.date.unique()
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)

for date_obj in unique_dates:
    date_str = date_obj.strftime('%Y%m%d')
    ice_file = os.path.join(ice_dir, f'osi_daily_{date_str}.tif')
    if os.path.exists(ice_file):
        with rasterio.open(ice_file) as src:
            mask = (gdf_target['Date_Str'] == date_str)
            group = gdf_target[mask]
            if group.empty: continue
            lats, lons = group[lat_col].values, group[lon_col].values
            lons_proj = np.where(lons > 180, lons - 360, lons)
            xs, ys = transformer.transform(lons_proj, lats)
            try:
                vals = np.array([v[0] for v in src.sample(list(zip(xs, ys)))])
                if src.nodata is not None: vals = np.where(vals == src.nodata, np.nan, vals)
                vals = np.where((vals < 0) | (vals > 100), np.nan, vals)
                gdf_target.loc[mask, 'SIC'] = vals
            except:
                pass

gdf_target = gdf_target[gdf_target['SIC'].notna()].copy()

# ================= 6. 核心统计：计算年度平均海冰密集度 =================
print(f"[{time.strftime('%H:%M:%S')}] 5. 正在处理年度统计并计算年均海冰密集度...")

# 明确提示年份断档审计
valid_behavior_years = set(gdf_target['Year'].unique())
missing_behavior_years = gps_years - valid_behavior_years
if missing_behavior_years:
    print(
        f"💡 明确提示：检测到年份 {sorted(list(missing_behavior_years))} 【完全缺失】高频捕食传感器数据，代码已自动将其安全排除。")

# ---- 提取行为变量 Y ----
yearly_behavior = (gdf_target.groupby('Year')['Forage_Proportion'].mean() * 100).reset_index(name='Foraging_Prop')

# ---- 💡 核心修改：提取海冰变量 X（计算研究周期内的平均海冰密集度） ----
years_with_august_data = gdf_target[gdf_target['timestamp'].dt.month == 8]['Year'].unique()

ice_mean_list = []
# 直接对筛选后的空间点位按年求海冰密集度均值
for year, group in gdf_target.groupby('Year'):
    # 同样保持硬性条件：必须包含核心的8月数据
    if year in years_with_august_data and len(group) >= 5:
        mean_sic = group['SIC'].mean()
        ice_mean_list.append({'Year': year, 'Mean_SIC': mean_sic})
    else:
        if year not in years_with_august_data:
            print(f"⚠️ 提示：年份 {year} 因在核心的 8 月份无任何有效观测数据，已被排除。")

df_ice_x = pd.DataFrame(ice_mean_list)

# 通过 how='inner' 对齐
final_stats = pd.merge(yearly_behavior, df_ice_x, on='Year', how='inner').dropna()

# 💡 此时不需要再额外强制剔除2019年，因为2019年算出来的“年均密集度接近0”是非常准确的真实特征点
print(f"📊 【有效年份对齐完成】")
print(
    f">> 最终参与学术回归的有效年份共 {len(final_stats)} 年，具体为: {sorted(final_stats['Year'].astype(int).tolist())}")

if len(final_stats) < 2:
    raise ValueError("严重错误：两项数据皆完整的有效年份少于 2 年，无法建立回归关系！")
# ================= 7. 生成核心学术回归图 (防重叠终极版) =================
print(f"[{time.strftime('%H:%M:%S')}] 6. 正在配置标准线性回归模型并使用自动避让绘制图表...")

fig, ax = plt.subplots(figsize=(8, 6.2))

x_data = final_stats['Mean_SIC'].values
y_data = final_stats['Foraging_Prop'].values

# 绘制散点
ax.scatter(
    x_data,
    y_data,
    s=190,
    color="#2C6BA0",
    edgecolor="white",
    linewidth=1.8,
    alpha=0.95,
    zorder=5
)

# 计算标准线性回归
slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
r_squared = r_value ** 2

# =========================================================
# Linear regression + 95% confidence interval
# =========================================================

x_fit = np.linspace(
    np.min(x_data) * 0.8,
    np.max(x_data) * 1.15,
    200
)

y_fit = slope * x_fit + intercept

# 样本量
n = len(x_data)

# 残差标准误
residuals = y_data - (slope * x_data + intercept)
s_err = np.sqrt(
    np.sum(residuals ** 2) / (n - 2)
)

# x 的均值和离差平方和
x_mean = np.mean(x_data)
Sxx = np.sum((x_data - x_mean) ** 2)
from scipy.stats import t
# 95% CI
t_crit = t.ppf(0.975, df=n - 2)

y_se = s_err * np.sqrt(
    1 / n + (x_fit - x_mean) ** 2 / Sxx
)

ci = t_crit * y_se

y_lower = y_fit - ci
y_upper = y_fit + ci

# 95% CI 阴影
ax.fill_between(
    x_fit,
    y_lower,
    y_upper,
    color='#C44E52',
    alpha=0.18,
    linewidth=0,
    zorder=2
)

# 回归线
ax.plot(
    x_fit,
    y_fit,
    color='#C44E52',
    linewidth=3,
    linestyle='-',
    zorder=6
)

# 💡 核心修改：使用 adjustText 智能排版年份标签
texts = []
for i, row in final_stats.iterrows():
    # 注意：这里去掉了原来坐标里的手动 +0.8，交由引擎自动分配位置
    texts.append(ax.text(row['Mean_SIC'], row['Foraging_Prop'], str(int(row['Year'])), fontsize=18, color='#555555',
        zorder=20  ))

# 启动防重叠引擎，自动绘制细灰线连接挤在一起的点和标签
adjust_text(
    texts,
    x=x_data,
    y=y_data,
    arrowprops=dict(arrowstyle='-', color='gray', lw=0.8, alpha=0.6), # 重叠的标签会带有极简的灰色指引线
    expand=(1.2, 1.4), # 拉开的斥力倍数
    ax=ax
)

# 精简统计文本面板
sig_star = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else "(ns)"))

stats_text = (
    f"$R^2$ = {r_squared:.2f}\n"
    f"$P$-value = {p_value:.4f} {sig_star}"
)

ax.text(
    0.95, 0.95, stats_text,
    transform=ax.transAxes,
    ha='right', va='top', fontsize=18, color='#222222',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor="0.8", alpha=0.9)
)

# 绘图样式配置
ax.tick_params(axis='both', which='major', labelsize=18)
ax.set_xlabel("  Mean SIC(%)  ", fontsize=16, fontweight='bold')
ax.set_ylabel("Proportion of time in water (%)", fontsize=16, fontweight='bold')

# Y 轴刻度间隔为 5%
ax.yaxis.set_major_locator(ticker.MultipleLocator(5))

# =============================
# 坐标轴与四周边框
# =============================
ax.tick_params(
    axis='both',
    which='major',
    direction='in',
    labelsize=18
)

# 四周完整黑色边框
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(1.2)

# 不显示网格
ax.grid(False)

plt.tight_layout()
# save_path = r'E:\Walrus\movebank\pacific walurs\Walrus_Foraging_vs_MeanSIC_Linear_SmartText.png'
# plt.savefig(save_path, dpi=300, bbox_inches='tight')
# print(f"[{time.strftime('%H:%M:%S')}] ✅ 图表已成功保存至: \n{save_path}")
plt.show()