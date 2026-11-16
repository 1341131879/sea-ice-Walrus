import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import warnings
import time
import rasterio
from pyproj import Transformer

warnings.filterwarnings('ignore')


# --- 1. Haversine 距离函数 ---
def haversine_vectorized_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


# --- 2. 基础设置与路径 ---
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

input_file = r'G:\E\Walrus\movebank\pacific walurs\pacificWalrus_USGS_ASC_argosGPS.csv'
ice_dir = r'G:\E\2024seaice\osidailytif'
output_dir = r'G:\E\Walrus\monthly_walrus_ice_speed'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 🌟 只保留指定的 3 只海象
target_ids = [603, 616]
global_start = '2012-07-15'
global_end = '2012-10-20'
freq = '6H'
output_filename = 'Walrus_SIC_Speed_Combined_2012.png'

# --- 3. 加载、清洗轨迹数据并计算速度 ---
print(f"[{time.strftime('%H:%M:%S')}] 1. 正在加载并清洗轨迹数据...")
df = pd.read_csv(input_file, low_memory=False)
id_col = 'Animal_ID' if 'Animal_ID' in df.columns else 'individual-local-identifier'
lat_col = 'location-lat' if 'location-lat' in df.columns else 'Latitude'
lon_col = 'location-long' if 'location-long' in df.columns else 'Longitude'
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 解决 180度经线跨越问题
df[lon_col] = np.where(df[lon_col] < 0, df[lon_col] + 360, df[lon_col])
time_grid = pd.date_range(start=global_start, end=global_end, freq=freq)
anim_data = {}

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
land_polar = world[world.name.isin(['Russia', 'United States of America', 'Canada'])].to_crs(
    "EPSG:3413").unary_union.buffer(-5000)

for aid in target_ids:
    df_aid = df[(df[id_col] == aid) & (df['timestamp'] >= global_start) & (df['timestamp'] <= global_end)].copy()
    if df_aid.empty: continue

    # 空间清洗
    gdf = gpd.GeoDataFrame(df_aid, geometry=gpd.points_from_xy(df_aid[lon_col], df_aid[lat_col]), crs="EPSG:4326")
    df_aid = df_aid[~gdf.to_crs("EPSG:3413").within(land_polar)].copy()

    # 异常速度过滤
    max_speed_kmh = 18.0
    while True:
        df_aid['prev_time'] = df_aid['timestamp'].shift(1)
        df_aid['prev_lat'] = df_aid[lat_col].shift(1)
        df_aid['prev_lon'] = df_aid[lon_col].shift(1)
        df_aid['time_diff_h'] = (df_aid['timestamp'] - df_aid['prev_time']).dt.total_seconds() / 3600.0
        df_aid['dist_km'] = haversine_vectorized_km(df_aid['prev_lat'], df_aid['prev_lon'], df_aid[lat_col],
                                                    df_aid[lon_col])
        df_aid['speed_kmh'] = df_aid['dist_km'] / df_aid['time_diff_h']

        overspeed_mask = df_aid['speed_kmh'] > max_speed_kmh
        if not overspeed_mask.any(): break
        df_aid = df_aid.drop(df_aid['speed_kmh'].idxmax())

    # 平滑插值
    df_aid = df_aid.drop_duplicates('timestamp').set_index('timestamp').sort_index()
    combined_index = df_aid.index.union(time_grid).sort_values()
    df_interp = df_aid.reindex(combined_index)
    df_interp[lat_col] = df_interp[lat_col].interpolate(method='pchip')
    df_interp[lon_col] = df_interp[lon_col].interpolate(method='pchip')

    final_df = df_interp.loc[time_grid].copy()
    final_df['speed_kmh'] = haversine_vectorized_km(final_df[lat_col].shift(1), final_df[lon_col].shift(1),
                                                    final_df[lat_col], final_df[lon_col]) / 6.0

    actual_min, actual_max = df_aid.index.min(), df_aid.index.max()
    final_df.loc[
        (final_df.index < actual_min) | (final_df.index > actual_max), [lat_col, lon_col, 'speed_kmh']] = np.nan
    final_df['sic'] = np.nan
    anim_data[aid] = final_df

if not anim_data:
    print("\n❌ 错误：所有数据为空！")
    exit()

# --- 4. 提取海冰密集度 (SIC) ---
print(f"[{time.strftime('%H:%M:%S')}] 2. 正在提取海冰数据 (这部分最耗时，请稍候)...")
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)
unique_dates = pd.to_datetime(time_grid.date).unique()

for date_obj in unique_dates:
    date_str = date_obj.strftime('%Y%m%d')
    ice_file = os.path.join(ice_dir, f'osi_daily_{date_str}.tif')

    if os.path.exists(ice_file):
        with rasterio.open(ice_file) as src:
            for aid in target_ids:
                if aid not in anim_data: continue
                df_aid = anim_data[aid]

                mask = (df_aid.index.date == date_obj.date()) & (df_aid[lat_col].notna())
                if not mask.any(): continue

                lats = df_aid.loc[mask, lat_col].values
                lons = df_aid.loc[mask, lon_col].values
                lons_proj = np.where(lons > 180, lons - 360, lons)
                xs, ys = transformer.transform(lons_proj, lats)

                try:
                    sampled_values = np.array([val[0] for val in src.sample(list(zip(xs, ys)))])
                    if src.nodata is not None:
                        sampled_values = np.where(sampled_values == src.nodata, np.nan, sampled_values)
                    sampled_values = np.where((sampled_values < 0) | (sampled_values > 100), np.nan, sampled_values)
                    anim_data[aid].loc[mask, 'sic'] = sampled_values
                except:
                    pass

# --- 5. 绘制柱状图与折线的组合图 (双Y轴排版) ---
print(f"[{time.strftime('%H:%M:%S')}] 3. 正在生成高级排版图表...")

# 创建 3 行 1 列的画布，高度设置为 10 保证不拥挤
fig, axes = plt.subplots(nrows=len(target_ids), ncols=1, figsize=(5,5), dpi=600,sharex=True)
fig.subplots_adjust(hspace=0.25)

# 🌟 沿用你代码中的前三个荧光亮色配色
color_map = {
    603: '#6c5070',  # 亮绿
    616: '#df6a6a',  # 亮粉
    645: '#c2dbc1'  # 亮橘
}

for i, aid in enumerate(target_ids):
    if aid not in anim_data: continue
    plot_data = anim_data[aid].dropna(subset=[lat_col])
    color = color_map.get(aid, '#000000')

    ax1 = axes[i]  # 左Y轴：海冰柱状图
    ax2 = ax1.twinx()  # 右Y轴：速度折线图

    # === 【新增】绘制底层时间分段背景 ===
    cutoff_date = pd.to_datetime('2012-09-19')
    start_date = pd.to_datetime(global_start)
    end_date = pd.to_datetime(global_end)

    # 9月19日前：浅冰蓝背景 (代表海冰期)
    ax1.axvspan(start_date, cutoff_date, facecolor='white', alpha=0.7, zorder=0)

    # 9月19日后：浅青蓝/水灰背景 (代表开阔水域)
    # 你可以把 '#E0F7FA' 换成 '#EAEDED' 看看哪种对比效果更好
    ax1.axvspan(cutoff_date, end_date, facecolor='#CBEBF6', alpha=0.7, zorder=0)  ##F6EAC1沙滩色

    # === 绘制柱状图 (海冰密集度 - 浅灰色) ===
    # 注意：此处加上 zorder=1 确保柱状图在背景上方
    bar_plot = ax1.bar(plot_data.index, plot_data['sic'],
                       color='#7F8C8D', alpha=0.7, width=0.25, zorder=1)

    # ... 后续原本的折线图代码保持不变 ...

    # === 绘制折线图 (移动速度) ===
    line_plot = ax2.plot(plot_data.index, plot_data['speed_kmh'],
                         color=color, linewidth=2.5, label=f'ID {aid}')

    # === 格式化坐标轴 ===
    ax1.set_ylim(0, 105)
    ax2.set_ylim(0, 8)

    ax1.set_ylabel('SIC (%)', fontsize=14, fontweight='bold', color='#555555')
    ax2.set_ylabel('Speed (km/h)', fontsize=14, fontweight='bold', color=color)

    ax1.grid(True, linestyle='--', alpha=0.4)

    # === 合并图例 ===
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize=12, frameon=False)
# 右侧 Speed
    ax2.tick_params(axis='y',which='both',direction='in',length=4,width=1)
    ax1.tick_params(axis='y', which='both', direction='in', length=4, width=1)
# === 底部X轴时间格式化 (14天间隔 + 30度倾斜) ===
axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=14))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
axes[-1].set_xlabel('Date (2013)', fontsize=16, fontweight='bold', labelpad=10)
axes[-1].tick_params(axis='x', labelsize=11, direction='in',rotation=30)
axes[-1].tick_params(axis='y', labelsize=14, direction='in',rotation=0)
axes[0].tick_params(axis='y', labelsize=14, direction='in',rotation=0)


plt.show()

# # 想要静默保存图片的话，取消下面三行的注释即可
# save_path = os.path.join(output_dir, output_filename)
# plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
# print(f"[{time.strftime('%H:%M:%S')}] ✅ 完美结束！静态双轴关系图已成功保存至：\n{save_path}")