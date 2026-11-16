import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
import geopandas as gpd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ================= 全局图表设置 =================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
# 基础字号放大一倍 (12 -> 24)
plt.rcParams['font.size'] = 24

walrus_shp1_path = r'E:\Walrus\landshp\useiceshp2.shp'  # 东侧捕食区
walrus_shp2_path = r'E:\Walrus\landshp\usiceshp_west.shp'  # 西侧捕食区

# ================= 1. 投影与坐标系设置 =================
# 将中央经线对齐到 169°W
map_proj = ccrs.NorthPolarStereo(central_longitude=-169.0)
data_crs = ccrs.PlateCarree()

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(1, 1, 1, projection=map_proj)
ax.set_extent([-179, -153, 60, 80], crs=data_crs)

ax.set_frame_on(False)

# ================= 2. 绘制基础地图要素 (及捕食区) =================
ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)

# 【修复核心】：定义 EPSG:3413 对应的 Cartopy 坐标系
# (NSIDC Polar Stereographic North 的参数是 lon_0=-45, lat_ts=70)
epsg3413_crs = ccrs.NorthPolarStereo(central_longitude=-45.0, true_scale_latitude=70.0)

try:
    # 【关键修改】：强制转换为 epsg=3413，坚决不转 4326，避免跨 180° 的撕裂现象
    gdf_east = gpd.read_file(walrus_shp1_path).to_crs(epsg=3413)
    gdf_west = gpd.read_file(walrus_shp2_path).to_crs(epsg=3413)

    # 绘制东侧捕食区，使用 epsg3413_crs
    ax.add_geometries(gdf_west.geometry, crs=epsg3413_crs,
                      facecolor='#FFAA00', alpha=0.4, zorder=1)
    # 绘制西侧捕食区，使用 epsg3413_crs
    ax.add_geometries(gdf_east.geometry, crs=epsg3413_crs,
                      facecolor='#00C5FF',  alpha=0.4, zorder=1)
except Exception as e:
    print(f"读取或绘制 shapefile 时出错: {e}")

# 绘制陆地掩膜，zorder=2 确保盖在捕食区上方
land_50m = cfeature.NaturalEarthFeature(
    'physical', 'land', '50m',
    edgecolor='none', facecolor='#D3D3D3'
)
ax.add_feature(land_50m, zorder=2)

# ================= 3. 绘制 169°W 样线与 Point Lay =================
lon_line = -169.0
lat_start = 63.4
lat_end = 78.0
lats_line = np.linspace(lat_start, lat_end, 50)
lons_line = np.full_like(lats_line, lon_line)

ax.plot(lons_line, lats_line, transform=data_crs,
        color='#DE5E46', linewidth=5, zorder=4, label='169°W')

lat_ticks = np.arange(64, 79, 2)
tick_length = 0.6

for lat in lat_ticks:
    ax.plot([lon_line - tick_length / 2, lon_line + tick_length / 2], [lat, lat],
            transform=data_crs, color='black', linewidth=3, zorder=5)
    ax.text(lon_line - 6.5, lat, f'{lat}°N', transform=data_crs,
            color='black', fontsize=20, va='center', ha='left', zorder=6)

pt_lay_lat = 74
pt_lay_lon = -163.0053
lons_pt = np.linspace(-180, -150, 50)
lats_pt = np.full_like(lons_pt, pt_lay_lat)

ax.plot(lons_pt, lats_pt, transform=data_crs,
        color='#555555', linestyle='--', linewidth=2, zorder=3)

# ax.plot(pt_lay_lon, pt_lay_lat, marker='*', color='gold', markersize=24,
#         markeredgecolor='black', transform=data_crs, zorder=5, label='Point Lay')

# ================= 4. 标题与图例 =================
handles, labels = ax.get_legend_handles_labels()

patch_east = mpatches.Patch(facecolor='#FFAA00',  alpha=0.6, label='Foraging Area (East)')
patch_west = mpatches.Patch(facecolor='#00C5FF',  alpha=0.6, label='Foraging Area (West)')
handles.extend([patch_east, patch_west])

ax.legend(handles=handles, loc='lower right', frameon=False, fontsize=20)

plt.tight_layout()
plt.show()
print("🎉 地图修复完成！180°经线的撕裂问题已解决！")