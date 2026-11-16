import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import rasterio
from matplotlib.lines import Line2D

# --- 1. 设置与路径 ---
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
# 在代码最开头（1. 设置与路径）加上这句，全局调细网格线
plt.rcParams['hatch.linewidth'] = 0.8

shp_dir = r'G:\E\2024seaice\osimonthly_avg_edge_shp'  # 边缘线
output_dir = r'G:\E\Walrus\climate_comparison_plots'
topo_file = r'G:\E\arcgisprodata\ETOPO\topo_Layer.tif'  # 地形

walrus_shp1_path = r'G:\E\Walrus\landshp\useiceshp2.shp'  # 东侧捕食区
walrus_shp2_path = r'G:\E\Walrus\landshp\usiceshp_west.shp'  # 西侧捕食区

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

target_months = [7, 8, 9]

# --- 2. 投影设置 (严格对齐 EPSG:3413) ---
epsg3413 = ccrs.NorthPolarStereo(central_longitude=-45.0, true_scale_latitude=70.0)
plot_crs = ccrs.NorthPolarStereo(central_longitude=-169.0)
data_crs = ccrs.PlateCarree()

# --- 3. 预加载水深地形底图 ---
print("正在加载 ETOPO 水深底图...")
with rasterio.open(topo_file) as src:
    topo_data = src.read(1).astype(float)
    topo_bounds = src.bounds
    topo_extent = [topo_bounds.left, topo_bounds.right, topo_bounds.bottom, topo_bounds.top]

    # 剔除陆地
    topo_data[topo_data > 0] = np.nan
    # 剔除200m以下的深水（使其变透明，从而露出底部的纯黑背景）
    topo_data[topo_data < -200] = np.nan

# --- 4. 开始绘图 ---
print("正在绘制合成图...")

fig = plt.figure(figsize=(11, 10))
ax = plt.axes(projection=plot_crs)

ax.set_extent([179, 201, 63, 77], crs=data_crs)

# A. 底层：深水限制区 (zorder=0) -> 【修改：去掉 hatch='xx'，变成纯黑色】
background_patch = mpatches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                      facecolor='#3B566C', edgecolor='none', zorder=0)

ax.add_patch(background_patch)

# B. 第1层：水深底图 (zorder=1) -> 保留灰色分层设色
levels = [-200, -150, -100, -50, 0]
# grey_colors = ['#808080', '#A0A0A0', '#C0C0C0', '#E0E0E0']
# grey_colors = [
#     "#969696",   # 150–200 m
#     "#BFBFBF",   # 100–150 m
#     "#D9D9D9",   # 50–100 m
#     "#F2F2F2"    # 0–50 m
# ]
grey_colors = [
    "#718B99",  # 150–200 m
    "#9CB0B9",  # 100–150 m
    "#C4D1D6",  # 50–100 m
    "#E5EBED"   # 0–50 m
]
cmap_shallow = mcolors.ListedColormap(grey_colors)
norm_shallow = mcolors.BoundaryNorm(levels, cmap_shallow.N)

ax.imshow(topo_data, origin='upper', extent=topo_extent, transform=data_crs,
          cmap=cmap_shallow, norm=norm_shallow, alpha=1.0, zorder=1)

# C. 第2层：海象捕食区域 (zorder=2) -> 【注意：东西侧一定要分开读取和判断】

# C. 第2层：海象捕食区域 (zorder=2)

# 1. 东侧捕食区 (shp1) -> 对应青蓝色 (Cyan)
if os.path.exists(walrus_shp1_path):
    gdf_w1 = gpd.read_file(walrus_shp1_path)
    # 步骤A：先画半透明底色 (青蓝色)
    ax.add_geometries(gdf_w1.geometry, crs=epsg3413,
                      facecolor='#00C5FF', edgecolor='none',
                      alpha=0.05, zorder=2)
    # 步骤B：再画单向斜线网格 (右斜线 //)
    ax.add_geometries(gdf_w1.geometry, crs=epsg3413,
                      facecolor='none', edgecolor='#0099CC',
                      linewidth=1.5, hatch='//', zorder=2.1)

# 2. 西侧捕食区 (shp2) -> 对应橙黄色 (Orange)
if os.path.exists(walrus_shp2_path):
    gdf_w2 = gpd.read_file(walrus_shp2_path)
    # 步骤A：先画半透明底色 (橙黄色)
    ax.add_geometries(gdf_w2.geometry, crs=epsg3413,
                      facecolor='#FFAA00', edgecolor='none',
                      alpha=0.05, zorder=2)
    # 步骤B：再画单向斜线网格 (左斜线 \\)
    ax.add_geometries(gdf_w2.geometry, crs=epsg3413,
                      facecolor='none', edgecolor='#CC8800',
                      linewidth=1.5, hatch='\\\\', zorder=2.1)

# D. 第3层：海冰边缘线 (zorder=3)
# month_colors = ['#0033FF', '#FFCC00', '#CC0000']
month_colors = [
    "#3366AA",
    "#7A8F45",
    "#B03030"
]
for i, month in enumerate(target_months):
    mm_str = f"{month:02d}"
    shp_pre = os.path.join(shp_dir, f'ice_edge_pre2002_{mm_str}.shp')
    shp_post = os.path.join(shp_dir, f'ice_edge_post2002_{mm_str}.shp')

    current_color = month_colors[i]

    if os.path.exists(shp_pre):
        gdf_pre = gpd.read_file(shp_pre)
        ax.add_geometries(gdf_pre.geometry, crs=epsg3413,
                          facecolor='none', edgecolor=current_color,
                          linewidth=3.5, alpha = 0.8,linestyle='--', zorder=3)

    if os.path.exists(shp_post):
        gdf_post = gpd.read_file(shp_post)
        ax.add_geometries(gdf_post.geometry, crs=epsg3413,
                          facecolor='none', edgecolor=current_color,
                          linewidth=3.5, linestyle='-', zorder=3)

# 陆地修改为略带暖意的浅卡其灰，与海水的冷灰形成对比
ax.add_feature(cfeature.LAND, facecolor='#B9B09F', edgecolor='none', zorder=4)


# F. 第5层：海岸线与网格线 (zorder=5)
ax.add_feature(cfeature.COASTLINE, linewidth=1, color='#263640', zorder=5)
gl = ax.gridlines(crs=data_crs, draw_labels=False, linewidth=1, color='white', alpha=0.6, linestyle='--', zorder=5)
gl.ylocator = mticker.MultipleLocator(5)
gl.xlocator = mticker.MultipleLocator(20)
# G. 自定义图例排版（去除了水深文字，并在最底部留出整行空白）
# G. 自定义图例排版（单列严密逻辑，底部留白）
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

legend_elements = [

    # ================= Sea-ice edge =================
    Line2D([], [], linestyle='none', label=''),

    Line2D([0], [0], color=month_colors[0], lw=4, label='Jul.'),
    Line2D([0], [0], color=month_colors[1], lw=4, label='Aug.'),
    Line2D([0], [0], color=month_colors[2], lw=4, label='Sept.'),


    # ================= Period =================
    Line2D([], [], linestyle='none', label='\n'),

    Line2D([0], [0], color='black', lw=2.0, linestyle='--', label='1979–2001'),
    Line2D([0], [0], color='black', lw=3.2, linestyle='-', label='2002–2024'),



    # ================= Foraging area =================
    Line2D([], [], linestyle='none', label='\n'),

    mpatches.Patch(
        facecolor='#00C5FF',
        edgecolor='#0099CC',
        hatch='//',
        alpha=0.6,
        label='East'
    ),

    mpatches.Patch(
        facecolor='#FFAA00',
        edgecolor='#CC8800',
        hatch='\\\\',
        alpha=0.6,
        label='West'
    ),

    # 预留给 Bathymetry Colorbar 的空白
    Line2D([], [], linestyle='none', label='\n\n')
]
leg = ax.legend(
    handles=legend_elements,
    loc='lower left',
    bbox_to_anchor=(1.01, 0.45),
    fontsize=13,
    frameon=True,
    facecolor='white',
    edgecolor='0',
    framealpha=0,
    borderpad=0.8,
    handlelength=2.8,
    handletextpad=0.8,
    labelspacing=0.45
)

# 所有内容左对齐
leg._legend_box.align = "left"

# 加粗标题
titles = ['Sea-ice edge', 'Period', 'Foraging area']

for text in leg.get_texts():
    if text.get_text() in titles:
        text.set_fontweight('bold')
        text.set_fontsize(11.5)

for handle, text in zip(leg.legendHandles, leg.get_texts()):
    if text.get_text() in titles:
        handle.set_visible(False)

plt.savefig(os.path.join(output_dir, 'Merged_Ice_Edge_Walrus_Areas.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.show()