import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
# =============================
# 1. 读取每日 CSV 表格并构建 7-11 月矩阵（不插值）
# =============================
csv_save_path = r'G:\E\2024seaice\daily_mean_sic_1979_2024.csv'
print(f"正在读取数据表格: {csv_save_path}")

df = pd.read_csv(csv_save_path)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

full_date_index = pd.date_range(start='1979-01-01', end='2024-12-31', freq='D')
df = df.set_index('Date').reindex(full_date_index).rename_axis('Date').reset_index()

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['MonthDay'] = df['Date'].dt.strftime('%m-%d')

df_sub = df[(df['Year'] >= 1979) & (df['Year'] <= 2024) & (df['Month'].isin([7, 8, 9, 10, 11]))].copy()

mat_west = df_sub.pivot(index='MonthDay', columns='Year', values='MeanSIC_West')
mat_east = df_sub.pivot(index='MonthDay', columns='Year', values='MeanSIC_East')

years = mat_west.columns.values
n_days, n_years = mat_west.shape

# =============================
# 2. 自定义配色（>1% 区域色系，<=1% 纯白，缺失数据 浅灰）
# =============================
# missing_color = '#D9D9D9'
missing_color = '#737373'
cmap_west = LinearSegmentedColormap.from_list(
    'WestOrange', ['#FDE5D0', '#E67E22', '#8F3A00']
)
cmap_west.set_under('white')
cmap_west.set_bad(missing_color)

cmap_east = LinearSegmentedColormap.from_list(
    'EastBlue', ['#D6E8F5', '#2b6a99', '#10314B']
)
cmap_east.set_under('white')
cmap_east.set_bad(missing_color)

# =============================
# 3. 高清设置与紧凑型双子图热力图绘制（严格保留原比例）
# =============================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.dpi'] = 300       # 提升屏幕预览清晰度
plt.rcParams['savefig.dpi'] = 600      # 600 DPI 超清保存

# 固定原始画布比例 (5.2 : 2.7)，通过边距控制确保所有元素在画布内，不被裁切变形
fig, axes = plt.subplots(2, 1, figsize=(5.2, 2.7), sharex=True, sharey=True)
plt.subplots_adjust(hspace=0.10, left=0.09, right=0.88, top=0.87, bottom=0.14)

plot_configs = [
    {
        'ax': axes[0],
        'data': mat_west.values,
        'title': 'West',
        'cmap': cmap_west,
        'text_color': '#B85D11'
    },
    {
        'ax': axes[1],
        'data': mat_east.values,
        'title': 'East',
        'cmap': cmap_east,
        'text_color': '#1B4F72'
    }
]

# 计算 7-11 月分界线与居中位置
month_lengths = [31, 31, 30, 31, 30]
month_boundaries = np.cumsum(month_lengths)[:-1] - 0.5
month_centers = np.cumsum([0] + month_lengths[:-1]) + np.array(month_lengths) / 2.0 - 0.5
month_labels = ['Jul', 'Aug', 'Sep', 'Oct', 'Nov']

# 2001与2002年的分界索引
split_idx = np.where(years == 2001)[0][0] + 0.5

for cfg in plot_configs:
    ax = cfg['ax']
    im = ax.imshow(
        cfg['data'],
        aspect='auto',
        origin='lower',
        cmap=cfg['cmap'],
        vmin=1.0,        # <=1% 映射为纯白
        vmax=100.0,
        interpolation='nearest',
        rasterized=True  # 保证导出 PDF 矢量图时色块边缘锐利不发虚
    )

    # 极简月份水平分隔线
    for b in month_boundaries:
        ax.axhline(y=b, color='#888888', linestyle=':', linewidth=0.5, alpha=0.5)

    # 2001/2002 时期分界竖线
    ax.axvline(x=split_idx, color='#333333', linestyle='--', linewidth=0.9, alpha=0.8)

    # 左上角轻量化区域标签（原字体颜色 + 白色描边外框，无背景块）
    ax.text(
        0.02, 0.88, cfg['title'], transform=ax.transAxes,
        fontsize=9.5, fontweight='bold', color=cfg['text_color'],
        va='top', ha='left',
        path_effects=[pe.withStroke(linewidth=2.0, foreground='white')]
    )
    # Y 轴月份刻度
    ax.set_yticks(month_centers)
    ax.set_yticklabels(month_labels, fontsize=8.5)
    ax.tick_params(axis='both', direction='in', length=2.5, width=0.8, pad=2)

    # 右侧紧凑型 Colorbar（恢复纵向 SIC (%) 标签紧贴在 Colorbar 旁）
    cbar = fig.colorbar(im, ax=ax, pad=0.02, aspect=10)
    cbar.set_label('SIC (%)', fontsize=8.5, labelpad=3)
    cbar.ax.tick_params(labelsize=8, direction='in', length=2, width=0.6, pad=1.5)
    cbar.set_ticks([1, 50, 100])
    cbar.set_ticklabels(['0', '50', '100'])
    cbar.outline.set_linewidth(0.6)

# 顶部居中放置单行极简图例（相对于热力图区域水平居中）
legend_patches = [
    mpatches.Patch(facecolor='white', edgecolor='#888888', linewidth=0.7, label='SIC < 1%'),
    mpatches.Patch(facecolor=missing_color, edgecolor='#888888', linewidth=0.7, label='No data')
]
fig.legend(
    handles=legend_patches,
    loc='upper center',
    bbox_to_anchor=(0.46, 0.99),  # 0.46 为左侧热力图主体的水平正中心
    frameon=False,
    fontsize=8.5,
    ncol=2,
    handlelength=1.1,
    handleheight=0.8,
    columnspacing=1.2
)

# =============================
# 4. X 轴年份刻度设置（首尾 1979 与 2024 清晰不重叠）
# =============================
major_years = [1979, 1990, 2000, 2010, 2024]
major_indices = [np.where(years == y)[0][0] for y in major_years]
axes[-1].set_xticks(major_indices)
axes[-1].set_xticklabels([str(y) for y in major_years], fontsize=8.5)

minor_years = [1985, 1995, 2005, 2015, 2020]
minor_indices = [np.where(years == y)[0][0] for y in minor_years]
axes[-1].set_xticks(minor_indices, minor=True)
axes[-1].tick_params(axis='x', which='minor', direction='in', length=1.5, width=0.6)
# =============================
# 5. 按原始画布比例导出 600 DPI 高清 PNG 与 PDF 矢量图
# =============================
png_path = r'G:\E\2024seaice\inset_sic_heatmap_600dpi.png'
pdf_path = r'G:\E\2024seaice\inset_sic_heatmap_vector.pdf'

# 不加 bbox_inches='tight'，严格保持 figsize=(5.2, 2.7) 的原始宽高比（3120 × 1620 像素）
plt.savefig(png_path, dpi=600, transparent=True)
plt.savefig(pdf_path, dpi=600, transparent=True)

print(f"已按原比例保存 600 DPI 高清图至: {png_path}")
print(f"已按原比例保存 PDF 矢量图至: {pdf_path}")

plt.show()