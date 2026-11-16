import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =============================
# 1. 参数设置与数据读取
# =============================
csv_save_path = r'G:\E\2024seaice\daily_mean_sic_1979_2024.csv'
SIC_THRESHOLD = 1.0   # 无冰阈值 (%)
CONSEC_DAYS = 3       # 连续无冰天数判定标准

df = pd.read_csv(csv_save_path)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

full_date_index = pd.date_range(start='1979-01-01', end='2024-12-31', freq='D')
df = df.set_index('Date').reindex(full_date_index).rename_axis('Date').reset_index()

df[['MeanSIC_West', 'MeanSIC_East']] = df[['MeanSIC_West', 'MeanSIC_East']].interpolate(
    method='linear', limit=3
)
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# =============================
# 2. 统计每年无冰总天数与首次无冰日期（剔除有冰年份）
# =============================
def find_first_ice_free_day(df_year_jul_sep, col_name, thresh=5.0, n_consec=5):
    df_year_jul_sep = df_year_jul_sep.sort_values('Date').reset_index(drop=True)
    is_free = df_year_jul_sep[col_name] < thresh

    cond = is_free.copy()
    for k in range(1, n_consec):
        cond = cond & is_free.shift(-k, fill_value=False)

    if cond.any():
        first_idx = int(cond.idxmax())
        first_date_str = df_year_jul_sep.loc[first_idx, 'Date'].strftime('%m-%d')
        return first_idx, first_date_str
    return np.nan, None

years = list(range(1979, 2025))
summary_records = []

for yr in years:
    df_jul_nov = df[(df['Year'] == yr) & (df['Month'].isin([7, 8, 9, 10, 11]))]
    days_west = float((df_jul_nov['MeanSIC_West'] < SIC_THRESHOLD).sum())
    days_east = float((df_jul_nov['MeanSIC_East'] < SIC_THRESHOLD).sum())

    df_jul_sep = df[(df['Year'] == yr) & (df['Month'].isin([7, 8, 9]))]
    onset_idx_west, _ = find_first_ice_free_day(df_jul_sep, 'MeanSIC_West', SIC_THRESHOLD, CONSEC_DAYS)
    onset_idx_east, _ = find_first_ice_free_day(df_jul_sep, 'MeanSIC_East', SIC_THRESHOLD, CONSEC_DAYS)

    # 【核心修改】：将“有冰年份”设为 np.nan，不纳入 Ice-Free Days 统计
    # 判定标准：无冰天数为 0，或未满足连续 CONSEC_DAYS 天无冰（保证左右两图样本年份一致）
    # （注：若仅想排除天数为 0 的年份，可改为 if days_west == 0:）
    if days_west == 0 or pd.isna(onset_idx_west):
        days_west = np.nan
    if days_east == 0 or pd.isna(onset_idx_east):
        days_east = np.nan

    period_label = '1979–2001' if yr <= 2001 else '2002–2024'
    summary_records.append({
        'Year': yr,
        'Period': period_label,
        'IceFreeDays_West': days_west,
        'OnsetIndex_West': onset_idx_west,
        'IceFreeDays_East': days_east,
        'OnsetIndex_East': onset_idx_east,
    })

yearly_df = pd.DataFrame(summary_records)

def index_to_date(mean_idx):
    if pd.isna(mean_idx):
        return None
    day_offset = int(np.floor(mean_idx + 0.5))
    mean_date = pd.Timestamp('2001-07-01') + pd.Timedelta(days=day_offset)
    return mean_date.strftime('%b %d')

# =============================
# 3. 高清绘图设置
# =============================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 0.9
plt.rcParams['figure.dpi'] = 300       # 解决 plt.show() 预览模糊的问题
plt.rcParams['savefig.dpi'] = 600      # 导出超高清分辨率

fig, (ax_days, ax_onset) = plt.subplots(1, 2, figsize=(5.0, 2.9))
plt.subplots_adjust(wspace=0.40, top=0.83, bottom=0.14, left=0.11, right=0.97)

# 配色：浅色箱体填充 + 深色边框与散点
color_west = '#E67E22'
fill_west = '#FAD7A0'
dark_west = '#873600'

color_east = '#2b6a99'
fill_east = '#AED6F1'
dark_east = '#102A43'

periods = ['1979–2001', '2002–2024']
np.random.seed(42)

def draw_grouped_box_scatter(ax, col_west, col_east, ylabel, is_date_axis=False):
    positions_w = [0 - 0.18, 1 - 0.18]
    positions_e = [0 + 0.18, 1 + 0.18]

    data_w = [yearly_df[yearly_df['Period'] == p][col_west].dropna().values for p in periods]
    data_e = [yearly_df[yearly_df['Period'] == p][col_east].dropna().values for p in periods]

    # 1. 绘制箱线图（过滤掉可能为空的时段，防止全为有冰年份时报错）
    valid_w = [(pos, d) for pos, d in zip(positions_w, data_w) if len(d) > 0]
    if valid_w:
        ax.boxplot(
            [d for _, d in valid_w],
            positions=[pos for pos, _ in valid_w],
            widths=0.28, patch_artist=True, showfliers=False, zorder=2,
            medianprops=dict(color=dark_west, lw=1.3),
            boxprops=dict(facecolor=fill_west, edgecolor=color_west, alpha=0.85, lw=1.0),
            whiskerprops=dict(color=color_west, lw=1.0),
            capprops=dict(color=color_west, lw=1.0)
        )

    valid_e = [(pos, d) for pos, d in zip(positions_e, data_e) if len(d) > 0]
    if valid_e:
        ax.boxplot(
            [d for _, d in valid_e],
            positions=[pos for pos, _ in valid_e],
            widths=0.28, patch_artist=True, showfliers=False, zorder=2,
            medianprops=dict(color=dark_east, lw=1.3),
            boxprops=dict(facecolor=fill_east, edgecolor=color_east, alpha=0.85, lw=1.0),
            whiskerprops=dict(color=color_east, lw=1.0),
            capprops=dict(color=color_east, lw=1.0)
        )

    # 2. 叠加每年散点与均值菱形点
    for i, (vals_w, vals_e) in enumerate(zip(data_w, data_e)):
        if len(vals_w) > 0:
            jitter_w = np.random.uniform(-0.065, 0.065, size=len(vals_w))
            ax.scatter(
                np.repeat(positions_w[i], len(vals_w)) + jitter_w, vals_w,
                color=color_west, s=16, alpha=0.9,
                edgecolors='white', linewidths=0.4, zorder=3
            )
            mean_w = np.mean(vals_w)
            ax.scatter(
                positions_w[i], mean_w, color='white', edgecolor=dark_west,
                marker='D', s=26, lw=1.2, zorder=5
            )

        if len(vals_e) > 0:
            jitter_e = np.random.uniform(-0.065, 0.065, size=len(vals_e))
            ax.scatter(
                np.repeat(positions_e[i], len(vals_e)) + jitter_e, vals_e,
                color=color_east, s=16, alpha=0.9,
                edgecolors='white', linewidths=0.4, zorder=3
            )
            mean_e = np.mean(vals_e)
            ax.scatter(
                positions_e[i], mean_e, color='white', edgecolor=dark_east,
                marker='D', s=26, lw=1.2, zorder=5
            )

    # 3. 标注 2002–2024 年（无冰年份）均值文字
    text_bbox = dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.8)

    if len(data_w[1]) > 0:
        mean_w_post = np.mean(data_w[1])
        if not is_date_axis:
            ax.text(positions_w[1] - 0.02, 152, f"{mean_w_post:.0f} d", ha='center', va='bottom',
                    fontsize=8.5, fontweight='bold', color=dark_west, bbox=text_bbox, zorder=6)
        else:
            ax.text(positions_w[1] - 0.04, -7, index_to_date(mean_w_post), ha='center', va='bottom',
                    fontsize=8.0, fontweight='bold', color=dark_west, bbox=text_bbox, zorder=6)

    if len(data_e[1]) > 0:
        mean_e_post = np.mean(data_e[1])
        if not is_date_axis:
            ax.text(positions_e[1] + 0.02, 152, f"{mean_e_post:.0f} d", ha='center', va='bottom',
                    fontsize=8.5, fontweight='bold', color=dark_east, bbox=text_bbox, zorder=6)
        else:
            ax.text(positions_e[1] + 0.06, -7, index_to_date(mean_e_post), ha='center', va='bottom',
                    fontsize=8.0, fontweight='bold', color=dark_east, bbox=text_bbox, zorder=6)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['1979–2001', '2002–2024'], fontsize=9)
    ax.set_xlim(-0.48, 1.48)
    ax.set_ylabel(ylabel, fontsize=9.5, labelpad=3)
    ax.tick_params(axis='both', direction='in', labelsize=8.5, length=3, width=0.9)
    ax.grid(axis='y', linestyle=':', color='#888888', alpha=0.45, linewidth=0.7, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# 左子图：无冰总天数（仅统计出现无冰期的年份）
draw_grouped_box_scatter(ax_days, 'IceFreeDays_West', 'IceFreeDays_East', 'Ice-Free Season Length (d)', is_date_axis=False)
ax_days.set_ylim(-5, 172)
ax_days.set_yticks([0, 50, 100, 150])

# 右子图：首次无冰日期（仅统计出现无冰期的年份）
draw_grouped_box_scatter(ax_onset, 'OnsetIndex_West', 'OnsetIndex_East', 'First Ice-Free Date', is_date_axis=True)
ax_onset.set_ylim(-11, 96)
ax_onset.set_yticks([0, 31, 62, 91])
ax_onset.set_yticklabels(['Jul 01', 'Aug 01', 'Sep 01', 'Sep 30'])

# 顶部共用图例
legend_handles = [
    mpatches.Patch(facecolor=fill_west, edgecolor=color_west, lw=1.0, label='West foraging ground'),
    mpatches.Patch(facecolor=fill_east, edgecolor=color_east, lw=1.0, label='East foraging ground'),
    plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='w',
               markeredgecolor='#333333', markeredgewidth=1.2, markersize=5, label='Mean')
]
fig.legend(
    handles=legend_handles, loc='upper center', bbox_to_anchor=(0.52, 0.98),
    ncol=3, frameon=False, fontsize=8.8, handlelength=1.2, handletextpad=0.4, columnspacing=0.9
)

# =============================
# 4. 导出 600 DPI 超清 PNG 与 PDF 矢量图
# =============================
png_path = r'G:\E\2024seaice\inset_ice_free_boxplot_600dpi.png'
pdf_path = r'G:\E\2024seaice\inset_ice_free_boxplot_vector.pdf'

plt.savefig(png_path, dpi=600, bbox_inches='tight', transparent=True)
plt.savefig(pdf_path, format='pdf', bbox_inches='tight', transparent=True)
print(f"已保存 600 DPI 高清图至: {png_path}")
print(f"已保存 PDF 矢量图至: {pdf_path}")

plt.show()