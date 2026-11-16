import os
import re
import pandas as pd


# ============================================================
# 0. 参数设置
# ============================================================

# 你的 ESGF wget 脚本
WGET_SCRIPT = r"E:\Walrus\CMIP6\wget_script_2026-9-5_16-10-45.sh"

# 输出目录
OUTPUT_DIR = r"E:\Walrus\CMIP6\choose"

# 输出文件
ALL_FILES_CSV = os.path.join(
    OUTPUT_DIR, "all_parsed_files.csv"
)

SELECTED_FILES_CSV = os.path.join(
    OUTPUT_DIR, "CMIP6_selected_files.csv"
)

MODEL_SUMMARY_CSV = os.path.join(
    OUTPUT_DIR, "CMIP6_model_summary.csv"
)

SELECTED_WGET = os.path.join(
    OUTPUT_DIR, "CMIP6_selected_wget.sh"
)


# ============================================================
# 1. 创建输出目录
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. 读取 wget 脚本
# ============================================================

if not os.path.exists(WGET_SCRIPT):
    raise FileNotFoundError(
        f"找不到 wget 脚本：\n{WGET_SCRIPT}"
    )

with open(WGET_SCRIPT, "r", encoding="utf-8") as f:
    lines = f.readlines()

print("=" * 70)
print("CMIP6 海冰数据筛选")
print("=" * 70)

print(f"\n输入文件：{WGET_SCRIPT}")
print(f"总行数：{len(lines):,}")


# ============================================================
# 3. 解析 wget 文件
#
# 典型格式：
#
# 'siconc_SIday_MIROC6_ssp126_r1i1p1f1_gn_20150101-20241231.nc'
# 'https://...'
# 'SHA256'
# 'xxxxxxxx'
#
# ============================================================

# 第一层：提取 filename / URL / checksum type / checksum
wget_pattern = re.compile(
    r"^'([^']+\.nc)'\s+'([^']+)'\s+'([^']+)'\s+'([^']+)'"
)

# 第二层：从 filename 提取模型、情景和时间
filename_pattern = re.compile(
    r"siconc_SIday_"
    r"(.+?)_"
    r"(historical|ssp126|ssp245|ssp370|ssp585)_"
    r"r1i1p1f1_gn_"
    r"(\d{8})-(\d{8})\.nc$"
)


records = []

for line_number, line in enumerate(lines, start=1):

    line = line.strip()

    m = wget_pattern.match(line)

    if not m:
        continue

    filename, url, checksum_type, checksum = m.groups()

    fm = filename_pattern.match(filename)

    if not fm:
        continue

    model, scenario, start_date, end_date = fm.groups()

    # 注意：
    # 不使用 pandas.to_datetime()
    # 因为有些 CMIP6 数据会延伸到 2300 年，
    # pandas datetime64[ns] 会产生 OutOfBoundsDatetime。
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])

    records.append({
        "line": line_number,
        "filename": filename,
        "url": url,
        "checksum_type": checksum_type,
        "checksum": checksum,
        "model": model,
        "scenario": scenario,
        "start_date": start_date,
        "end_date": end_date,
        "start_year": start_year,
        "end_year": end_year,
    })


df = pd.DataFrame(records)


# ============================================================
# 4. 去重
# ============================================================

before = len(df)

df = (
    df
    .drop_duplicates(subset="filename")
    .reset_index(drop=True)
)

after = len(df)

print(f"\n解析到 NetCDF 文件：{before:,}")
print(f"去重后文件数：{after:,}")

print(f"模型数量：{df['model'].nunique()}")


# ============================================================
# 5. 保存完整解析结果
# ============================================================

df.to_csv(
    ALL_FILES_CSV,
    index=False,
    encoding="utf-8-sig"
)

print(
    f"\n完整文件清单已保存：\n"
    f"{ALL_FILES_CSV}"
)


# ============================================================
# 6. 时间连续性检查函数
#
# 不要求文件必须是一年一个文件。
#
# 例如：
#
# 1850-1859
# 1860-1869
# ...
# 2010-2014
#
# 是连续的。
#
# 即使文件区间重叠，也认为是连续的。
# ============================================================

def check_continuous_coverage(
    data,
    start_year,
    end_year
):
    """
    检查给定文件集合是否完整覆盖
    [start_year, end_year]

    例如：
        historical: 1850–2014
        SSP:        2015–2100

    返回：
        True / False
    """

    intervals = []

    for _, row in data.iterrows():

        s = max(
            int(row["start_year"]),
            start_year
        )

        e = min(
            int(row["end_year"]),
            end_year
        )

        if s <= e:
            intervals.append((s, e))

    if not intervals:
        return False

    # 按起始年份排序
    intervals.sort()

    current_end = start_year - 1

    for s, e in intervals:

        # 出现缺口
        if s > current_end + 1:
            return False

        current_end = max(
            current_end,
            e
        )

        # 已经覆盖到目标年份
        if current_end >= end_year:
            return True

    return current_end >= end_year


# ============================================================
# 7. 获取所有模型
# ============================================================

models = sorted(
    df["model"].unique()
)

SSP_SCENARIOS = [
    "ssp126",
    "ssp245",
    "ssp370",
    "ssp585"
]


# ============================================================
# 8. 开始筛选
#
# 条件：
#
# A. historical 必须完整覆盖 1850–2014
#
# B. SSP 至少有一个完整覆盖 2015–2100
#
# C. 满足条件的 SSP 全部保留
#
# ============================================================

selected_models = {}

summary_records = []

for model in models:

    model_df = df[
        df["model"] == model
    ].copy()

    # --------------------------------------------------------
    # Historical
    # --------------------------------------------------------

    historical_df = model_df[
        model_df["scenario"] == "historical"
    ]

    historical_ok = check_continuous_coverage(
        historical_df,
        1850,
        2014
    )

    # --------------------------------------------------------
    # SSP
    # --------------------------------------------------------

    valid_ssps = []

    for scenario in SSP_SCENARIOS:

        scenario_df = model_df[
            model_df["scenario"] == scenario
        ]

        ssp_ok = check_continuous_coverage(
            scenario_df,
            2015,
            2100
        )

        if ssp_ok:
            valid_ssps.append(scenario)

    # --------------------------------------------------------
    # 最终判断
    # --------------------------------------------------------

    model_ok = (
        historical_ok
        and
        len(valid_ssps) > 0
    )

    if model_ok:

        selected_models[model] = valid_ssps

    summary_records.append({
        "model": model,
        "historical_1850_2014": historical_ok,
        "valid_ssp_count": len(valid_ssps),
        "valid_ssps": ";".join(valid_ssps),
        "qualified": model_ok,
    })


summary_df = pd.DataFrame(
    summary_records
)

summary_df.to_csv(
    MODEL_SUMMARY_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 9. 输出筛选结果
# ============================================================

print("\n")
print("=" * 70)
print("模型筛选结果")
print("=" * 70)

for model in models:

    row = summary_df[
        summary_df["model"] == model
    ].iloc[0]

    historical_ok = row[
        "historical_1850_2014"
    ]

    valid_ssps = row[
        "valid_ssps"
    ]

    qualified = row[
        "qualified"
    ]

    if qualified:

        print(
            f"✓ {model:<25} "
            f"historical + {valid_ssps}"
        )

    else:

        if historical_ok:

            print(
                f"✗ {model:<25} "
                f"historical ✓，"
                f"但没有完整 SSP"
            )

        else:

            print(
                f"✗ {model:<25} "
                f"historical 不完整"
            )


# ============================================================
# 10. 根据筛选结果保留全部文件
#
# 注意：
#
# 如果一个模型：
#
# historical
# ssp126
# ssp245
# ssp370
# ssp585
#
# 全部完整，
# 那么五套全部保留。
#
# 如果只有：
#
# historical
# ssp370
#
# 那么 historical + ssp370 全部保留。
#
# ============================================================

selected_records = []

for model, valid_ssps in selected_models.items():

    valid_scenarios = [
        "historical"
    ] + valid_ssps

    model_selected = df[
        (df["model"] == model)
        &
        (df["scenario"].isin(valid_scenarios))
    ].copy()

    selected_records.append(
        model_selected
    )


if selected_records:

    selected_df = pd.concat(
        selected_records,
        ignore_index=True
    )

else:

    selected_df = pd.DataFrame(
        columns=df.columns
    )


# ============================================================
# 11. 保存最终文件清单
# ============================================================

selected_df.to_csv(
    SELECTED_FILES_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 12. 生成一个新的 wget 脚本
#
# 这个脚本以后可以直接用于下载。
#
# 但是这里暂时不下载。
# ============================================================

with open(
    SELECTED_WGET,
    "w",
    encoding="utf-8"
) as f:

    f.write("#!/bin/bash\n")
    f.write("\n")
    f.write("# =====================================================\n")
    f.write("# CMIP6 sea-ice concentration selected files\n")
    f.write("# Historical: 1850-2014\n")
    f.write("# SSP:        2015-2100\n")
    f.write("# =====================================================\n")
    f.write("\n")

    for _, row in selected_df.iterrows():

        f.write(
            f"'{row['filename']}' "
            f"'{row['url']}' "
            f"'{row['checksum_type']}' "
            f"'{row['checksum']}'\n"
        )


# ============================================================
# 13. 统计结果
# ============================================================

print("\n")
print("=" * 70)
print("最终筛选统计")
print("=" * 70)

print(
    f"\n符合条件的模型："
    f"{len(selected_models)}"
)

print(
    f"最终保留文件："
    f"{len(selected_df):,}"
)

print("\n各模型情况：")

for model, valid_ssps in selected_models.items():

    model_df = selected_df[
        selected_df["model"] == model
    ]

    print(
        f"  {model:<25} "
        f"{'historical + ' + ' + '.join(valid_ssps):<45} "
        f"{len(model_df):>4} files"
    )


# ============================================================
# 14. 检查最终结果是否真的完整
# ============================================================

print("\n")
print("=" * 70)
print("最终时间连续性检查")
print("=" * 70)

all_checks_passed = True

for model, valid_ssps in selected_models.items():

    model_df = selected_df[
        selected_df["model"] == model
    ]

    # Historical
    historical_ok = check_continuous_coverage(
        model_df[
            model_df["scenario"] == "historical"
        ],
        1850,
        2014
    )

    print(
        f"\n{model}"
    )

    print(
        f"  historical 1850-2014 : "
        f"{'OK' if historical_ok else 'FAILED'}"
    )

    if not historical_ok:
        all_checks_passed = False

    # SSP
    for scenario in valid_ssps:

        ssp_ok = check_continuous_coverage(
            model_df[
                model_df["scenario"] == scenario
            ],
            2015,
            2100
        )

        print(
            f"  {scenario:<8} 2015-2100 : "
            f"{'OK' if ssp_ok else 'FAILED'}"
        )

        if not ssp_ok:
            all_checks_passed = False


# ============================================================
# 15. 最终结果
# ============================================================

print("\n")
print("=" * 70)

if all_checks_passed:

    print(
        "✓ 所有筛选后的模型/情景均通过时间连续性检查"
    )

else:

    print(
        "✗ 存在时间连续性问题，请检查上面的结果"
    )

print("=" * 70)

print(
    f"\n模型汇总：\n"
    f"{MODEL_SUMMARY_CSV}"
)

print(
    f"\n最终文件清单：\n"
    f"{SELECTED_FILES_CSV}"
)

print(
    f"\n筛选后的 wget 脚本：\n"
    f"{SELECTED_WGET}"
)

print("\n筛选完成。当前程序没有下载任何 NetCDF 数据。")