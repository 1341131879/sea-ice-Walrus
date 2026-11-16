# -*- coding: utf-8 -*-

"""
CMIP6 daily sea-ice edge latitude at 169°W
===========================================================
Algorithm
-----------------------------------------------------------
1. Read daily sea-ice concentration (SIC)
2. Normalize SIC to 0–1
3. Restrict latitude to 60–90°N
4. Generate SIC = 15% contour using ContourPy
   WITHOUT creating matplotlib Figure/Axes
5. Intersect contour with 169°W ruler
6. If multiple intersections exist, select southernmost latitude
7. Special rule:
       Historical 1995–2014:
           June 1 – October 31
           If 169°W does NOT intersect the 15% SIC contour,
           assign edge latitude = 90°N

       2015–2034:
           June 1 – October 31
           If 169°W does NOT intersect the 15% SIC contour,
           assign edge latitude = 90°N

       2035–2054:
           June 1 – October 31
           If 169°W does NOT intersect the 15% SIC contour,
           assign edge latitude = 90°N

       2055–2074:
           May 1 – October 31
           If 169°W does NOT intersect the 15% SIC contour,
           assign edge latitude = 90°N

       2075–2100:
           May 1 – October 31
           If 169°W does NOT intersect the 15% SIC contour,
           assign edge latitude = 90°N

       Outside the corresponding period:
           no intersection → NaN

8. Calculate model climatology:
       period × month × day
9. Calculate multi-model ensemble:
       scenario × period × month × day
10. Ensemble uncertainty:
       10th percentile – 90th percentile

Output
-----------------------------------------------------------
I:\CMIP6_SIday\edge_169W_results
"""

from pathlib import Path
import gc

import numpy as np
import pandas as pd
import xarray as xr

from shapely.geometry import (
    LineString,
    Point,
)
from shapely.ops import unary_union


# ============================================================
# ContourPy
# ============================================================

try:
    from contourpy import contour_generator

except ImportError:

    raise ImportError(
        "\nContourPy is required.\n"
        "Please install it with:\n"
        "    pip install contourpy\n"
    )


# ============================================================
# Basic settings
# ============================================================

DATA_ROOT = Path(
    r"I:\CMIP6_SIday"
)

MODEL_LIST_FILE = (
    DATA_ROOT
    / "CMIP6_plot_ready_models.csv"
)

OUTPUT_DIR = (
    DATA_ROOT
    / "edge_169W_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Scenarios
# ============================================================

HISTORICAL = "historical"

SSP_LIST = [
    "ssp126",
    "ssp245",
    "ssp370",
    "ssp585",
]


# ============================================================
# Time
# ============================================================

START_YEAR = 1995
END_YEAR = 2100

HIST_START = 1995
HIST_END = 2014

SSP_PERIODS = {

    "2015–2034": (
        2015,
        2034,
    ),

    "2035–2054": (
        2035,
        2054,
    ),

    "2055–2074": (
        2055,
        2074,
    ),

    "2075–2100": (
        2075,
        2100,
    ),
}


MONTHS = [
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12
]


# ============================================================
# Sea-ice edge
# ============================================================

TARGET_LON = -169.0

TARGET_LON_360 = (
    TARGET_LON % 360.0
)

LAT_MIN = 60.0
LAT_MAX = 90.0

# ============================================================
# IMPORTANT:
# Keep the original value unchanged.
#
# Although the comment above says 15% SIC contour,
# the original code uses 0.05.
# ============================================================

SIC_THRESHOLD = 0.05


# ============================================================
# Special treatment
# ============================================================

# ============================================================
# Period-specific start dates
#
# 1995–2014 : June 1 – October 31
# 2015–2034 : June 1 – October 31
# 2035–2054 : June 1 – October 31
# 2055–2074 : May 1  – October 31
# 2075–2100 : May 1  – October 31
# ============================================================

SPECIAL_END_MONTH = 12
SPECIAL_END_DAY = 31

NO_INTERSECTION_LAT = 90.0

# Periods for which the special treatment starts on May 1
MAY_START_PERIODS = {
    "2055–2074",
    "2075–2100",
}

# Default start date: June 1
SPECIAL_START_MONTH = 6
SPECIAL_START_DAY = 1

# Early start date: May 1
EARLY_SPECIAL_START_MONTH = 5
EARLY_SPECIAL_START_DAY = 1


# ============================================================
# Ensemble uncertainty
# ============================================================

LOWER_PERCENTILE = 10
UPPER_PERCENTILE = 90


# ============================================================
# Cache tag
# ============================================================

# IMPORTANT:
# The date treatment has changed from the old version.
# Therefore use a new cache tag.

CACHE_TAG = (
    "lat60-90_"
    "sic15_"
    "edge90_"
    "period-specific_"
    "jun01-may01-oct31_"
    "p10-90"
)


# ============================================================
# Utility functions
# ============================================================

def longitude_to_360(lon):
    """
    Convert longitude to 0–360°.
    """

    return np.mod(
        lon,
        360.0,
    )


def normalize_sic(sic):
    """
    Normalize SIC to 0–1.

    If values look like percentages (0–100),
    divide by 100.
    """

    # ========================================================
    # THIS FUNCTION IS KEPT EXACTLY AS IN THE ORIGINAL CODE
    # ========================================================

    sic = np.asarray(
        sic,
        dtype=np.float64,
    )

    finite = np.isfinite(
        sic
    )

    if not finite.any():
        return sic

    p95 = np.nanpercentile(
        sic[finite],
        95,
    )

    if p95 > 1.5:
        sic = sic / 100.0

    return sic


def make_2d_coordinates(
    lat,
    lon,
):
    """
    Convert 1D lat/lon to 2D meshgrid.

    If already 2D, return directly.
    """

    lat = np.asarray(lat)
    lon = np.asarray(lon)

    if (
        lat.ndim == 1
        and lon.ndim == 1
    ):

        lon2d, lat2d = np.meshgrid(
            lon,
            lat,
        )

    elif (
        lat.ndim == 2
        and lon.ndim == 2
    ):

        lat2d = lat
        lon2d = lon

    else:

        raise ValueError(
            f"Unsupported coordinate dimensions: "
            f"lat={lat.shape}, "
            f"lon={lon.shape}"
        )

    return (
        lat2d,
        lon2d,
    )


# ============================================================
# Date helper
# ============================================================

def get_special_start_date(
    period,
):
    """
    Get the start date for the special
    no-intersection -> 90°N rule.

    1995–2014:
        June 1

    2015–2034:
        June 1

    2035–2054:
        June 1

    2055–2074:
        May 1

    2075–2100:
        May 1
    """

    if period in MAY_START_PERIODS:

        return (
            EARLY_SPECIAL_START_MONTH,
            EARLY_SPECIAL_START_DAY,
        )

    return (
        SPECIAL_START_MONTH,
        SPECIAL_START_DAY,
    )


def is_after_special_start(
    month,
    day,
    period,
):
    """
    Whether date is on or after the
    period-specific special start date.
    """

    start_month, start_day = (
        get_special_start_date(
            period
        )
    )

    return (
        (month > start_month)
        or (
            month == start_month
            and day >= start_day
        )
    )


def is_before_special_end(
    month,
    day,
):
    """
    Whether date is on or before October 31.
    """

    return (
        (month < SPECIAL_END_MONTH)
        or (
            month == SPECIAL_END_MONTH
            and day <= SPECIAL_END_DAY
        )
    )


def use_90_if_no_intersection(
    month,
    day,
    period,
):
    """
    Whether the no-intersection case should
    be assigned to 90°N.

    Period-specific rules:

        1995–2014:
            June 1 – October 31

        2015–2034:
            June 1 – October 31

        2035–2054:
            June 1 – October 31

        2055–2074:
            May 1 – October 31

        2075–2100:
            May 1 – October 31
    """

    return (
        is_after_special_start(
            month,
            day,
            period,
        )
        and
        is_before_special_end(
            month,
            day,
        )
    )


# ============================================================
# Variable discovery
# ============================================================

def find_sic_variable(ds):
    """
    Automatically identify sea-ice concentration variable.
    """

    candidates = [
        "siconc",
        "sic",
        "ice_conc",
        "sea_ice_concentration",
    ]

    for name in candidates:

        if name in ds.data_vars:
            return name

    for name in ds.data_vars:

        lname = name.lower()

        if (
            "siconc" in lname
            or "seaice" in lname
            or "sea_ice" in lname
            or lname == "sic"
        ):

            return name

    raise KeyError(
        "Cannot find sea-ice concentration variable. "
        f"Available variables: "
        f"{list(ds.data_vars)}"
    )


def find_latitude(ds):
    """
    Find latitude coordinate.
    """

    candidates = [
        "lat",
        "latitude",
        "nav_lat",
    ]

    for name in candidates:

        if name in ds.coords:
            return name

    for name in candidates:

        if name in ds.variables:
            return name

    for name in ds.variables:

        attrs = ds[name].attrs

        if (
            attrs.get(
                "standard_name"
            )
            == "latitude"
        ):

            return name

    raise KeyError(
        "Cannot find latitude coordinate. "
        f"Available: "
        f"{list(ds.variables)}"
    )


def find_longitude(ds):
    """
    Find longitude coordinate.
    """

    candidates = [
        "lon",
        "longitude",
        "nav_lon",
    ]

    for name in candidates:

        if name in ds.coords:
            return name

    for name in candidates:

        if name in ds.variables:
            return name

    for name in ds.variables:

        attrs = ds[name].attrs

        if (
            attrs.get(
                "standard_name"
            )
            == "longitude"
        ):

            return name

    raise KeyError(
        "Cannot find longitude coordinate. "
        f"Available: "
        f"{list(ds.variables)}"
    )


# ============================================================
# Date handling
# ============================================================

def get_datetime_components(
    time_value,
):
    """
    Extract year/month/day from numpy datetime64 or cftime.
    """

    try:

        return (
            int(time_value.year),
            int(time_value.month),
            int(time_value.day),
        )

    except AttributeError:

        ts = pd.Timestamp(
            time_value
        )

        return (
            ts.year,
            ts.month,
            ts.day,
        )


# ============================================================
# FAST contour extraction
# ============================================================

def extract_edge_from_contour(
    sic,
    lat,
    lon,
    month,
    day,
    period,
    target_lon=TARGET_LON,
    lat_min=LAT_MIN,
    lat_max=LAT_MAX,
    threshold=SIC_THRESHOLD,
):
    """
    Extract southernmost SIC contour intersection at target longitude.

    Algorithm
    ---------
    SIC field
        ↓
    SIC = 15% contour
        ↓
    contour lines
        ↓
    intersect 169°W ruler
        ↓
    all intersection points
        ↓
    southernmost latitude

    Special rule
    ------------
    The special no-intersection rule depends on period:

        1995–2014:
            June 1 – October 31 → 90°N

        2015–2034:
            June 1 – October 31 → 90°N

        2035–2054:
            June 1 – October 31 → 90°N

        2055–2074:
            May 1 – October 31 → 90°N

        2075–2100:
            May 1 – October 31 → 90°N

    Outside the corresponding date range:

        no intersection → NaN
    """

    sic = np.asarray(
        sic,
        dtype=np.float64,
    )

    lat = np.asarray(
        lat
    )

    lon = np.asarray(
        lon
    )

    # --------------------------------------------------------
    # Basic checks
    # --------------------------------------------------------

    if sic.ndim != 2:
        return np.nan

    try:

        lat2d, lon2d = (
            make_2d_coordinates(
                lat,
                lon,
            )
        )

    except Exception:

        return np.nan

    if lat2d.shape != sic.shape:
        return np.nan

    if lon2d.shape != sic.shape:
        return np.nan

    # --------------------------------------------------------
    # Longitude
    # --------------------------------------------------------

    lon2d_360 = longitude_to_360(
        lon2d
    )

    target_lon_360 = (
        target_lon % 360.0
    )

    # --------------------------------------------------------
    # Latitude mask
    # --------------------------------------------------------

    valid_lat = (
        np.isfinite(lat2d)
        &
        (lat2d >= lat_min)
        &
        (lat2d <= lat_max)
    )

    if valid_lat.sum() < 4:
        return np.nan

    # --------------------------------------------------------
    # SIC
    # --------------------------------------------------------

    sic_plot = normalize_sic(
        sic
    )

    sic_plot = sic_plot.copy()

    sic_plot[
        ~np.isfinite(lat2d)
    ] = np.nan

    sic_plot[
        ~np.isfinite(lon2d)
    ] = np.nan

    sic_plot[
        ~valid_lat
    ] = np.nan

    # Remove impossible SIC
    sic_plot[
        (sic_plot < 0.0)
        |
        (sic_plot > 1.0)
    ] = np.nan

    finite = np.isfinite(
        sic_plot
    )

    if finite.sum() < 4:

        if use_90_if_no_intersection(
            month,
            day,
            period,
        ):
            return NO_INTERSECTION_LAT

        return np.nan

    # --------------------------------------------------------
    # Check whether contour is possible
    # --------------------------------------------------------

    sic_min = np.nanmin(
        sic_plot
    )

    sic_max = np.nanmax(
        sic_plot
    )

    if sic_min > threshold:

        # Entire valid domain is above threshold.
        # There is no contour.

        if use_90_if_no_intersection(
            month,
            day,
            period,
        ):
            return NO_INTERSECTION_LAT

        return np.nan

    if sic_max < threshold:

        # Entire valid domain is below threshold.
        # There is no contour.

        if use_90_if_no_intersection(
            month,
            day,
            period,
        ):
            return NO_INTERSECTION_LAT

        return np.nan

    # ========================================================
    # ContourPy
    # ========================================================

    try:

        contour_gen = contour_generator(
            x=lon2d_360,
            y=lat2d,
            z=sic_plot,
            name="serial",
            corner_mask=True,
            line_type="Separate",
            fill_type="OuterCode",
            chunk_size=0,
        )

        contour_lines = (
            contour_gen.lines(
                threshold
            )
        )

    except Exception:

        if use_90_if_no_intersection(
            month,
            day,
            period,
        ):
            return NO_INTERSECTION_LAT

        return np.nan

    if (
        contour_lines is None
        or len(contour_lines) == 0
    ):

        if use_90_if_no_intersection(
            month,
            day,
            period,
        ):
            return NO_INTERSECTION_LAT

        return np.nan

    # --------------------------------------------------------
    # Convert contour → Shapely
    # --------------------------------------------------------

    contour_geometries = []

    for vertices in contour_lines:

        if vertices is None:
            continue

        vertices = np.asarray(
            vertices
        )

        if vertices.ndim != 2:
            continue

        if vertices.shape[0] < 2:
            continue

        if vertices.shape[1] < 2:
            continue

        valid = (
            np.isfinite(
                vertices[:, 0]
            )
            &
            np.isfinite(
                vertices[:, 1]
            )
        )

        vertices = vertices[
            valid
        ]

        if len(vertices) < 2:
            continue

        try:

            line = LineString(
                vertices
            )

            if (
                not line.is_empty
                and line.length > 0
            ):

                contour_geometries.append(
                    line
                )

        except Exception:

            continue

    # --------------------------------------------------------
    # No usable contour
    # --------------------------------------------------------

    if not contour_geometries:

        if use_90_if_no_intersection(
            month,
            day,
            period,
        ):
            return NO_INTERSECTION_LAT

        return np.nan

    # --------------------------------------------------------
    # Merge contours
    # --------------------------------------------------------

    try:

        ice_edge_geom = unary_union(
            contour_geometries
        )

    except Exception:

        if use_90_if_no_intersection(
            month,
            day,
            period,
        ):
            return NO_INTERSECTION_LAT

        return np.nan

    if ice_edge_geom.is_empty:

        if use_90_if_no_intersection(
            month,
            day,
            period,
        ):
            return NO_INTERSECTION_LAT

        return np.nan

    # --------------------------------------------------------
    # 169°W ruler
    # --------------------------------------------------------

    ruler_line = LineString(
        [
            (
                target_lon_360,
                lat_min,
            ),
            (
                target_lon_360,
                lat_max,
            ),
        ]
    )

    # --------------------------------------------------------
    # Intersection
    # --------------------------------------------------------

    try:

        intersection = (
            ruler_line.intersection(
                ice_edge_geom
            )
        )

    except Exception:

        if use_90_if_no_intersection(
            month,
            day,
            period,
        ):
            return NO_INTERSECTION_LAT

        return np.nan

    # --------------------------------------------------------
    # No intersection
    # --------------------------------------------------------

    if intersection.is_empty:

        if use_90_if_no_intersection(
            month,
            day,
            period,
        ):
            return NO_INTERSECTION_LAT

        return np.nan

    # --------------------------------------------------------
    # Collect points
    # --------------------------------------------------------

    points = []

    def collect_points(geom):

        if geom.is_empty:
            return

        geom_type = geom.geom_type

        if geom_type == "Point":

            points.append(
                geom
            )

        elif geom_type in (
            "MultiPoint",
            "GeometryCollection",
        ):

            for g in geom.geoms:

                collect_points(g)

        elif geom_type == "LineString":

            coords = list(
                geom.coords
            )

            for x, y in coords:

                points.append(
                    Point(x, y)
                )

        elif geom_type == "MultiLineString":

            for g in geom.geoms:

                collect_points(g)

    collect_points(
        intersection
    )

    # --------------------------------------------------------
    # No valid points
    # --------------------------------------------------------

    if not points:

        if use_90_if_no_intersection(
            month,
            day,
            period,
        ):
            return NO_INTERSECTION_LAT

        return np.nan

    # --------------------------------------------------------
    # Extract latitudes
    # --------------------------------------------------------

    lats = []

    for p in points:

        if not np.isfinite(
            p.y
        ):
            continue

        if p.y < lat_min:
            continue

        if p.y > lat_max:
            continue

        lats.append(
            float(p.y)
        )

    # --------------------------------------------------------
    # No valid latitude
    # --------------------------------------------------------

    if not lats:

        if use_90_if_no_intersection(
            month,
            day,
            period,
        ):
            return NO_INTERSECTION_LAT

        return np.nan

    # ========================================================
    # Same rule as original SHP algorithm:
    # southernmost intersection
    # ========================================================

    return float(
        min(lats)
    )


# ============================================================
# Find model files
# ============================================================

def find_files_for_model(
    model,
    scenario,
):

    scenario_dir = (
        DATA_ROOT
        / model
        / scenario
    )

    if not scenario_dir.exists():

        print(
            f"[WARNING] Directory not found: "
            f"{scenario_dir}"
        )

        return []

    files = sorted(
        scenario_dir.rglob(
            "*.nc"
        )
    )

    if not files:

        print(
            f"[WARNING] No NetCDF files found: "
            f"{scenario_dir}"
        )

    return files


# ============================================================
# Model climatology
# ============================================================

def calculate_model_period_climatology(
    df
):
    """
    Calculate model climatology.

    Group:
        period × month × day

    Output:
        edge_lat
        year_count
    """

    if df.empty:

        return pd.DataFrame()

    result = (
        df
        .groupby(
            [
                "period",
                "month",
                "day",
                "month_day",
            ],
            as_index=False,
        )
        .agg(
            edge_lat=(
                "edge_lat",
                "mean",
            ),

            year_count=(
                "year",
                "nunique",
            ),
        )
    )

    # --------------------------------------------------------
    # Plot date
    # --------------------------------------------------------

    result[
        "plot_date"
    ] = pd.to_datetime(
        "2000-"
        + result[
            "month_day"
        ],
        format="%Y-%m-%d",
        errors="coerce",
    )

    return result


# ============================================================
# Ensemble statistics
# ============================================================

def calculate_ensemble_statistics(
    model_climatology_list,
    scenario,
):
    """
    Calculate multi-model ensemble statistics.

    Statistics:
        model_count
        mean_lat
        p10_lat
        p90_lat
        std_lat
    """

    if not model_climatology_list:

        return pd.DataFrame()

    all_models = pd.concat(
        model_climatology_list,
        ignore_index=True,
    )

    if all_models.empty:

        return pd.DataFrame()

    # --------------------------------------------------------
    # Scenario
    # --------------------------------------------------------

    all_models[
        "scenario"
    ] = scenario

    # --------------------------------------------------------
    # Ensemble
    # --------------------------------------------------------

    ensemble = (
        all_models
        .groupby(
            [
                "scenario",
                "period",
                "month",
                "day",
                "month_day",
            ],
            as_index=False,
        )
        .agg(

            model_count=(
                "model",
                "nunique",
            ),

            mean_lat=(
                "edge_lat",
                "mean",
            ),

            p10_lat=(
                "edge_lat",
                lambda x:
                np.nanpercentile(
                    x,
                    LOWER_PERCENTILE,
                ),
            ),

            p90_lat=(
                "edge_lat",
                lambda x:
                np.nanpercentile(
                    x,
                    UPPER_PERCENTILE,
                ),
            ),

            std_lat=(
                "edge_lat",
                "std",
            ),
        )
    )

    # --------------------------------------------------------
    # Plot date
    # --------------------------------------------------------

    ensemble[
        "plot_date"
    ] = pd.to_datetime(
        "2000-"
        + ensemble[
            "month_day"
        ],
        format="%Y-%m-%d",
        errors="coerce",
    )

    return ensemble


# ============================================================
# Process one model
# ============================================================

def process_model(
    model,
    scenario,
):

    print()
    print(
        "=" * 70
    )

    print(
        f"Processing model: "
        f"{model} | "
        f"scenario: {scenario}"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Cache
    # --------------------------------------------------------

    daily_cache = (
        OUTPUT_DIR
        / f"{model}_{scenario}_"
        f"daily_{CACHE_TAG}.csv"
    )

    clim_cache = (
        OUTPUT_DIR
        / f"{model}_{scenario}_"
        f"climatology_{CACHE_TAG}.csv"
    )

    # --------------------------------------------------------
    # Load cache
    # --------------------------------------------------------

    if (
        daily_cache.exists()
        and clim_cache.exists()
    ):

        print(
            "[CACHE] Loading existing results:"
        )

        print(
            f"        {daily_cache.name}"
        )

        print(
            f"        {clim_cache.name}"
        )

        daily_df = pd.read_csv(
            daily_cache
        )

        climatology_df = pd.read_csv(
            clim_cache
        )

        return (
            daily_df,
            climatology_df,
        )

    # --------------------------------------------------------
    # Find files
    # --------------------------------------------------------

    files = find_files_for_model(
        model,
        scenario,
    )

    if not files:

        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    print(
        f"Found {len(files)} NetCDF files."
    )

    records = []

    # ========================================================
    # Process NetCDF files
    # ========================================================

    for file_index, nc_file in enumerate(
        files,
        1,
    ):

        print(
            f"[{file_index}/{len(files)}] "
            f"{nc_file.name}"
        )

        try:

            ds = xr.open_dataset(
                nc_file,
                decode_times=True,
                use_cftime=True,
            )

        except Exception as e:

            print(
                f"[WARNING] Cannot open file: "
                f"{e}"
            )

            continue

        try:

            # ------------------------------------------------
            # Variables
            # ------------------------------------------------

            sic_name = (
                find_sic_variable(
                    ds
                )
            )

            lat_name = (
                find_latitude(
                    ds
                )
            )

            lon_name = (
                find_longitude(
                    ds
                )
            )

            sic_var = ds[
                sic_name
            ]

            lat = ds[
                lat_name
            ].values

            lon = ds[
                lon_name
            ].values

            # ------------------------------------------------
            # Time
            # ------------------------------------------------

            if "time" in sic_var.coords:

                time_values = (
                    sic_var[
                        "time"
                    ].values
                )

            elif "time" in ds.coords:

                time_values = (
                    ds[
                        "time"
                    ].values
                )

            else:

                raise KeyError(
                    "No time coordinate found."
                )

            time_dim = (
                sic_var[
                    "time"
                ].dims[0]
            )

            # =================================================
            # Daily loop
            # =================================================

            for t_index, time_value in enumerate(
                time_values
            ):

                # ------------------------------------------------
                # Date
                # ------------------------------------------------

                try:

                    year, month, day = (
                        get_datetime_components(
                            time_value
                        )
                    )

                except Exception:

                    continue

                # ------------------------------------------------
                # Year
                # ------------------------------------------------

                if year < START_YEAR:
                    continue

                if year > END_YEAR:
                    continue

                # ------------------------------------------------
                # Month
                # ------------------------------------------------

                if month not in MONTHS:
                    continue

                # ------------------------------------------------
                # Period
                # ------------------------------------------------

                if scenario == HISTORICAL:

                    if (
                        year < HIST_START
                        or year > HIST_END
                    ):

                        continue

                    period = (
                        "1995–2014"
                    )

                else:

                    period = None

                    for (
                        period_name,
                        (
                            p_start,
                            p_end,
                        ),
                    ) in SSP_PERIODS.items():

                        if (
                            p_start
                            <= year
                            <= p_end
                        ):

                            period = (
                                period_name
                            )

                            break

                    if period is None:
                        continue

                # ------------------------------------------------
                # SIC
                # ------------------------------------------------

                try:

                    sic = (
                        sic_var
                        .isel(
                            {
                                time_dim:
                                t_index
                            }
                        )
                        .values
                    )

                except Exception:

                    continue

                # ------------------------------------------------
                # Edge
                # ------------------------------------------------

                edge_lat = (
                    extract_edge_from_contour(
                        sic=sic,
                        lat=lat,
                        lon=lon,
                        month=month,
                        day=day,
                        period=period,
                        target_lon=TARGET_LON,
                        lat_min=LAT_MIN,
                        lat_max=LAT_MAX,
                        threshold=SIC_THRESHOLD,
                    )
                )

                # ------------------------------------------------
                # Record
                # ------------------------------------------------

                records.append(
                    {
                        "scenario": scenario,

                        "period": period,

                        "year": year,

                        "month": month,

                        "day": day,

                        "month_day":
                            f"{month:02d}-{day:02d}",

                        "edge_lat":
                            edge_lat,

                        "model": model,
                    }
                )

        except Exception as e:

            print(
                f"[WARNING] Error processing "
                f"{nc_file.name}: {e}"
            )

        finally:

            ds.close()

            del ds

            gc.collect()

    # ========================================================
    # Daily DataFrame
    # ========================================================

    daily_df = pd.DataFrame(
        records
    )

    if daily_df.empty:

        print(
            "[WARNING] "
            "No valid edge records."
        )

        return (
            daily_df,
            pd.DataFrame(),
        )

    # --------------------------------------------------------
    # Remove NaN
    #
    # NOTE:
    # 90° records are NOT NaN and therefore remain.
    # --------------------------------------------------------

    valid_count_before = len(
        daily_df
    )

    daily_df = daily_df[
        np.isfinite(
            daily_df[
                "edge_lat"
            ]
        )
    ].copy()

    valid_count_after = len(
        daily_df
    )

    print(
        f"Valid edge records: "
        f"{valid_count_after}/"
        f"{valid_count_before}"
    )

    if daily_df.empty:

        return (
            daily_df,
            pd.DataFrame(),
        )

    # --------------------------------------------------------
    # Remove duplicate dates
    # --------------------------------------------------------

    daily_df = (
        daily_df
        .sort_values(
            [
                "year",
                "month",
                "day",
            ]
        )
        .drop_duplicates(
            subset=[
                "year",
                "month",
                "day",
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Save daily cache
    # --------------------------------------------------------

    daily_df.to_csv(
        daily_cache,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"[SAVE] {daily_cache}"
    )

    # ========================================================
    # Model climatology
    # ========================================================

    climatology_df = (
        calculate_model_period_climatology(
            daily_df
        )
    )

    if climatology_df.empty:

        return (
            daily_df,
            climatology_df,
        )

    # --------------------------------------------------------
    # Model / scenario
    # --------------------------------------------------------

    climatology_df[
        "model"
    ] = model

    climatology_df[
        "scenario"
    ] = scenario

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    climatology_df.to_csv(
        clim_cache,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"[SAVE] {clim_cache}"
    )

    return (
        daily_df,
        climatology_df,
    )


# ============================================================
# Process one scenario
# ============================================================

def process_scenario(
    scenario,
    model_list,
):

    print()
    print(
        "#" * 80
    )

    print(
        f"SCENARIO: {scenario}"
    )

    print(
        "#" * 80
    )

    all_model_climatologies = []

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    for model in model_list:

        try:

            (
                daily_df,
                climatology_df,
            ) = process_model(
                model=model,
                scenario=scenario,
            )

            if not climatology_df.empty:

                all_model_climatologies.append(
                    climatology_df
                )

        except Exception as e:

            print(
                f"[ERROR] "
                f"{model} / "
                f"{scenario}: "
                f"{e}"
            )

    # --------------------------------------------------------
    # No models
    # --------------------------------------------------------

    if not all_model_climatologies:

        print(
            f"[WARNING] "
            f"No valid models for "
            f"{scenario}"
        )

        return

    # ========================================================
    # All-model climatology
    # ========================================================

    all_models = pd.concat(
        all_model_climatologies,
        ignore_index=True,
    )

    all_models_cache = (
        OUTPUT_DIR
        / f"all_models_{scenario}_"
        f"climatology_{CACHE_TAG}.csv"
    )

    all_models.to_csv(
        all_models_cache,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "[SAVE] All model climatology:"
    )

    print(
        f"       {all_models_cache}"
    )

    # ========================================================
    # Ensemble
    # ========================================================

    ensemble = (
        calculate_ensemble_statistics(
            all_model_climatologies,
            scenario=scenario,
        )
    )

    if ensemble.empty:

        print(
            "[WARNING] Ensemble is empty."
        )

        return

    # --------------------------------------------------------
    # Save ensemble
    # --------------------------------------------------------

    ensemble_cache = (
        OUTPUT_DIR
        / f"ensemble_{scenario}_"
        f"169W_{CACHE_TAG}.csv"
    )

    ensemble.to_csv(
        ensemble_cache,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "[SAVE] Ensemble:"
    )

    print(
        f"       {ensemble_cache}"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print(
        f"Models included: "
        f"{all_models['model'].nunique()}"
    )

    print(
        f"Valid ensemble records: "
        f"{len(ensemble)}"
    )

    print()

    print(
        ensemble[
            [
                "scenario",
                "period",
                "month",
                "day",
                "model_count",
                "mean_lat",
                "p10_lat",
                "p90_lat",
                "std_lat",
            ]
        ].head(10)
    )


# ============================================================
# Read model list
# ============================================================

def read_model_list():

    if not MODEL_LIST_FILE.exists():

        raise FileNotFoundError(
            f"\nModel list not found:\n"
            f"{MODEL_LIST_FILE}"
        )

    df = pd.read_csv(
        MODEL_LIST_FILE
    )

    model_col = None

    for col in df.columns:

        if col.lower() == "model":

            model_col = col

            break

    if model_col is None:

        raise KeyError(
            "Cannot find 'model' column "
            f"in {MODEL_LIST_FILE}"
        )

    models = (
        df[model_col]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    return models


# ============================================================
# Main
# ============================================================

def main():

    print()
    print(
        "=" * 80
    )

    print(
        "CMIP6 169°W Sea-Ice Edge Extraction"
    )

    print(
        "=" * 80
    )

    print(
        f"Data root: "
        f"{DATA_ROOT}"
    )

    print(
        f"Output: "
        f"{OUTPUT_DIR}"
    )

    print(
        f"Target longitude: "
        f"{TARGET_LON}°"
    )

    print(
        f"Latitude range: "
        f"{LAT_MIN}–{LAT_MAX}°N"
    )

    print(
        f"SIC threshold: "
        f"{SIC_THRESHOLD * 100:.0f}%"
    )

    print(
        "No-intersection rules:"
    )

    print(
        "  Historical 1995–2014: "
        "June 1 – October 31 → 90°N"
    )

    print(
        "  2015–2034: "
        "June 1 – October 31 → 90°N"
    )

    print(
        "  2035–2054: "
        "June 1 – October 31 → 90°N"
    )

    print(
        "  2055–2074: "
        "May 1 – October 31 → 90°N"
    )

    print(
        "  2075–2100: "
        "May 1 – October 31 → 90°N"
    )

    print(
        "Ensemble uncertainty: "
        f"P{LOWER_PERCENTILE}–"
        f"P{UPPER_PERCENTILE}"
    )

    print(
        f"Months: "
        f"{MONTHS}"
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    model_list = (
        read_model_list()
    )

    print()
    print(
        f"Models found: "
        f"{len(model_list)}"
    )

    for model in model_list:

        print(
            f"  - {model}"
        )

    # ========================================================
    # Historical
    # ========================================================

    print()
    print(
        "=" * 80
    )

    print(
        "PROCESSING HISTORICAL"
    )

    print(
        "=" * 80
    )

    process_scenario(
        scenario=HISTORICAL,
        model_list=model_list,
    )

    # ========================================================
    # SSPs
    # ========================================================

    for scenario in SSP_LIST:

        process_scenario(
            scenario=scenario,
            model_list=model_list,
        )

    # ========================================================
    # Finish
    # ========================================================

    print()
    print(
        "=" * 80
    )

    print(
        "ALL PROCESSING FINISHED"
    )

    print(
        "=" * 80
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        OUTPUT_DIR
    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    main()