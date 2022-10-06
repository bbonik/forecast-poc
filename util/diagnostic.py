"""Diagnostic utilities for validating data prepared for Amazon Forecast"""

# Python Built-Ins:
from collections import defaultdict
import contextlib
import csv
import json
import math
import os
import time
from types import SimpleNamespace
from typing import Iterable, List, Tuple, Union

# External Dependencies:
import dateutil
from IPython.display import display, Markdown
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sqlite3

# Local Dependencies
from . import fcst_utils as fcst
from . import notebook_utils as notebook


# Configuration:
CHUNKSIZE = 50000  # Max records per file processed in one go - reduce to cut memory consumption, lower speed
WARN_THRESH_MIN_ITEMS = 10  # Warn if number of distinct timeseries to forecast is <N
WARN_THRESH_LOGLOG_HEAD_HEAVY = -2  # Warn that tail items may be sparse if Pareto log-log less than this.
EXTENT_BREAKPOINTS = [
    # Reported range extent breakpoints [0] as proportion of global history, and thresholds [1] to trigger
    # warnings when less than N% of items reach the breakpoint. [2] (if warning triggered) is additional msg.
    (1., 1., "This could indicate data timestamp alignment issues if not expected"),
    (.9, None),
    (.5, None),
    (.05, None),
    (.01, 1., "Forecasting accuracy may be poor for items with very short histories"),
]
WARN_AVG_EXTENT_THRESHS = [
    # Warnings to trigger if the average timestep extent across items falls below `thresh`.
    # Only the first matched warning is displayed (order of decresing severity).
    SimpleNamespace(
        thresh=25,  # 24 could = 2yrs of monthly data
        level="danger",
        details=(
            "On average, there's very little history available from which to learn the time dynamics of",
            "each series. Ability to predict seasonal/cyclical effects will likely be limited, and simple",
            "statistical baseline methods like ARIMA or ETS may produce better results than deep learning",
            "algorithms. Consider modelling at a smaller time granularity or adding more historical data.",
        ),
    ),
    SimpleNamespace(
        thresh=200, # 365/2 could be half-year daily data
        level="warning",
        details=(
            "On average, there's limited history available from which to learn the time dynamics of each",
            "series. Ability to predict seasonal/cyclical effects may be limited, and the power of advanced",
            "algorithms may be limited. Consider modelling at a smaller time granularity or adding more",
            "historical data.",
        ),
    ),
    SimpleNamespace(
        thresh=500,
        level="warning",
        details=(
            "It's likely that accuracy could be improved by modelling at a smaller time granularity or",
            "adding more historical data. Ideally historical data would cover 3-5 (or more) cycles of the",
            "longest important seasonality in the data (annual, in many cases e.g. retail).",
        ),
    ),
]
SPAN_BREAKPOINTS = [
    # Reported contiguous span extent breakpoints [0] as proportion of that item's span, and thresholds [1]
    # to trigger warnings when less than N% of items reach the breakpoint. [2] (if warning triggered) is
    # additional context
    (1., 1., "At least some of your items have gaps (missing timesteps between first and last records)."),
    (.9, None),
    (.5, .8, "\n".join((
        "A significant number of your items have sparse data between their first and last timestamps.",
        "Be sure to configure Forecast's",
        '<a href="https://docs.aws.amazon.com/forecast/latest/dg/howitworks-missing-values.html">',
        "missing value filling</a> logic appropriately.",
    ))),
    (.1, None),
    (.01, 1., "At least of your items are extremely sparse between their first and last timestamps."),
]
STEPCOUNT_REFERENCES = [
    # Reference time-step counts to plot as levels on log-log charts
    ("300", 300),
    ("500", 500),
    ("1k", 1000)
]


def sniff_csv_file(filepath: str) -> Tuple[Union[List[str], None], int]:
    """Examine the start of a CSV file to test metadata

    Returns
    -------
    headers :
        List of column name strs, or None if the file seems not to have headers
    ncols :
        Number of columns in the data (inferred from start of file, assumed consistent)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        row1 = next(reader)
        ncols = len(row1)
        # If any cells in first row don't fit the valid header type, assume not a header
        # TODO: Improve this - probably breaks for metadata files
        if not all(fcst.SchemaAttribute.is_valid_name(cell) for cell in row1):
            return None, ncols
        else:
            return row1, ncols


def bin_timestamps_to_frequency(timestamps: pd.Series, freq: str) -> pd.Series:
    """Map string or datetime timestamps series to frequency bins for Forecast

    Parameters
    ----------
    timestamps : pd.Series
        Accepts series with either string or datetime dtypes
    freq :
        A valid Forecast frequency per 'fcst_utils.FREQUENCIES' config variable.

    Returns
    -------
    binned_timestamps :
        Pandas series of a datetime type
    """
    fcst.validate_forecast_frequency(freq)
    freq_spec = fcst.FREQUENCIES[freq]
    dtype = timestamps.dtype
    if pd.api.types.is_datetime64_any_dtype(dtype):
        # Series is datetimes: Apply the datetime mapper fn
        return freq_spec["dt_series_mapper"](timestamps)
    elif pd.api.types.is_string_dtype(dtype):
        return freq_spec["dt_series_mapper"](pd.to_datetime(timestamps))
    else:
        raise ValueError(f"Series does not seem to contain timestamps: dtype={dtype}")


def validate_tts_schema_on_domain(
    tts_schema,
    domain: str,
    is_tts_schema_explicit: bool=True
) -> Tuple[List[str], List[str], List[str]]:
    """Validate that a target time-series schema conforms to an Amazon Forecast domain

    Parameters
    ----------
    tts_schema :
        Schema doc as would be passed to boto3
    domain :
        Forecast domain name
    is_tts_schema_explicit : (Optional)
        Set 'False' to annotate that schema is inferred when raising errors

    Returns
    -------
    required_fields : List[str]
        List of names of fields required by the domain
    optional_fields : List[str]
        List of names used optional fields in the domain (where types match specification)
    custom_fields : List[str]
        List of names of used custom fields not specified by the domain
    """
    domain_spec = fcst.DOMAINS[domain]
    for fname in domain_spec.tts.required_fields:
        matching_fields = [f for f in tts_schema["Attributes"] if f["AttributeName"] == fname]
        if len(matching_fields) < 1:
            raise ValueError(
                "{}TTS schema is missing required field '{}' for domain '{}'".format(
                    "" if is_tts_schema_explicit else "Inferred ",
                    fname,
                    domain,
                ),
            )
        elif (
            matching_fields[0]["AttributeType"]
            != domain_spec.tts.required_fields[fname].AttributeType
        ):
            raise ValueError(" ".join((
                "{}TTS schema has type '{}' for required field '{}'".format(
                    "" if is_tts_schema_explicit else "Inferred ",
                    fname,
                    domain,
                ),
                "which domain '{}' specifies as '{}'".format(
                    domain,
                    domain_spec.tts.required_fields[fname].AttributeType,
                )
            )))
    optional_fields_used = []
    for fname in domain_spec.tts.optional_fields:
        try:
            matching_field = next(
                f for f in tts_schema["Attributes"] if f["AttributeName"] == fname
            )
        except StopIteration:
            continue
        if (
            matching_field["AttributeType"]
            == domain_spec.tts.optional_fields[fname].AttributeType
        ):
            optional_fields_used.append(fname)
        else:
            # TODO: Warning instead
            print(" ".join((
                f"WARNING: Field '{fname}', which domain '{domain}' specifies as optional with",
                "type '{}', has been used with different type '{}'.".format(
                    domain_spec.tts.optional_fields[fname].AttributeType,
                    matching_field["AttributeType"],
                ),
                "Consider changing field name or using this optional field per the domain spec.",
            )))
    schema_fnames = [f["AttributeName"] for f in tts_schema["Attributes"]]
    required_fnames = list(domain_spec.tts.required_fields.keys())
    custom_fnames = [
        f for f in schema_fnames
        if f not in (optional_fields_used + list(domain_spec.tts.required_fields.keys()))
    ]
    print("\n".join((
        f"Validated schema conforms to domain '{domain}' with:",
        f"Required fields {required_fnames}",
        f"Optional fields {optional_fields_used}",
        f"Custom fields {custom_fnames}",
    )))
    return required_fnames, optional_fields_used, custom_fnames


def plot_loglog(
    val: Iterable[Union[int, float]],
    quantity: str="value",
    instance_units: str="examples",
    show: bool=True,
    ref_yvals: Iterable[Tuple[str, Union[int, float]]]=None
) -> Union[Tuple[float], None]:
    """Plot a log-log describing a Pareto distribution, and fit a line characterizing the distribution

    Count data over items (e.g. sales by product) often follows an approximate Pareto distribution, with
    most volume concentrated in a small number of items and a "long tail" of lower-volume items. On a log-log
    graph the Pareto distribution is a straight line, with slope characterising the degree of "concentration"
    of the value in few items.

    Parameters
    ----------
    val :
        A list/series/whatever of positive numeric values (often counts)
    quantity : (Optional)
        A (singular) name of the quantity measured in val (defaults to 'value')
    instance_units : (Optional)
        A (plural) name describing the entries in val (defaults to 'examples')
    show : (Optional)
        Whether to call pyplot.show() when the plot is ready (defaults to True)
    ref_yvals : (Optional)
        A sequence of [name, value] pairs to plot as reference thresholds on the quantity (y) axis

    Returns
    -------
    (slope, intercept, RMSE) of model fit if the plot could be constructed, else None
    """
    if len(val) < 2:
        display(Markdown("*(Skipping log-log plot: Insufficient data to graph)*"))
        return
    x = 1+np.arange(len(val))
    slope, intercept = np.polyfit(np.log10(x)[val>0], np.log10(val[val>0]), deg=1)
    full_ret = np.polyfit(np.log10(x)[val>0], np.log10(val[val>0]), deg=1, full=True)
    rmse = np.mean(full_ret[0]**2)**0.5
    fitted = 10**(intercept + slope*np.log10(x))
    fig = plt.figure()
    ax = fig.gca()
    ax.loglog(x, val)
    ax.loglog(x, fitted, ':')
    ax.grid()
    ax.annotate(
        f"{10**intercept:.2e} * x^({slope:.2f})\nRMSE={rmse:.2f}",
        xy=(0.3, 0.05),
        xycoords="axes fraction",
        bbox={ "edgecolor": "black", "facecolor": "#eeeeee" },
    )
    ax.set_title(f"Distribution of {instance_units} by {quantity} (log-log)")
    ax.set_ylabel(f"Min {quantity}")
    ax.set_xlabel(f"# {instance_units}")
    
    if ref_yvals:
        ylims = ax.get_ylim()
        for k,v in ref_yvals:
            if ylims[0] < v < ylims[1]:
                plt.plot(plt.xlim(), [v,v], "--")
                plt.text(plt.xlim()[0], v, k)
    if show:
        plt.show()
    return (slope, intercept, rmse)


def add_pct_to_value_counts(value_counts: pd.Series, clip: Union[int, None]=None) -> pd.DataFrame:
    """Convert a Pandas value_counts output (series) to a displayable dataframe with (string) % column"""
    n_entries = value_counts.sum()
    result = value_counts[:clip].to_frame("Records")
    result["Percentage"] = result["Records"] / n_entries
    # Convert to % string representation:
    result["Percentage"] = pd.Series(
        ["{0:.2f}%".format(val * 100) for val in result["Percentage"]],
        index=result.index,
    )
    return result


def diagnose(
    tts_path: str,
    frequency: Union[str, None]=None,
    domain: Union[str, None]=None,
    tts_schema=None,
# TODO: Implement analysis of other datasets too
#     rts_path: Union[str, None]=None,
#     metadata_path: Union[str, None]=None,
) -> None:
    """Perform a variety of analyses and checks on prepared data for Amazon Forecast, displaying to notebook

    Parameters
    ----------
    tts_path :
        Local path to target time-series data CSV or folder of CSVs.
    frequency : (Optional, recommended)
        The 'ForecastFrequency' string per Forecast's CreatePredictor API. If this parameter is omitted, the
        diagnostic cannot analyze the time range extents spanned and contiguous ranges covered for items:
        Decreasing RAM requirement but also usefulness!
    domain : (Optional)
        'Domain' for the dataset group as configured in Amazon Forecast per
        https://docs.aws.amazon.com/forecast/latest/dg/howitworks-domains-ds-types.html (May be omitted and
        inferred automatically if possible from column headers / provided schemas).
    tts_schema : (Optional)
        Schema dict for the target time-series, as defined in the Forecast docs (i.e. containing key
        'Attributes').
    """
    # Local TZ would be nice, but it seems to behave a bit unpredictably... UTC will do:
    print(f"Analysis started at {pd.Timestamp.utcnow()}")

    is_tts_schema_explicit = tts_schema is not None
    is_domain_explicit = domain is not None
    if is_domain_explicit and domain not in fcst.DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in supported list {list(fcst.DOMAINS.keys())}")
    if frequency is not None:
        fcst.validate_forecast_frequency(frequency)

    if is_domain_explicit and is_tts_schema_explicit:
        # Validate explicit schema conforms to target domain
        reqd_fields, optional_fields, custom_fields = validate_tts_schema_on_domain(
            tts_schema,
            domain,
            is_tts_schema_explicit
        )
        timestamp_field = next(
            f["AttributeName"] for f in tts_schema["Attributes"]
            if f["AttributeName"] in reqd_fields and f["AttributeType"] == "timestamp"
        )
        target_field = fcst.DOMAINS[domain].target_field if domain is not None else next(
            f["AttributeName"] for f in tts_schema["Attributes"]
            if f["AttributeName"] in reqd_fields and f["AttributeType"] not in ("timestamp", "string")
        )
        dimension_fields = [
            f["AttributeName"] for f in tts_schema["Attributes"]
            if f["AttributeName"] not in (timestamp_field, target_field)
        ]

    if os.path.isdir(tts_path):
        tts_filenames = sorted(
            os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(tts_path)) for f in fn
        )
        filtered_filenames = sorted(filter(lambda f: f.lower().endswith(".csv"), tts_filenames))
        n_raw = len(tts_filenames)
        n_filtered = len(filtered_filenames)
        if n_filtered > 0 and n_filtered < n_raw:
            print(f"Ignoring {n_raw - n_filtered} non-CSV files in TTS directory")
            tts_filenames = filtered_filenames
    elif os.path.isfile(tts_path):
        tts_filenames = [tts_path]
    else:
        raise ValueError(f"tts_path must be a valid local file or directory, got: {tts_path}")
    print("Found {} target time-series files:\n    {}".format(
        len(tts_filenames),
        "\n    ".join(tts_filenames[:10] + (["...etc."] if len(tts_filenames) > 10 else []))
    ))

    # Tracking variables for analysis:
    n_chunks_global = 0  # Total number of batches across all files
    global_tts_start = None
    global_tts_end = None
    total_records = 0
    total_records_nonulls = 0
    num_nulls_by_field = None
    unique_dimension_vals = {}
    unique_dimension_combos = None
    total_ranges = None  # Needs to be initialized once dimension_fields is known
    # The most records we've ever seen aggregated/mapped to a timestep-[dimensions] bucket (which is a
    # lower bound, because we only see the contents of one chunk at a time):
    max_chunk_aggregated_records = 0

    for tts_filename in tts_filenames:
        header_columns, ncols = sniff_csv_file(tts_filename)

        # If schema has been explicitly provided or already inferred from previous file, validate column
        # names against the list:
        if tts_schema is not None:
            if len(tts_schema["Attributes"]) != ncols:
                raise ValueError("{}TTS schema specified {} attributes, but {} were present in {}".format(
                    "" if is_tts_schema_explicit else "Inferred ",
                    len(tts_schema["Attributes"]),
                    ncols,
                    tts_filename,
                ))
            # If the CSV has header and the schema was provided, check the fields match:
            if header_columns is not None:
                for ixcol, header in enumerate(header_columns):
                    if header != tts_schema["Attributes"][ixcol]["AttributeName"]:
                        raise ValueError(" ".join((
                            "{}TTS schema column {} is {},".format(
                                "" if is_tts_schema_explicit else "Inferred ",
                                ixcol,
                                tts_schema["Attributes"][ixcol]["AttributeName"],
                            ),
                            f"but found {header} at this location in file {tts_filename}",
                            "- Column orders must match between TTS schema and all CSVs.",
                        )))
        tts_chunker = pd.read_csv(
            tts_filename,
            chunksize=CHUNKSIZE,
            header=None if header_columns is None else 0,
            names=[f["AttributeName"] for f in tts_schema["Attributes"]] if tts_schema else None,
            dtype={
                f["AttributeName"]: fcst.SchemaAttribute.type_to_numpy_type(f["AttributeType"])
                for f in tts_schema["Attributes"]
            } if tts_schema else None,
        )

        # Next we'll read actual file contents and infer data types, so it might be that we're able to infer
        # column names from the provided Domain where neither CSV headers or tts_schema were provided. This
        # 'renames' dict will track the mapping from automated Pandas colnames to "true" headings for future
        # chunks:
        renames = None if tts_schema or header_columns else {}

        # TODO: Progress bar instead of prints
        for ixchunk, tts_chunk in enumerate(tts_chunker):
            print(f"Processing file chunk {ixchunk} (global chunk {n_chunks_global})")
            n_chunks_global += 1
            if tts_schema is None:
                # TTS schema was not explicitly provided and hasn't been inferred yet - infer from data.
                tts_schema = {
                    "Attributes": []
                }
                field_counts_by_type = defaultdict(int)
                for ixcol, col in enumerate(tts_chunk):
                    dtype = tts_chunk[col].dtype
                    if pd.api.types.is_integer_dtype(dtype):
                        schematype = "integer"
                    elif pd.api.types.is_float_dtype(dtype):
                        schematype = "float"
                    elif pd.api.types.is_string_dtype(dtype):
                        try:
                            pd.to_datetime(tts_chunk[col][0:5], infer_datetime_format=True)
                            schematype = "timestamp"
                        except (pd.errors.ParserError, dateutil.parser.ParserError):
                            schematype = "string"
                    else:
                        raise ValueError(
                            f"Unexpected pandas dtype {dtype} at column {ixcol} ({col}) of {tts_filename}"
                        )
                    field_counts_by_type[schematype] += 1
                    tts_schema["Attributes"].append({
                        "AttributeName": col,
                        "AttributeType": schematype,
                    })
                if header_columns is None:
                    # Try to infer column names from types, if missing:
                    if domain is None:
                        # TODO: Infer from field type counts? Only works in very few cases
                        raise NotImplementedError(
                            "domain must be provided when tts_schema is not and source files have no headers"
                        )
                    # For data types where there's exactly one matching field in the data and in the
                    # domain schema, we can infer correspondence.
                    for schematype in field_counts_by_type:
                        if field_counts_by_type[schematype] > 1:
                            raise ValueError(" ".join([
                                "Cannot infer column names from domain and detected data types:",
                                "{} (>1) fields in input have detected type '{}'".format(
                                    field_counts_by_type[schematype],
                                    schematype,
                                ),
                            ]))
                        else:  # Implicitly =1 due to defaultdict, so the below next() will work
                            attribute = next(
                                a for a in tts_schema["Attributes"] if a["AttributeType"] == schematype
                            )
                            matching_required_fields = [
                                f for f in fcst.DOMAINS[domain].tts.required_fields
                                if fcst.DOMAINS[domain].tts.required_fields[f].AttributeType == schematype
                            ]
                            n_matching_required = len(matching_required_fields)
                            if n_matching_required > 1:
                                raise ValueError(" ".join((
                                    f"Domain {domain} requires {n_matching_required} fields of type",
                                    f"{schematype}, but data contains only one.",
                                )))
                            matching_optional_fields = [
                                f for f in fcst.DOMAINS[domain].tts.optional_fields
                                if fcst.DOMAINS[domain].tts.optional_fields[f].AttributeType == schematype
                            ]
                            n_matching_optional = len(matching_optional_fields)
                            if n_matching_required == 1:
                                renames[attribute["AttributeName"]] = matching_required_fields[0]
                                attribute["AttributeName"] = matching_required_fields[0]
                            elif n_matching_required == 0 and n_matching_optional == 1:
                                renames[attribute["AttributeName"]] = matching_optional_fields[0]
                                attribute["AttributeName"] = matching_optional_fields[0]
                print(f"Inferred target time-series schema:\n{json.dumps(tts_schema, indent=2)}")
                reqd_fields, optional_fields, custom_fields = validate_tts_schema_on_domain(
                    tts_schema,
                    domain,
                    is_tts_schema_explicit
                )
                timestamp_field = next(
                    f["AttributeName"] for f in tts_schema["Attributes"]
                    if f["AttributeName"] in reqd_fields and f["AttributeType"] == "timestamp"
                )
                target_field = fcst.DOMAINS[domain].target_field if domain is not None else next(
                    f["AttributeName"] for f in tts_schema["Attributes"]
                    if f["AttributeName"] in reqd_fields
                    and f["AttributeType"] not in ("timestamp", "string")
                )
                dimension_fields = [
                    f["AttributeName"] for f in tts_schema["Attributes"]
                    if f["AttributeName"] not in (timestamp_field, target_field)
                ]
            # endif tts_schema is None: tts_schema has now been successfully inferred or error raised.

            if renames is not None:
                tts_chunk.rename(columns=renames, inplace=True)

            if total_ranges is None:
                total_ranges = pd.DataFrame()
                for f in dimension_fields:
                    total_ranges[f] = pd.Series([], dtype=tts_chunk[f].dtype)
                total_ranges["starts"] = pd.Series([], dtype="datetime64[ns]")
                total_ranges["ends"] = pd.Series([], dtype="datetime64[ns]")

            # Update statistics from this chunk:
            chunk_min_ts = tts_chunk[timestamp_field].min()
            chunk_max_ts = tts_chunk[timestamp_field].max()
            global_tts_start = chunk_min_ts if global_tts_start is None else min(
                (global_tts_start, chunk_min_ts)
            )
            global_tts_end = chunk_max_ts if global_tts_end is None else max(
                (global_tts_end, chunk_max_ts)
            )
            total_records += len(tts_chunk)
            total_records_nonulls += len(tts_chunk) - tts_chunk.isnull().any(axis=1).sum()

            chunk_nulls_by_field = tts_chunk.isna().sum()
            num_nulls_by_field = chunk_nulls_by_field if num_nulls_by_field is None else (
                num_nulls_by_field + chunk_nulls_by_field
            )
            for fname in dimension_fields:
                chunk_unique_vals = tts_chunk[fname].value_counts(dropna=False)
                if fname in unique_dimension_vals:
                    unique_dimension_vals[fname] = unique_dimension_vals[fname].add(
                        chunk_unique_vals,
                        fill_value=0,  # Need to use this or we get NaN for combos not in the chunk
                    ).astype(int)  # For some reason this add() converts to float by default
                else:
                    unique_dimension_vals[fname] = chunk_unique_vals
            chunk_dimension_combos = tts_chunk.groupby(dimension_fields).size()
            if unique_dimension_combos is None:
                unique_dimension_combos = chunk_dimension_combos
            else:
                unique_dimension_combos = unique_dimension_combos.add(
                    chunk_dimension_combos,
                    fill_value=0,  # Need to use this or we get NaN for combos not in the chunk
                ).astype(int)  # For some reason this add() converts to float by default

            if frequency is not None:
                # In this section, we'll construct/update the list of observed contiguous ranges.
                freq_spec = fcst.FREQUENCIES[frequency]

                # First map the raw timestamps to their bin locations, and calculate one step back:
                binned_timestamps = bin_timestamps_to_frequency(tts_chunk[timestamp_field], frequency)
                binned_timestamps.name = "binned_timestamps"
                prev_timestamps = binned_timestamps - freq_spec["dt_offset"]
                prev_timestamps.name = "prev_timestamps"

                # Count the number of valid (non-blank target field) records in each bin, and update the
                # metric for most records ever seen aggregated to a single bin:
                record_counts = pd.concat(
                    [binned_timestamps, tts_chunk[dimension_fields + [target_field]]],
                    axis=1,
                ).groupby(["binned_timestamps"] + dimension_fields).count()
                record_counts = record_counts[record_counts[target_field] > 0]
                max_chunk_aggregated_records = max(
                    max_chunk_aggregated_records,
                    record_counts[target_field].max()
                )

                # Offset the index and self-join to calculate which bins don't have any data at the timestep
                # directly preceding them, and therefore are the `start` of a contiguous run:
                prev_count = record_counts.copy().rename(columns={ target_field: "prev" })
                prev_count.index = prev_count.index.set_levels(
                    prev_count.index.levels[0].shift(
                        periods=freq_spec["dt_periods"],
                        freq=freq_spec["dt_freq"]
                    ),
                    level=0,
                )
                starts = record_counts.join(prev_count)
                starts = starts[starts["prev"].isna()].reset_index()[
                    ["binned_timestamps"] + dimension_fields
                ].rename(columns={ "binned_timestamps": "starts" })

                # ...And do the same thing with opposite offset to find the `ends` of a contiguous run:
                next_count = record_counts.copy().rename(columns={ target_field: "next" })
                next_count.index = next_count.index.set_levels(
                    next_count.index.levels[0].shift(
                        periods=-1 * freq_spec["dt_periods"],
                        freq=freq_spec["dt_freq"]
                    ),
                    level=0,
                )
                ends = record_counts.join(next_count)
                ends = ends[ends["next"].isna()].reset_index()[
                    ["binned_timestamps"] + dimension_fields
                ].rename(columns={"binned_timestamps": "ends"})

                # Merge the two together to describe the contiguous ranges in the dataset:
                # Every start necessarily has a corresponding end, so it should be sufficient to join all
                # end dates >= each start date and then just pick the earliest one - as we do here.
                ranges = pd.merge(starts, ends, on=dimension_fields)
                ranges = ranges[
                    ranges["starts"] <= ranges["ends"]
                ].groupby(dimension_fields + ["starts"]).min().reset_index()

                # Because we're processing the data in chunks, we now need to take on the trickier task of
                # *consolidating* these new detected time ranges with whatever we might have seen before. We
                # don't want to encode any assumptions about how users have sharded up files or sorted their
                # data - so we don't know much about hoow these new ranges might overlap with existing.
                #
                # We know that ranges should be consolidated when they're on *adjacent* timesteps (not just
                # overlapping), so will start by calculating fields to join on for that and then appending
                # the new ranges to the existing list:
                ranges["prestarts"] = ranges["starts"] - freq_spec["dt_offset"]
                ranges["postends"] = ranges["ends"] + freq_spec["dt_offset"]

                total_ranges = total_ranges.append(ranges)

                # Consolidating detected ranges is an iterative process because a consolidation could bring
                # two previously separate ranges into overlap.
                # TODO: Is there an upper bound on the # iterations required for combining two sane sets?
                prev_n_ranges = float("inf")

                # Pandas only supports equality joins, with inequality joins implemented via merge() and then
                # filtering the result set. This is memory-inefficient for big chunk sizes we'd like to 
                # process, so we'll use an in-mem SQLite connection to do the join in SQL instead.
                with contextlib.closing(sqlite3.connect(":memory:")) as conn: # Auto-close when done
                    # Could also consider using:
                    # with conn: (to auto-commit transaction)
                    # ...but we don't actually want the overhead of committing.
                    while prev_n_ranges > len(total_ranges):
                        prev_n_ranges = len(total_ranges)
                        total_ranges.to_sql("total_ranges", conn, index=False)  # Export the table to SQLite
                        consolidated = pd.read_sql_query(
                            # Self-join on overlapping ranges and pick the furthest-apart start/end:
                            f"""
                                select distinct
                                    { ", ".join(map(lambda dim: f"X.{dim}", dimension_fields)) },
                                    X.starts as starts_x,
                                    min(X.starts, Y.starts) as starts,
                                    min(X.prestarts, Y.prestarts) as prestarts,
                                    max(X.ends, Y.ends) as ends,
                                    max(X.postends, Y.postends) as postends
                                from
                                    total_ranges X join total_ranges Y on
                                    { " and ".join(map(lambda dim: f"X.{dim} = Y.{dim}", dimension_fields))}
                                    and X.ends >= Y.prestarts
                                    and X.starts <= Y.postends
                                    and (X.starts >= Y.starts or X.ends <= Y.ends)
                            """,
                            conn,
                        )
                        # Restore the pandas field types from SQLite import:
                        for field in ["starts_x", "starts", "prestarts", "ends", "postends"]:
                            consolidated[field] = pd.to_datetime(consolidated[field])
                        for field in dimension_fields:
                            consolidated[field] = consolidated[field].astype(total_ranges[field].dtype)

                        # Clear out the SQLLite table for next loop
                        with contextlib.closing(conn.cursor()) as cursor: # auto-closes
                            cursor.execute("drop table total_ranges")#

                        # The above join isn't sufficient by itself, because the self-join path allows
                        # redundant ranges through. So we also summarize by left start date and drop
                        # duplicates, to clear those out:
                        # TODO: Move entirely into SQL for more efficient execution planning?
                        consolidated = consolidated.groupby(dimension_fields + ["starts_x"]).agg({
                            "starts": "min",
                            "prestarts": "min",
                            "ends": "max",
                            "postends": "max",
                        }).reset_index()
                        total_ranges = consolidated[
                            dimension_fields + ["starts", "prestarts", "ends", "postends"]
                        ].drop_duplicates()
                    # endwhile prev_n_ranges > len(total_ranges):
                # endwith sqllite connection
            # endif frequency is not None:
        # endfor tts_chunk in tts_chunker
    # endfor tts_filename in tts_filenames

    # Processing loop finished - report generation starts here:

    # Some useful pre-report setup:
    if frequency is not None:
        # global timestamps are strings only because they can be tracked even when frequency
        # is not provided, so we'll use the same method as for later sections to calculate steps:
        # TODO: Off by one, the .n + 1 is not a sufficient fix
        globallims = pd.to_datetime(
            pd.Series([global_tts_start, global_tts_end])
        ).dt.to_period(frequency)
        global_steps = math.ceil(((globallims[1] - globallims[0]).n + 1) / freq_spec["dt_periods"])
    else:
        global_steps = None

    # Initial summary section:
    display(Markdown("\n".join((
        "### Target Time-Series Summary",
        "- **Time span:** {} to {}{}".format(
            global_tts_start,
            global_tts_end,
            f" ({global_steps} timesteps)" if global_steps is not None else "",
        ),
        f"- **Total records:** {total_records} of which {total_records_nonulls} with no missing values",
    ))))
    if total_records != total_records_nonulls:
        n_missing = total_records - total_records_nonulls
        display(notebook.generate_warnbox(
            f"{n_missing} ({100*n_missing/total_records:.2f}% of total) records contain missing values"
        ))
        display(pd.DataFrame({ "Missing/Empty Values": num_nulls_by_field }))

    # Top-level analysis of items in the forecast (item_id and whatever other dimensions):
    display(Markdown("\n".join((
        "### Items",
        f"- **Unique items to forecast:** {len(unique_dimension_combos)}",
    ))))
    if len(unique_dimension_combos) < WARN_THRESH_MIN_ITEMS:
        display(notebook.generate_warnbox(
            f"Small number of items ({len(unique_dimension_combos)})",
            context_html=(
                "The more advanced, deep learning-based algorithms in Amazon Forecast often excel in",
                "scenarios where a large number of timeseries are to be jointly modelled.",
            ),
        ))
    display(Markdown(f"**Top items by record count:**"))
    unique_dimension_combos_sorted = unique_dimension_combos.sort_values(ascending=False)
    display(add_pct_to_value_counts(unique_dimension_combos_sorted, clip=10))
    item_records_loglog = plot_loglog(
        unique_dimension_combos_sorted,
        quantity="record count",
        instance_units=f"items",
    )
    del unique_dimension_combos_sorted  # Free up some memory
    if item_records_loglog is not None:
        slope = item_records_loglog[0]
        if slope < WARN_THRESH_LOGLOG_HEAD_HEAVY:
            display(generate_warnbox(
                "Head-heavy distribution of item data",
                context_html=(
                    "Available data seems particularly concentrated in a small proportion of items.",
                    "This may mean forecast accuracy is weaker for the 'long tail' of items with sparser",
                    "data.",
                ),
            ))
    # TODO: Only output the values per dim if multiple dimensions
    for fname in unique_dimension_vals:
        display(Markdown(f"**Unique values in dimension '{fname}'**: {len(unique_dimension_vals[fname])}"))
        display(Markdown(f"**Top record counts by dimension {fname}:**"))
        display(add_pct_to_value_counts(unique_dimension_vals[fname], clip=10))
        plot_loglog(
            unique_dimension_vals[fname].sort_values(ascending=False),
            quantity="record count",
            instance_units=f"{fname}s",
        )

    # Time-wise analyses:
    display(Markdown("### Data Ranges"))
    if frequency is None:
        display(notebook.generate_warnbox(
            f"Detailed timestamp analysis skipped",
            context_html=((
                "Provide the 'frequency' argument to enable analysis of the time ranges covered for each",
                "item in the forecast.",
            )),
            level="danger",
        ))
    else:  # frequency is not None
        freq_spec = fcst.FREQUENCIES[frequency]
        total_ranges["range"] = (
            # TODO: Could using Periods help us earlier, too?
            # TODO: Off by one (handled by e.n + 1 below), same should be 1 period not zero
            (total_ranges["ends"].dt.to_period(frequency))
            - total_ranges["starts"].dt.to_period(frequency)
        )
        n_ranges_total = len(total_ranges)
        display(Markdown("\n".join((
            f"- **Forecast frequency:** '{frequency}'",
            "- **Total detected contiguous data ranges:** {} (avg {:.2f} per item)".format(
                n_ranges_total,
                n_ranges_total / len(unique_dimension_combos)
            ),
        ))))
        if max_chunk_aggregated_records > 1:
            if n_chunks_global > 1:
                detail = f"Detected at least {max_chunk_aggregated_records} in one interval (lower bound)."
            else:
                detail = f"Detected max {max_chunk_aggregated_records} in one time interval."
            display(notebook.generate_warnbox(
                f"Target time-series contains aggregations at the specified granularity '{frequency}'",
                context_html=((
                    "<p>",
                    "Your data contains instances where multiple records will map to a single ",
                    "timestamp-item bucket. This is supported, but you should check it's expected (rather",
                    "than duplicate records) and configure appropriate",
                    '<a href="https://docs.aws.amazon.com/forecast/latest/dg/howitworks-datasets-groups.html#howitworks-data-alignment">',
                    "aggregation settings</a> when creating your Predictor.",
                    "</p>",
                    f"<p>{detail}</p>",
                )),
            ))

        display(Markdown("#### Overall extents (including gaps between records)"))
        extents = total_ranges.groupby(dimension_fields).agg({
            "starts": "min",
            "ends": "max",
            "postends": "max",
        })
        # pd.Period - pd.Period gives pd.tseries.Offset which has property `.n` describing number of steps
        extents["range"] = (
            # TODO: Could using Periods help us earlier, too?
            extents["postends"].dt.to_period(frequency)
            - extents["starts"].dt.to_period(frequency)
        )
        # TODO: Remove 'range' column from output when debugging is done
        extents["steps"] = extents["range"].apply(lambda e: e.n / freq_spec["dt_periods"])
        extent_cumulative_counts = [
            (extents["steps"] / global_steps >= bp[0]).sum() for bp in EXTENT_BREAKPOINTS
        ]
        avg_extent_steps = extents["steps"].mean()
        display(Markdown(
            "On average, each item spans {} time steps ({} min, {} max)".format(
                avg_extent_steps,
                extents["steps"].min(),
                extents["steps"].max(),
            )
        ))
        for spec in WARN_AVG_EXTENT_THRESHS:
            if avg_extent_steps < spec.thresh:
                display(notebook.generate_warnbox(
                    f"Average item extent is less than {spec.thresh} time steps",
                    context_html = spec.details,
                    level=spec.level,
                ))
                break
        display(Markdown("\n".join((
            "Ignoring/filling gaps, items span proportion of the total history as follows:",
            f"- **≥{100 * EXTENT_BREAKPOINTS[0][0]}% of global history:** {extent_cumulative_counts[0]}",
            "\n".join(map(
                lambda o: "- **{}%> x ≥{}% of global history:** {} ({}% of all items)".format(
                    100 * EXTENT_BREAKPOINTS[o[0]][0],
                    100 * EXTENT_BREAKPOINTS[o[0] + 1][0],
                    o[1],
                    100 * o[1] / len(unique_dimension_combos),
                ),
                enumerate(np.diff(extent_cumulative_counts).tolist()),
            )),
            "- **<{}% of global history:** {} ({}% of all items)".format(
                100 * EXTENT_BREAKPOINTS[-1][0],
                len(unique_dimension_combos) - max(extent_cumulative_counts),
                (len(unique_dimension_combos) - max(extent_cumulative_counts))/len(unique_dimension_combos),
            ),
        ))))
        for ix, count in enumerate(extent_cumulative_counts):
            warn_thresh = EXTENT_BREAKPOINTS[ix][1]
            if warn_thresh is not None and (count/len(unique_dimension_combos)) < warn_thresh:
                display(notebook.generate_warnbox(
                    "<{}% of all items cover at least {}% of the global history".format(
                        100 * warn_thresh,
                        100 * EXTENT_BREAKPOINTS[ix][0],
                    ),
                    context_html=EXTENT_BREAKPOINTS[ix][2]
                ))
        extents = extents.sort_values(["steps"], ascending=False).drop("postends", axis=1)
        display(Markdown("**Top items by total extent:**"))
        display(extents.head(10))
        plot_loglog(
            extents["steps"],
            ref_yvals=STEPCOUNT_REFERENCES,
            quantity="extent (timesteps)",
            instance_units="items",
        )
        if len(extents) > 10:
            display(Markdown("**Bottom items by total extent:**"))
            display(extents.tail(10))

        display(Markdown("#### Contiguous ranges (split by gaps)"))
        total_ranges["range"] = (
            # TODO: Could using Periods help us earlier, too?
            (total_ranges["postends"].dt.to_period(frequency))
            - total_ranges["starts"].dt.to_period(frequency)
        )
        # TODO: Remove 'range' column from output when debugging is done
        total_ranges["steps"] = total_ranges["range"].apply(lambda e: e.n / freq_spec["dt_periods"])
        sizes = total_ranges.groupby(dimension_fields).agg({
            "steps": ["count", "min", "mean", "max", "sum"]
        }).sort_values([("steps", "sum")], ascending=False)
        # Replace the multilevel index with column names that make more sense:
        sizes.columns = [
            "# Contiguous Ranges",
            "Min Range Length",
            "Mean Range Length",
            "Max Range Length",
            "Timesteps Covered"
        ]
        # Join on the total extent (non-contiguous) covered by each item:
        sizes = sizes.join(extents[["steps"]]).rename(columns={ "steps": "Extent Timesteps" })
        sizes_cover_pct = sizes["Timesteps Covered"] / sizes["Extent Timesteps"]
        span_cumulative_counts = [
            (sizes_cover_pct >= bp[0]).sum() for bp in SPAN_BREAKPOINTS
        ]
        display(Markdown("\n".join((
            "Items contiguously cover proportion of their overall extents as follows:",
            f"- **≥{100 * SPAN_BREAKPOINTS[0][0]}% of extent:** {span_cumulative_counts[0]}",
            "\n".join(map(
                lambda o: "- **{}%> x ≥{}% of extent:** {} ({}% of all items)".format(
                    100 * SPAN_BREAKPOINTS[o[0]][0],
                    100 * SPAN_BREAKPOINTS[o[0] + 1][0],
                    o[1],
                    100 * o[1] / len(unique_dimension_combos),
                ),
                enumerate(np.diff(span_cumulative_counts).tolist()),
            )),
            "- **<{}% of global history:** {} ({}% of all items)".format(
                100 * SPAN_BREAKPOINTS[-1][0],
                len(unique_dimension_combos) - max(span_cumulative_counts),
                (len(unique_dimension_combos) - max(span_cumulative_counts))/len(unique_dimension_combos),
            ),
        ))))
        for ix, count in enumerate(span_cumulative_counts):
            warn_thresh = SPAN_BREAKPOINTS[ix][1]
            if warn_thresh is not None and (count/len(unique_dimension_combos)) < warn_thresh:
                display(notebook.generate_warnbox(
                    "<{}% of all items cover at least {}% of their total extent".format(
                        100 * warn_thresh,
                        100 * SPAN_BREAKPOINTS[ix][0],
                    ),
                    context_html=SPAN_BREAKPOINTS[ix][2]
                ))
        display(Markdown("**Top items by timesteps covered:**"))
        display(sizes.head(10))
        # TODO: Maybe more useful to plot Coverage % here?
        plot_loglog(
            sizes["Timesteps Covered"],
            ref_yvals=STEPCOUNT_REFERENCES,
            quantity="timesteps covered",
            instance_units="items",
        )
        if len(sizes) > 10:
            display(Markdown("**Most sparse items:**"))
            sizes["Coverage %"] = sizes_cover_pct
            sizes = sizes.sort_values(["Coverage %"])
            sizes["Coverage %"] = pd.Series(
                ["{0:.2f}%".format(val * 100) for val in sizes["Coverage %"]],
                index=sizes["Coverage %"].index,
            )
            display(sizes.head(10))
