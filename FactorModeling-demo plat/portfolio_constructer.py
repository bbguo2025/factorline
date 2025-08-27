import numpy as np
import pandas as pd
import time


def simulation_settings(
    pct=0.1,
    min_universe=1000,
    equal_weight=True,
    max_weight=0.05,
    transaction_cost=False,
    cost_per_turnover=0.0003,
    rebalance_period="daily",
):
    """
    Settings for the long-short simulation.

    Parameters:
    - pct: float
        Top/bottom percentile for signal ranking to select stocks.
    - min_universe: int
        Minimum number of symbols required to form the portfolio on a given day.

    - transaction_cost: bool
        Whether to subtract transaction costs from returns.

    - equal_weight: bool
        Whether to assign equal weights within long and short legs.

    - cost_per_turnover: float
        Transaction cost per 100% turnover (e.g., 0.0003 = 3 basis points).

    - rebalance_period: str
        Rebalancing frequency. Options: "daily", "monthly".
        "daily": Rebalance every trading day (default behavior).
        "monthly": Rebalance only on the first trading day of each month.
    """
    return locals()


def get_rebalance_dates(dates, rebalance_period="daily"):
    """
    Determine which dates are rebalancing dates based on the rebalance period.

    Parameters:
    - dates: list or pd.Index of dates
    - rebalance_period: str, "daily" or "monthly"

    Returns:
    - set of dates that are rebalancing dates
    """
    # Ensure dates are properly converted to datetime
    dates = pd.to_datetime(dates)

    if rebalance_period == "daily":
        return set(dates)
    elif rebalance_period == "monthly":
        # Convert to DataFrame to group by year-month
        df = pd.DataFrame({'date': dates})
        df['year_month'] = df['date'].dt.to_period('M')
        # Get first date of each month
        monthly_first_dates = df.groupby('year_month')['date'].min()
        return set(monthly_first_dates.values)
    else:
        raise ValueError(
            f"Unsupported rebalance_period: {rebalance_period}. Use 'daily' or 'monthly'."
        )


def daily_trade_list(equal_weight,
                     custom_feature,
                     pct,
                     min_universe,
                     max_weight,
                     rebalance_period="daily"):
    try:
        if equal_weight:
            return equal_weight_daily_trade_list(custom_feature, pct,
                                                 min_universe,
                                                 rebalance_period)
        else:
            return flexible_weight_daily_trade_list(custom_feature,
                                                    min_universe, max_weight,
                                                    rebalance_period)
    except Exception as e:
        print(f"Error in daily_trade_list: {e}")
        # Return empty weights and counts as fallback
        empty_weights = pd.Series(dtype=float)
        empty_counts = pd.DataFrame(columns=['long_count', 'short_count'])
        return empty_weights, empty_counts


def cap_and_redistribute(weights: pd.Series,
                         max_weight: float,
                         max_iter=10,
                         tol=1e-6):
    """
    Iteratively cap weights at max_weight and redistribute leftover to uncapped weights
    so longs sum to 1 and shorts sum to -1.

    weights: pd.Series with sum longs=1 and shorts=-1
    max_weight: max absolute weight per stock

    Returns capped and normalized weights.
    """
    # Handle edge case of empty weights
    if weights.empty:
        return weights

    weights = weights.copy()
    for _ in range(max_iter):
        # Clip weights
        capped = weights.clip(lower=-max_weight, upper=max_weight)

        # Calculate leftover on each side
        long_leftover = 1 - capped[capped > 0].sum()
        short_leftover = -1 - capped[capped < 0].sum()

        # Identify uncapped longs and shorts
        uncapped_longs = capped[(weights > 0) & (capped < max_weight)]
        uncapped_shorts = capped[(weights < 0) & (capped > -max_weight)]

        # If no uncapped weights to redistribute leftover, break
        if (abs(long_leftover) < tol and abs(short_leftover) < tol) or \
           (len(uncapped_longs) == 0 and len(uncapped_shorts) == 0):
            break

        # Redistribute leftover proportionally on uncapped longs
        if len(uncapped_longs) > 0 and abs(long_leftover) > tol:
            long_weights = uncapped_longs / uncapped_longs.sum()
            capped.loc[long_weights.index] += long_leftover * long_weights

        # Redistribute leftover proportionally on uncapped shorts
        if len(uncapped_shorts) > 0 and abs(short_leftover) > tol:
            short_weights = uncapped_shorts / uncapped_shorts.sum()
            capped.loc[short_weights.index] += short_leftover * short_weights

        weights = capped

    # Final clipping to be safe
    return weights.clip(lower=-max_weight, upper=max_weight)


def flexible_weight_daily_trade_list(custom_feature: pd.Series,
                                     min_universe: int = 1000,
                                     max_weight: float = 0.05,
                                     rebalance_period: str = "daily"):
    """
    OPTIMIZED VERSION: Much faster monthly rebalancing with vectorized operations
    """
    if rebalance_period == "monthly":
        return _flexible_weight_monthly_ultra_optimized(
            custom_feature, min_universe, max_weight)
    else:
        return _flexible_weight_daily_original(custom_feature, min_universe,
                                               max_weight)


def _flexible_weight_monthly_ultra_optimized(custom_feature: pd.Series,
                                             min_universe: int = 1000,
                                             max_weight: float = 0.05):
    """
    ULTRA OPTIMIZED monthly rebalancing - processes entire date ranges at once
    """
    start_time = time.time()

    # Get all dates and determine rebalancing dates
    all_dates = custom_feature.index.get_level_values(
        'date').unique().sort_values()
    rebalance_dates = get_rebalance_dates(all_dates, "monthly")

    total_dates = len(all_dates)
    rebalance_count = len(rebalance_dates)

    print(
        f"[ULTRA OPTIMIZED] Total dates: {total_dates}, Rebalancing dates: {rebalance_count}"
    )

    # Pre-calculate weights only on rebalancing dates
    rebalance_weights = {}
    rebalance_counts = {}

    for i, date in enumerate(rebalance_dates):
        if i % 10 == 0:
            print(
                f"[ULTRA OPTIMIZED] Processing rebalancing date {i+1}/{rebalance_count}: {date}"
            )

        group = custom_feature.loc[date]
        x = group.dropna()
        # Remove duplicate symbols (keep first) to ensure unique index
        if x.index.duplicated().any():
            x = x[~x.index.duplicated(keep='first')]

        if len(x) < min_universe:
            continue

        # Calculate weights for this rebalancing date
        weights = pd.Series(0.0, index=x.index)

        # Calculate positive weights
        pos_mask = x > 0
        if pos_mask.sum() > 0:
            weights[pos_mask] = x[pos_mask] / x[pos_mask].sum()

        # Calculate negative weights
        neg_mask = x < 0
        if neg_mask.sum() > 0:
            weights[neg_mask] = x[neg_mask] / -x[neg_mask].sum()

        # Apply weight caps
        if (weights.abs() > max_weight).any():
            print(
                f"[{date}] {sum(weights.abs() > max_weight)} weights exceed max_weight ({max_weight})"
            )

        long_count = (weights > 0).sum()
        short_count = (weights < 0).sum()
        weights = cap_and_redistribute(weights, max_weight)

        # Store symbol-level weights (handle both MultiIndex and single-level Index)
        symbol_index = x.index.get_level_values('symbol') if isinstance(
            x.index, pd.MultiIndex) else x.index
        rebalance_weights[date] = pd.Series(weights.values,
                                            index=symbol_index,
                                            name='weight')
        rebalance_counts[date] = {
            "long_count": long_count,
            "short_count": short_count
        }

    print("[ULTRA OPTIMIZED] Creating complete weights DataFrame...")

    # Create a mapping from each date to its rebalancing date
    date_to_rebalance = {}
    sorted_rebalance_dates = sorted(rebalance_weights.keys())

    # Ensure all dates are properly converted to datetime for consistent comparison
    all_dates = pd.to_datetime(all_dates)
    # Convert rebalance dates to the same type as the keys in rebalance_weights
    sorted_rebalance_dates = [
        pd.to_datetime(d) for d in sorted_rebalance_dates
    ]

    for i, rebalance_date in enumerate(sorted_rebalance_dates):
        if i < len(sorted_rebalance_dates) - 1:
            end_date = sorted_rebalance_dates[i + 1]
            date_range = all_dates[(all_dates >= rebalance_date)
                                   & (all_dates < end_date)]
        else:
            date_range = all_dates[all_dates >= rebalance_date]

        for date in date_range:
            date_to_rebalance[date] = rebalance_date

    # Process all dates at once using vectorized operations
    all_weights_list = []
    all_counts_list = []

    # Group by rebalancing date for vectorized processing
    for rebalance_date in sorted_rebalance_dates:
        # Use the original date type from rebalance_weights keys, not converted
        # Find the matching key in rebalance_weights by comparing dates
        matching_key = None
        for key in rebalance_weights.keys():
            if pd.to_datetime(key) == rebalance_date:
                matching_key = key
                break

        if matching_key is None:
            continue

        current_weights = rebalance_weights[matching_key]
        current_counts = rebalance_counts[matching_key]

        # Get all dates that use this rebalancing date
        period_dates = [
            d for d, rd in date_to_rebalance.items() if rd == rebalance_date
        ]

        # Process all dates in this period at once
        for date in period_dates:
            try:
                group = custom_feature.loc[date]
                x = group.dropna()
                # Remove duplicate symbols (keep first) to ensure unique index
                if x.index.duplicated().any():
                    x = x[~x.index.duplicated(keep='first')]

                if len(x) < min_universe:
                    # Create zero weights for this date - use simple index like daily
                    weights = pd.Series(0.0, index=x.index)
                else:
                    # Robust weight lookup (supports duplicate symbols)
                    current_symbols = x.index.get_level_values(
                        'symbol') if isinstance(x.index,
                                                pd.MultiIndex) else x.index
                    weight_map = current_weights.to_dict()
                    weights = pd.Series(
                        [weight_map.get(sym, 0.0) for sym in current_symbols],
                        index=x.index,
                    )

                all_weights_list.append(weights)
                all_counts_list.append({"date": date, **current_counts})

            except Exception as e:
                print(f"[WARNING] Error processing date {date}: {e}")
                try:
                    group = custom_feature.loc[date]
                    weights = pd.Series(0.0, index=group.index)
                    all_weights_list.append(weights)
                    all_counts_list.append({"date": date, **current_counts})
                except:
                    continue

    # Handle empty lists case
    if not all_weights_list or not all_counts_list:
        print("[WARNING] No weights generated. Creating empty result.")
        # Create empty result with proper structure
        empty_weights = pd.Series(dtype=float)
        empty_counts = pd.DataFrame(columns=['long_count', 'short_count'])
        return empty_weights, empty_counts

    # Concatenate efficiently - use same pattern as daily functions
    # Ensure unique keys by adding a counter for duplicate dates
    unique_keys = []
    date_counter = {}
    for d in all_counts_list:
        date = d['date']
        if date in date_counter:
            date_counter[date] += 1
            unique_keys.append(f"{date}_{date_counter[date]}")
        else:
            date_counter[date] = 0
            unique_keys.append(str(date))

    all_weights = pd.concat(all_weights_list,
                            keys=unique_keys).rename_axis(['date', 'symbol'])

    # Fix: Convert string dates to proper datetime in the index
    # Extract the date level and convert to datetime
    date_level = all_weights.index.get_level_values('date')
    if date_level.dtype == 'object':  # If dates are strings
        # Convert string dates to datetime
        converted_dates = pd.to_datetime(date_level)
        # Create new MultiIndex with proper datetime dates
        new_index = pd.MultiIndex.from_arrays(
            [converted_dates,
             all_weights.index.get_level_values('symbol')],
            names=['date', 'symbol'])
        all_weights.index = new_index

    shifted_weights = all_weights.groupby("symbol").shift(1)
    count_df = pd.DataFrame(all_counts_list).set_index("date")

    execution_time = time.time() - start_time
    print(
        f"[ULTRA OPTIMIZED] Completed in {execution_time:.2f} seconds with {len(shifted_weights)} weight entries"
    )
    print(
        f"[ULTRA OPTIMIZED] Processing speed: {total_dates/execution_time:.1f} dates/second"
    )

    return shifted_weights, count_df


def _flexible_weight_daily_original(custom_feature: pd.Series,
                                    min_universe: int = 1000,
                                    max_weight: float = 0.05):
    """
    Original daily rebalancing implementation (unchanged for daily mode)
    """
    weights_list = []
    count_list = []
    last_symbol_weights = None
    last_counts = {"long_count": 0, "short_count": 0}

    for date, group in custom_feature.groupby(level='date'):
        x = group.droplevel('date').dropna()
        # Remove duplicate symbols (keep first) to ensure unique index
        if x.index.duplicated().any():
            x = x[~x.index.duplicated(keep='first')]

        if len(x) < min_universe:
            if last_symbol_weights is not None:
                weights = pd.Series(0.0, index=x.index)
                current_symbols = x.index.get_level_values('symbol')
                for symbol in current_symbols:
                    if symbol in last_symbol_weights.index:
                        symbol_indices = x.index[current_symbols == symbol]
                        if len(symbol_indices) > 0:
                            weights[symbol_indices[0]] = last_symbol_weights[
                                symbol]
                weights_list.append(weights)
                count_list.append({"date": date, **last_counts})
            continue

        # Recalculate weights every day
        weights = pd.Series(0.0, index=x.index)

        pos_mask = x > 0
        if pos_mask.sum() > 0:
            weights[pos_mask] = x[pos_mask] / x[pos_mask].sum()

        neg_mask = x < 0
        if neg_mask.sum() > 0:
            weights[neg_mask] = x[neg_mask] / -x[neg_mask].sum()

        if (weights.abs() > max_weight).any():
            print(
                f"[{date}] {sum(weights.abs() > max_weight)} weights exceed max_weight ({max_weight})"
            )

        long_count = (weights > 0).sum()
        short_count = (weights < 0).sum()
        weights = cap_and_redistribute(weights, max_weight)

        last_symbol_weights = pd.Series(
            weights.values,
            index=x.index.get_level_values('symbol'),
            name='weight')
        last_counts = {"long_count": long_count, "short_count": short_count}

        weights_list.append(weights)
        count_list.append({"date": date, **last_counts})

    # Ensure unique keys by adding a counter for duplicate dates
    unique_keys = []
    date_counter = {}
    for d in count_list:
        date = d['date']
        if date in date_counter:
            date_counter[date] += 1
            unique_keys.append(f"{date}_{date_counter[date]}")
        else:
            date_counter[date] = 0
            unique_keys.append(str(date))

    all_weights = pd.concat(weights_list,
                            keys=unique_keys).rename_axis(['date', 'symbol'])
    shifted_weights = all_weights.groupby("symbol").shift(1)
    count_df = pd.DataFrame(count_list).set_index("date")

    return shifted_weights, count_df


def equal_weight_daily_trade_list(custom_feature: pd.Series,
                                  pct=0.1,
                                  min_universe=1000,
                                  rebalance_period: str = "daily"):
    """
    OPTIMIZED VERSION: Much faster monthly rebalancing with vectorized operations
    """
    if rebalance_period == "monthly":
        return _equal_weight_monthly_ultra_optimized(custom_feature, pct,
                                                     min_universe)
    else:
        return _equal_weight_daily_original(custom_feature, pct, min_universe)


def _equal_weight_monthly_ultra_optimized(custom_feature: pd.Series,
                                          pct=0.1,
                                          min_universe=1000):
    """
    ULTRA OPTIMIZED monthly rebalancing for equal weight strategy
    """
    start_time = time.time()

    # Get all dates and determine rebalancing dates
    all_dates = custom_feature.index.get_level_values(
        'date').unique().sort_values()
    rebalance_dates = get_rebalance_dates(all_dates, "monthly")

    total_dates = len(all_dates)
    rebalance_count = len(rebalance_dates)

    print(
        f"[ULTRA OPTIMIZED] Total dates: {total_dates}, Rebalancing dates: {rebalance_count}"
    )

    # Pre-calculate weights only on rebalancing dates
    rebalance_weights = {}
    rebalance_counts = {}

    for i, date in enumerate(rebalance_dates):
        if i % 10 == 0:
            print(
                f"[ULTRA OPTIMIZED] Processing rebalancing date {i+1}/{rebalance_count}: {date}"
            )

        group = custom_feature.loc[date]
        x = group.dropna()
        # Remove duplicate symbols (keep first) to ensure unique index
        if x.index.duplicated().any():
            x = x[~x.index.duplicated(keep='first')]

        if len(x) < min_universe:
            continue

        # Calculate equal weights for this rebalancing date
        n = len(x)
        k = max(int(np.floor(n * pct)), 1)
        ranked = x.rank(method="first", ascending=False)

        long = (ranked <= k).astype(float)
        long_weight = long / long.sum() if long.sum() > 0 else 0
        long_count = int(long.sum())

        short = (ranked > n - k).astype(float)
        short_weight = short / short.sum() if short.sum() > 0 else 0
        short_count = int(short.sum())

        weights = long_weight - short_weight
        weights.index = x.index

        # Store symbol-level weights (handle both MultiIndex and single-level Index)
        symbol_index = x.index.get_level_values('symbol') if isinstance(
            x.index, pd.MultiIndex) else x.index
        rebalance_weights[date] = pd.Series(weights.values,
                                            index=symbol_index,
                                            name='weight')
        rebalance_counts[date] = {
            "long_count": long_count,
            "short_count": short_count
        }

    print("[ULTRA OPTIMIZED] Creating complete weights DataFrame...")

    # Create a mapping from each date to its rebalancing date
    date_to_rebalance = {}
    sorted_rebalance_dates = sorted(rebalance_weights.keys())

    # Ensure all dates are properly converted to datetime for consistent comparison
    all_dates = pd.to_datetime(all_dates)
    # Convert rebalance dates to the same type as the keys in rebalance_weights
    sorted_rebalance_dates = [
        pd.to_datetime(d) for d in sorted_rebalance_dates
    ]

    for i, rebalance_date in enumerate(sorted_rebalance_dates):
        if i < len(sorted_rebalance_dates) - 1:
            end_date = sorted_rebalance_dates[i + 1]
            date_range = all_dates[(all_dates >= rebalance_date)
                                   & (all_dates < end_date)]
        else:
            date_range = all_dates[all_dates >= rebalance_date]

        for date in date_range:
            date_to_rebalance[date] = rebalance_date

    # Process all dates at once using vectorized operations
    all_weights_list = []
    all_counts_list = []

    # Group by rebalancing date for vectorized processing
    for rebalance_date in sorted_rebalance_dates:
        # Use the original date type from rebalance_weights keys, not converted
        # Find the matching key in rebalance_weights by comparing dates
        matching_key = None
        for key in rebalance_weights.keys():
            if pd.to_datetime(key) == rebalance_date:
                matching_key = key
                break

        if matching_key is None:
            continue

        current_weights = rebalance_weights[matching_key]
        current_counts = rebalance_counts[matching_key]

        # Get all dates that use this rebalancing date
        period_dates = [
            d for d, rd in date_to_rebalance.items() if rd == rebalance_date
        ]

        # Process all dates in this period at once
        for date in period_dates:
            try:
                group = custom_feature.loc[date]
                x = group.dropna()
                # Remove duplicate symbols (keep first) to ensure unique index
                if x.index.duplicated().any():
                    x = x[~x.index.duplicated(keep='first')]

                if len(x) < min_universe:
                    # Create zero weights for this date - use simple index like daily
                    weights = pd.Series(0.0, index=x.index)
                else:
                    # === Robust weight lookup (supports duplicate symbols) ===
                    current_symbols = x.index.get_level_values(
                        'symbol') if isinstance(x.index,
                                                pd.MultiIndex) else x.index
                    weight_map = current_weights.to_dict()
                    weights = pd.Series(
                        [weight_map.get(sym, 0.0) for sym in current_symbols],
                        index=x.index,
                    )

                all_weights_list.append(weights)
                all_counts_list.append({"date": date, **current_counts})

            except Exception as e:
                print(f"[WARNING] Error processing date {date}: {e}")
                try:
                    group = custom_feature.loc[date]
                    weights = pd.Series(0.0, index=group.index)
                    all_weights_list.append(weights)
                    all_counts_list.append({"date": date, **current_counts})
                except:
                    continue

    # Handle empty lists case
    if not all_weights_list or not all_counts_list:
        print("[WARNING] No weights generated. Creating empty result.")
        # Create empty result with proper structure
        empty_weights = pd.Series(dtype=float)
        empty_counts = pd.DataFrame(columns=['long_count', 'short_count'])
        return empty_weights, empty_counts

    # Concatenate efficiently - use same pattern as daily functions
    # Ensure unique keys by adding a counter for duplicate dates
    unique_keys = []
    date_counter = {}
    for d in all_counts_list:
        date = d['date']
        if date in date_counter:
            date_counter[date] += 1
            unique_keys.append(f"{date}_{date_counter[date]}")
        else:
            date_counter[date] = 0
            unique_keys.append(str(date))

    all_weights = pd.concat(all_weights_list,
                            keys=unique_keys).rename_axis(['date', 'symbol'])

    # Fix: Convert string dates to proper datetime in the index
    # Extract the date level and convert to datetime
    date_level = all_weights.index.get_level_values('date')
    if date_level.dtype == 'object':  # If dates are strings
        # Convert string dates to datetime
        converted_dates = pd.to_datetime(date_level)
        # Create new MultiIndex with proper datetime dates
        new_index = pd.MultiIndex.from_arrays(
            [converted_dates,
             all_weights.index.get_level_values('symbol')],
            names=['date', 'symbol'])
        all_weights.index = new_index

    shifted_weights = all_weights.groupby("symbol").shift(1)
    count_df = pd.DataFrame(all_counts_list).set_index("date")

    execution_time = time.time() - start_time
    print(
        f"[ULTRA OPTIMIZED] Completed in {execution_time:.2f} seconds with {len(shifted_weights)} weight entries"
    )
    print(
        f"[ULTRA OPTIMIZED] Processing speed: {total_dates/execution_time:.1f} dates/second"
    )

    return shifted_weights, count_df


def _equal_weight_daily_original(custom_feature: pd.Series,
                                 pct=0.1,
                                 min_universe=1000):
    """
    Original daily rebalancing implementation (unchanged for daily mode)
    """
    weights_list = []
    count_list = []
    last_symbol_weights = None
    last_counts = {"long_count": 0, "short_count": 0}

    for date, group in custom_feature.groupby(level="date"):
        x = group.dropna()
        # Remove duplicate symbols (keep first) to ensure unique index
        if x.index.duplicated().any():
            x = x[~x.index.duplicated(keep='first')]

        if len(x) < min_universe:
            if last_symbol_weights is not None:
                weights = pd.Series(0.0, index=x.index)
                current_symbols = x.index.get_level_values('symbol')
                for symbol in current_symbols:
                    if symbol in last_symbol_weights.index:
                        symbol_indices = x.index[current_symbols == symbol]
                        if len(symbol_indices) > 0:
                            weights[symbol_indices[0]] = last_symbol_weights[
                                symbol]
                weights_list.append(weights)
                count_list.append({"date": date, **last_counts})
            continue

        # Recalculate weights every day
        n = len(x)
        k = max(int(np.floor(n * pct)), 1)
        ranked = x.rank(method="first", ascending=False)

        long = (ranked <= k).astype(float)
        long_weight = long / long.sum() if long.sum() > 0 else 0
        long_count = int(long.sum())

        short = (ranked > n - k).astype(float)
        short_weight = short / short.sum() if short.sum() > 0 else 0
        short_count = int(short.sum())

        weights = long_weight - short_weight
        weights.index = x.index

        last_symbol_weights = pd.Series(
            weights.values,
            index=x.index.get_level_values('symbol'),
            name='weight')
        last_counts = {"long_count": long_count, "short_count": short_count}

        weights_list.append(weights)
        count_list.append({"date": date, **last_counts})

    all_weights = pd.concat(weights_list).sort_index()
    shifted_weights = all_weights.groupby("symbol").shift(1)
    count_df = pd.DataFrame(count_list).set_index("date")

    return shifted_weights, count_df


def daily_portfolio_returns(weights: pd.Series,
                            returns: pd.Series,
                            settings: dict,
                            contributor: bool = False):
    """
    Compute long-short portfolio daily returns and turnover, including transaction cost adjustment.

    Parameters:
    - weights: pd.Series with MultiIndex (date, symbol)
    - returns: pd.Series with MultiIndex (date, symbol)
    - settings: dict, output from simulation_settings()

    Returns:
    - result: pd.DataFrame with daily return, turnover, and cost-adjusted return
    - top_longs: pd.Series of top 10 long-leg contributors (total PnL)
    - top_shorts: pd.Series of top 10 short-leg contributors (total PnL)
    """
    # Handle empty weights case
    if weights.empty:
        print("[WARNING] Empty weights passed to daily_portfolio_returns")
        empty_result = pd.DataFrame(columns=[
            'log_return', 'long_return', 'short_return', 'long_turnover',
            'short_turnover', 'turnover'
        ])
        return empty_result, None, None

    # Ensure both weights and returns have unique (date, symbol) index
    if weights.index.duplicated().any():
        weights = weights[~weights.index.duplicated(keep="first")]
    if returns.index.duplicated().any():
        returns = returns[~returns.index.duplicated(keep="first")]

    aligned = pd.concat([weights.rename("weight"),
                         returns.rename("ret")],
                        axis=1).dropna()
    aligned["pnl"] = aligned["weight"] * aligned["ret"]
    daily_return = aligned.groupby(level="date")["pnl"].sum()

    weights_df = weights.unstack().fillna(0)
    returns_df = returns.unstack().fillna(0)

    longs = weights_df.clip(lower=0)
    shorts = weights_df.clip(upper=0).abs()

    long_return = (longs * returns_df).sum(axis=1)
    short_return = -(shorts * returns_df).sum(axis=1)

    long_turnover = longs.diff().abs().sum(axis=1)
    short_turnover = shorts.diff().abs().sum(axis=1)
    total_turnover = long_turnover + short_turnover

    # === Adjust for transaction cost ===
    if settings["transaction_cost"]:
        cost = total_turnover * settings["cost_per_turnover"]
    else:
        cost = 0

    net_return = daily_return - cost

    result = pd.concat([
        net_return.rename("log_return"),
        long_return.rename("long_return"),
        short_return.rename("short_return"),
        long_turnover.rename("long_turnover"),
        short_turnover.rename("short_turnover"),
        total_turnover.rename("turnover")
    ],
                       axis=1).reset_index()

    # Ensure date column is properly converted to datetime before sorting
    result['date'] = pd.to_datetime(result['date'])
    result = result.sort_values("date", ascending=False).reset_index(drop=True)

    if contributor:
        # ==== Top contributors over the full period ====
        longs_total_pnl = (longs * returns_df).sum()
        shorts_total_pnl = -(shorts * returns_df).sum()

        top_longs = longs_total_pnl.sort_values(ascending=False).head(10)
        top_shorts = shorts_total_pnl.sort_values(ascending=False).head(10)

        return result, top_longs, top_shorts

    else:
        return result, None, None


def metrics_calculator(custom_feature: pd.Series, returns: pd.Series,
                       weights: pd.Series,
                       counts: pd.DataFrame) -> pd.DataFrame:
    # Compute IC and IR
    df = pd.concat([custom_feature.rename("alpha"),
                    returns.rename("ret")],
                   axis=1).dropna()
    df_grouped = df.groupby(level="date")

    ic = df_grouped.apply(lambda x: x["alpha"].corr(x["ret"]))
    ic_mean = ic.mean()
    ic_std = ic.std()
    ir = ic_mean / ic_std if ic_std != 0 else np.nan

    # Calculate turnover stats from weights
    weights_df = weights.unstack().fillna(0)
    longs = weights_df.clip(lower=0)
    shorts = weights_df.clip(upper=0).abs()

    long_turnover = longs.diff().abs().sum(axis=1)
    short_turnover = shorts.diff().abs().sum(axis=1)
    total_turnover = long_turnover + short_turnover

    avg_turnover = total_turnover.mean()
    std_turnover = total_turnover.std()

    # Counts mean and std
    avg_long_count = counts['long_count'].mean()
    std_long_count = counts['long_count'].std()
    avg_short_count = counts['short_count'].mean()
    std_short_count = counts['short_count'].std()

    return pd.DataFrame({
        "IC": [ic_mean],
        "IR": [ir],
        "IC Std": [ic_std],
        "Turnover Mean": [avg_turnover],
        "Turnover Std": [std_turnover],
        "Long Count Mean": [avg_long_count],
        "Long Count Std": [std_long_count],
        "Short Count Mean": [avg_short_count],
        "Short Count Std": [std_short_count],
    })
