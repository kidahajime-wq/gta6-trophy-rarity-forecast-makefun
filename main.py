"""gta6-trophy-rarity-forecast-01: an earners-and-pool forecast of GTA VI trophy rates.

Why not just extrapolate the rate? Because an Exophase earn rate is a ratio,

    rate = earners / players_tracked,

and after a launch the two sides move for different reasons. The denominator grows
in steps: Exophase refreshes a game's page on its own schedule, and each refresh can
add a wave of newly tracked players who have earned almost nothing, so every rate
dilutes at a refresh even though nobody un-earned anything (launch night's first
refresh took the pool from 638 to 3,868 and dropped nearly every rate). The
numerator grows by progression: the players already tracked keep finishing story
chapters, hunts and grinds. A single trend fitted to the ratio conflates the two;
this program fits them separately and lets the ratio fall out:

  - Pool: ln(players_tracked) is fitted against log-time since release and read at
    the target, shrunk toward no growth. Pool growth decelerates (the transform is
    ln(1 + hours)), so the dilution waves get smaller as the week goes on.
  - Earners: per trophy, ln(earners + 0.5) is fitted the same way, its slope shrunk
    toward the median slope of its tier (platinum/gold/silver/bronze progress at
    different speeds, and a noisy trophy borrows its tier's pace). The forecast is
    anchored on the last read's actual earners and pool, not on the fits'
    intercepts, so a level shift in the last refresh is carried, not smoothed away.
  - Saturation: earners cannot pass a per-trophy ceiling share of the pool. The
    ceiling is max(4 x last rate, last rate + 20 points), capped at 99.95% and
    never below the last rate itself: a long grind
    that sits at 1% a week in is not forecast to 40%. The ceiling binds the
    median only, never the tails: under logit pinball loss an upper quantile
    clipped below a fast climber's true rate costs far more than a generous one
    costs when the trophy stays rare. The platinum is then bounded by its
    slowest prerequisite: no quantile of the platinum may exceed the same
    quantile of the hardest other trophy, since nobody holds the platinum
    without holding it.

Reads are deduplicated to refresh events first: the page is frozen between
refreshes, so a stretch of identical reads is one observation repeated, and
counting it many times would fake confidence and hide the refresh steps that are
the real risk. All fits use distinct reads only, weighted by recency.

The spread is per trophy, priced from that trophy's own path:

  sigma^2 = fit residual variance
          + prediction variance at the target's distance from the fit
          + (detrended per-refresh step variance) x (refreshes expected in the
            horizon, from the case's own median gap between refreshes)
          + a random-walk term over the horizon.

A case whose history holds a single distinct read (the practice cases: one frozen
launch-night page) shows no refresh at all, yet a refresh before the target is
nearly certain and launch night's moved typical rates by about a logit. Such a
trophy therefore gets a structural refresh-risk spread instead of a false calm
one: sigma from REFRESH_PRIOR_SD, one expected step.

Quantiles are median_logit + z * sigma through the logistic at the case's levels,
clamped to the ceiling and to 0..100, non-decreasing. Rates are clamped to
0.01..99.99 before any logit, as the scoring clamps them. Deterministic, no
network, standard library only; run as `python3 /bot/main.py` with the case in
/input, or with MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR set locally.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_AT_LEVEL = [-1.645, -0.674, 0.0, 0.674, 1.645]

# The clock every trend is measured on: release, 2026-11-19T00:00:00Z.
EPOCH = datetime(2026, 11, 19, tzinfo=timezone.utc)

# Recency weighting: a distinct read this many hours older than the newest counts
# half. Short on purpose: growth near the anchor is what extrapolates, and the
# launch-night ramp (earners near zero, the pool exploding) must not set the pace
# for next week — a whole-history slope overshoots every saturating trophy.
WEIGHT_HALF_LIFE_HOURS = 12.0

# Trend extrapolation is systematically optimistic over a multi-day horizon
# (every growth curve here saturates), so the median's move away from the last
# observed rate is damped by this factor.
EXTRAP_DAMP = 0.75

# Slopes are fitted on a trailing window only. A whole-history slope measures the
# average growth since launch — dominated by the launch ramp, when earners were
# near zero — and anchoring that average at the last read overshoots every
# saturating trophy. The local slope near the anchor is what extrapolates.
WINDOW_HOURS = 36.0
WINDOW_MIN_READS = 4

# Slope shrinkage priors (log-space slope per unit of ln(1 + hours)): the pool's
# growth is shrunk toward 0, a trophy's earners growth toward its tier's median.
POOL_PRIOR_SD = 0.60
EARNER_PRIOR_SD = 0.35

# Saturation ceiling for a trophy's share of the pool, from its last rate. The cap
# never sits below the last rate itself: a trophy already at 100% saturates at 100%.
CEILING_CAP = 99.95
CEILING_MULTIPLE = 4.0
CEILING_HEADROOM_POINTS = 20.0

# Spread: floors, the random-walk term, the refresh-risk prior for histories that
# show no refresh, and the overall bounds.
RESID_VAR_FLOOR = 0.02
WALK_SIGMA_PER_SQRT_DAY = 0.25
REFRESH_PRIOR_SD = 0.80
MAX_REFRESH_STEPS = 12.0
SIGMA_MIN = 0.35
SIGMA_MAX = 3.50

# The median may not run away from the last observed rate, however strong a trend fits.
MAX_MEDIAN_MOVE_LOGIT = 2.75


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_epoch(moment: datetime) -> float:
    return (moment - EPOCH).total_seconds() / 3600.0


def log_time(hours: float) -> float:
    """ln(1 + hours); pre-release reads (hours slightly negative) stay finite."""
    return math.log(1.0 + max(hours, -0.9))


def clamp_rate(pct: float) -> float:
    return min(99.99, max(0.01, pct))


def logit(pct: float) -> float:
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def logistic(value: float) -> float:
    if value >= 0:
        return 100.0 / (1.0 + math.exp(-value))
    e = math.exp(value)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """Round to 4 decimals, half up; whole numbers return as ints for compact JSON."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def median_of(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def distinct_reads(history: list[dict]) -> list[dict]:
    """The history with frozen stretches collapsed: a read that repeats the
    previous kept read's players_tracked and rates exactly is the same page
    seen again, not a new observation."""
    kept: list[dict] = []
    for read in history:
        if kept:
            prev = kept[-1]
            if prev.get("players_tracked") == read.get("players_tracked") and prev.get("rates") == read.get("rates"):
                continue
        kept.append(read)
    return kept


def weighted_line(points: list[tuple[float, float, float]]) -> dict:
    """Weighted least squares y ~= a + b*x over (x, y, w) points.

    Returns the intercept, slope, weighted residual variance, and the pieces the
    prediction variance needs (total weight, x centre, Sxx). With no x-spread the
    slope is 0 and the level is the weighted mean.
    """
    total_w = sum(w for _, _, w in points)
    x_bar = sum(w * x for x, _, w in points) / total_w
    y_bar = sum(w * y for _, y, w in points) / total_w
    sxx = sum(w * (x - x_bar) ** 2 for x, _, w in points)
    if sxx <= 1e-12:
        resid = sum(w * (y - y_bar) ** 2 for _, y, w in points) / total_w
        return {"a": y_bar, "b": 0.0, "resid": resid, "w": total_w, "x_bar": x_bar, "sxx": sxx, "n": len(points)}
    sxy = sum(w * (x - x_bar) * (y - y_bar) for x, y, w in points)
    slope = sxy / sxx
    intercept = y_bar - slope * x_bar
    resid = sum(w * (y - (intercept + slope * x)) ** 2 for x, y, w in points) / total_w
    return {"a": intercept, "b": slope, "resid": resid, "w": total_w, "x_bar": x_bar, "sxx": sxx, "n": len(points)}


def shrink_slope(fit: dict, prior: float, prior_sd: float) -> float:
    """Precision-weight a fitted slope toward a prior: a slope the data pins down
    (large Sxx against the residual variance) survives; a noisy one borrows the prior.
    A fit on fewer than four points is distrusted outright — two points have zero
    residuals by construction, so their slope only looks certain. A many-point fit
    with genuinely tiny residuals (the pool's own count is exact and smooth) is
    believed: flooring its variance would shrink real pool growth toward zero and
    under-forecast the dilution this model exists to price."""
    if fit["sxx"] <= 1e-12:
        return prior
    slope_var = max(fit["resid"], 1e-4) / fit["sxx"]
    if fit["n"] < 4:
        slope_var = max(slope_var, 0.25)
    own = 1.0 / slope_var
    borrowed = 1.0 / (prior_sd * prior_sd)
    return (fit["b"] * own + prior * borrowed) / (own + borrowed)


def recency_weights(hours_list: list[float]) -> list[float]:
    newest = max(hours_list)
    return [0.5 ** ((newest - h) / WEIGHT_HALF_LIFE_HOURS) for h in hours_list]


def ceiling_for(last_rate: float) -> float:
    """The share of the pool a trophy can plausibly reach by the target."""
    ceiling = max(CEILING_MULTIPLE * last_rate, last_rate + CEILING_HEADROOM_POINTS)
    return min(CEILING_CAP, max(ceiling, last_rate))


def forecast_case(case: dict) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"case asks for levels {case['levels']}, not {LEVELS}")
    trophies = case["trophies"]
    history = distinct_reads(case["history"])
    horizon_days = case["horizon_minutes"] / 1440.0
    horizon_hours = case["horizon_minutes"] / 60.0
    target_x = log_time(hours_since_epoch(parse_time(case["target"])))

    if not history:
        # No reads at all: nothing to anchor on; a wide, low prior for every trophy.
        rows = []
        for trophy in trophies:
            qs = [round4(logistic(logit(0.5) + z * 1.5)) for z in Z_AT_LEVEL]
            rows.append({"id": trophy["id"], "q": qs})
        return {"trophies": rows}

    read_hours = [hours_since_epoch(parse_time(read["at"])) for read in history]
    read_x = [log_time(h) for h in read_hours]
    weights = recency_weights(read_hours)
    pools = []
    last_pool = 1.0
    for read in history:
        p = read.get("players_tracked") or 0
        if p <= 0:
            p = last_pool
        pools.append(float(p))
        last_pool = float(p)
    anchor_x = read_x[-1]
    dx = target_x - anchor_x

    # The trailing window the slopes are fitted on (see WINDOW_HOURS above); the
    # newest read is always in it, and too-short a window falls back to everything.
    window_from = read_hours[-1] - WINDOW_HOURS
    in_window = [h >= window_from for h in read_hours]
    if sum(in_window) < WINDOW_MIN_READS:
        in_window = [True] * len(read_hours)

    # Refresh cadence of this case: the median gap between distinct reads prices
    # how many more refreshes the horizon holds.
    gaps = [read_hours[i + 1] - read_hours[i] for i in range(len(read_hours) - 1)]
    gaps = [g for g in gaps if g > 0]
    median_gap = median_of(gaps) if gaps else 0.0
    if median_gap > 0:
        expected_steps = min(MAX_REFRESH_STEPS, max(0.5, horizon_hours / median_gap))
    else:
        expected_steps = 1.0
    saw_refresh = len(history) > 1

    # The pool's own trend, shrunk toward no growth.
    pool_fit = weighted_line(
        [(x, math.log(p), w) for x, p, w, keep in zip(read_x, pools, weights, in_window) if keep]
    )
    pool_slope = shrink_slope(pool_fit, 0.0, POOL_PRIOR_SD)
    pool_target = pools[-1] * math.exp(pool_slope * dx)

    # First pass per trophy: earners slope, and the logit path's own volatility.
    fits = []
    for index in range(len(trophies)):
        earn_points = []
        logit_points = []
        last_rate = None
        last_earners = None
        last_x = None
        for x, w, pool, read, keep in zip(read_x, weights, pools, history, in_window):
            if not keep:
                continue
            rate = read["rates"][index]
            if rate is None:
                continue
            earners = pool * rate / 100.0
            earn_points.append((x, math.log(earners + 0.5), w))
            logit_points.append((x, logit(rate), w))
            last_rate = rate
            last_earners = earners
            last_x = x
        earn_fit = weighted_line(earn_points) if earn_points else None
        logit_fit = weighted_line(logit_points) if logit_points else None
        # Detrended per-refresh steps of the logit path: what a refresh does to
        # this trophy beyond its trend.
        steps = []
        if logit_fit is not None:
            obs = [(x, y) for x, y, _ in logit_points]
            for (x0, y0), (x1, y1) in zip(obs, obs[1:]):
                steps.append((y1 - y0) - logit_fit["b"] * (x1 - x0))
        fits.append(
            {
                "earn_fit": earn_fit,
                "logit_fit": logit_fit,
                "steps": steps,
                "last_rate": last_rate,
                "last_earners": last_earners,
                "last_x": last_x,
            }
        )

    # Tier priors for earners growth: each trophy borrows its tier's median slope
    # computed WITHOUT itself (a one-trophy tier would otherwise borrow its own
    # noise and shrink nothing), falling back to the global median slope.
    tier_slopes: dict[str, list[float]] = {}
    for trophy, fit in zip(trophies, fits):
        if fit["earn_fit"] is not None:
            tier_slopes.setdefault(trophy["tier"], []).append(fit["earn_fit"]["b"])
    global_slope = median_of([s for slopes in tier_slopes.values() for s in slopes])
    priors = []
    for trophy, fit in zip(trophies, fits):
        own = fit["earn_fit"]["b"] if fit["earn_fit"] is not None else None
        peers = list(tier_slopes.get(trophy["tier"], []))
        if own is not None and own in peers:
            peers.remove(own)
        priors.append(median_of(peers) if peers else global_slope)
    all_steps = [s for fit in fits for s in fit["steps"]]
    if all_steps:
        mean_step = sum(all_steps) / len(all_steps)
        cross_step_var = sum((s - mean_step) ** 2 for s in all_steps) / len(all_steps)
    else:
        cross_step_var = REFRESH_PRIOR_SD**2

    rows = []
    for trophy, fit, prior in zip(trophies, fits, priors):
        if fit["last_rate"] is None or fit["earn_fit"] is None:
            median_logit, sigma = logit(0.5), 1.50
        else:
            earn_slope = shrink_slope(fit["earn_fit"], prior, EARNER_PRIOR_SD)
            # This trophy's own horizon runs from its last valid read, which may
            # be older than the case's last read when its newest rates are null.
            dx_own = target_x - (fit["last_x"] if fit["last_x"] is not None else anchor_x)
            # The +0.5 pseudocount is applied to the anchor as in the fit: a trophy
            # at zero earners can still grow, instead of being multiplied from 0.
            earners_target = (fit["last_earners"] + 0.5) * math.exp(earn_slope * dx_own) - 0.5
            earners_target = max(earners_target, 0.0)
            ceiling = ceiling_for(fit["last_rate"])
            rate_target = 100.0 * earners_target / pool_target
            rate_target = min(rate_target, ceiling)
            rate_target = max(rate_target, 0.005)
            median_logit = logit(rate_target)
            # No trend, however fitted, may drag the median far from the last read,
            # and what move remains is damped: growth saturates over the horizon.
            # Boundary trophies (at ~0% or ~100%) are exempt from the move cap:
            # there the earners/pool arithmetic is the trustworthy signal, and a
            # cap around logit(99.99) would forbid ordinary dilution outright.
            last_logit = logit(fit["last_rate"])
            median_logit = last_logit + EXTRAP_DAMP * (median_logit - last_logit)
            if 0.1 < fit["last_rate"] < 99.9:
                median_logit = min(last_logit + MAX_MEDIAN_MOVE_LOGIT, max(last_logit - MAX_MEDIAN_MOVE_LOGIT, median_logit))

            logit_fit = fit["logit_fit"]
            resid_var = max(logit_fit["resid"], RESID_VAR_FLOOR) if logit_fit else RESID_VAR_FLOOR
            if logit_fit and logit_fit["sxx"] > 1e-12:
                pred_var = resid_var * (1.0 / logit_fit["w"] + (target_x - logit_fit["x_bar"]) ** 2 / logit_fit["sxx"])
            else:
                pred_var = resid_var
            steps = fit["steps"]
            if steps:
                mean_s = sum(steps) / len(steps)
                own_var = sum((s - mean_s) ** 2 for s in steps) / len(steps)
                step_var = (len(steps) * own_var + 3.0 * cross_step_var) / (len(steps) + 3.0)
            elif saw_refresh:
                step_var = cross_step_var
            else:
                step_var = REFRESH_PRIOR_SD**2
            sigma = math.sqrt(
                resid_var + pred_var + step_var * expected_steps + WALK_SIGMA_PER_SQRT_DAY**2 * horizon_days
            )
            sigma = min(SIGMA_MAX, max(SIGMA_MIN, sigma))

        quantiles = []
        for z in Z_AT_LEVEL:
            q = logistic(median_logit + z * sigma)
            # The saturation ceiling binds the median (applied above), not the
            # tails: under logit pinball loss an upper quantile clipped below a
            # fast climber's true rate costs far more than a generous one costs
            # when the trophy stays rare. Only the natural bound applies here.
            quantiles.append(round4(min(CEILING_CAP, max(0.0, q))))
        for k in range(1, len(quantiles)):
            if quantiles[k] < quantiles[k - 1]:
                quantiles[k] = quantiles[k - 1]
        rows.append({"id": trophy["id"], "q": quantiles})

    # The platinum is bounded by its slowest prerequisite, quantile by quantile.
    plat_index = next((i for i, t in enumerate(trophies) if t["tier"] == "platinum"), None)
    if plat_index is not None and len(rows) > 1:
        plat = rows[plat_index]["q"]
        for k in range(len(Z_AT_LEVEL)):
            hardest = min(row["q"][k] for i, row in enumerate(rows) if i != plat_index)
            if plat[k] > hardest:
                plat[k] = hardest
        for k in range(1, len(plat)):
            if plat[k] < plat[k - 1]:
                plat[k] = plat[k - 1]

    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
