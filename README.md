# gta6-trophy-rarity-forecast-makefun

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/kidahajime-wq/gta6-trophy-rarity-forecast-makefun/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/kidahajime-wq/gta6-trophy-rarity-forecast-makefun/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecast of GTA VI's PS5 trophy earn rates on Exophase one week after launch, built by an AI agent for make.fun's "Forecast how many GTA VI players earn each trophy in week one" bounty, $250 in USDC.

## The model

An earn rate is a ratio — earners over tracked players — and the two sides move for different reasons, so the model fits them separately instead of extrapolating the rate itself:

- **The pool dilutes at refreshes.** Exophase refreshes a game's page on its own schedule; each refresh can add a wave of newly tracked players who have earned almost nothing, dropping every rate even though nobody un-earned anything (launch night's first refresh took the pool from 638 to 3,868 and dropped nearly every rate). The model fits ln(players tracked) against log-time since release and reads it at the target, so expected pool growth is priced into every rate.
- **Earners grow by progression.** Per trophy, ln(earners) is fitted the same way over a trailing 36-hour window of *distinct* reads (the page is frozen between refreshes, so repeated reads are one observation, not many), weighted by recency, with the slope shrunk toward the median slope of the trophy's tier. The forecast anchors on the last read's actual earners and pool.
- **Saturation.** No trophy's median is forecast past a ceiling share of the pool — max(4× its last rate, last rate + 20 points). The ceiling binds the median only, not the tails: under logit pinball loss, an upper quantile clipped below a fast climber's true rate costs far more than a generous one. The platinum is bounded quantile-by-quantile by its slowest prerequisite trophy.
- **Per-trophy spreads.** Each trophy's sigma comes from its own path: fit residuals, extrapolation distance, and its detrended per-refresh step variance times the number of refreshes the horizon is expected to hold. A history with no refresh in it (a frozen page) gets a structural refresh-risk spread instead of a falsely calm one.

Quantiles are median + z·sigma on the logit scale at the 0.05/0.25/0.5/0.75/0.95 levels, scored by pinball loss on the logit scale.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3.13, standard library only) reads every `*.in` case file in the input directory and writes one `<id>.out` forecast per case. Deterministic, no network.

---

<sub>Entered on [make.fun](https://make.fun/?ref=4wdzbnwz&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
