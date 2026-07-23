# Spencer Framework — a daily-frequency equity factor research system, built from scratch

A complete factor-research pipeline built exclusively on **public data and
public methodology**: data layer → factor zoo → style model & neutralization →
three-tier evaluation panels → cost-aware backtest → portfolio construction →
research discipline. The goal is not "yet another backtest library" but to
personally answer, in code, every engineering question a serious factor
research system must answer — see `docs/量化研究系统必答30问.md`
(36 questions and growing; Chinese).

## Provenance statement

Three sources only, clean from the first line:

1. **Public data**: baostock / akshare free APIs (incl. point-in-time stock
   lists with delisted names — survivorship-bias-free);
2. **Public methodology**: qlib's data-layer design, alphalens-style tear
   sheets, MSCI Barra's published style-factor definitions, Newey-West (1987),
   Deflated Sharpe & PBO (Bailey & López de Prado), FISTA (Beck & Teboulle);
3. **Original engineering**: every interface, name and implementation.

No proprietary code, schemas, or constants from any institution.

## Highlights

- **Point-in-time everywhere**: PIT universe with delisted stocks; PIT
  fundamentals (effective the day *after* disclosure); forward returns via
  `shift(-(1+h))/shift(-1)`; look-ahead pinned down by unit tests that
  hand-verify single cells and mutate "the future" to prove invariance.
- **Three-tier readings**: every factor evaluated raw / size-neutral /
  industry+6-style-neutral (both factor and label residualized). The gap
  between tiers *is* information: it measures style free-riding.
- **Discipline layer**: append-only experiment ledger; the trial count N
  feeds Deflated Sharpe and PBO (CSCV) directly. The system's best composite
  currently reads **DSR 0.47 — i.e. not significant after N=60 trials**, and
  it says so itself.
- **Falsification as a feature**: a suspiciously good ML factor (rank IC 0.048
  vs residual labels, 9 years all-positive) was dissected by three
  falsification experiments and closed as "goodness-of-fit on an untradable
  label" — full post-mortem in `output/falsify_model_gb.md`.
- **White-box optimizer**: FISTA + capped-simplex projection with style
  exposure bands; KKT residuals and a λ→∞ identity check run on every demo.

## Quickstart

```bash
pip install -r requirements.txt
python tests/test_core.py                 # offline sanity (all tests are offline)
python examples/quickstart.py --limit 80  # end-to-end smoke on 80 names
python examples/quickstart.py             # full CSI300+CSI500 universe
python examples/fetch_pit.py              # full-market PIT data (incl. delisted)
python examples/pit_final_run.py          # survivorship-bias-free full run
```

## Honest limitations (by design)

Match-level execution simulation and capacity modelling (beyond daily data),
historical industry membership (no public source), live trading (rent a
broker platform — this repo outputs target holdings, `output/target_holdings_*.csv`).

MIT License.
