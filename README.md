# Nifty 200 Live Momentum Dashboard

A Streamlit dashboard that evaluates a Nifty 200 universe and generates target portfolio actions based on an index-based momentum strategy.

## What this app does

- Reads a Nifty 200 constituent CSV with `Symbol` and `Sector`
- Downloads adjusted daily price history from Yahoo Finance
- Applies a market filter, stock filters, and momentum ranking
- Builds an equal-weight target portfolio of up to 20 stocks
- Enforces a maximum 10% allocation per stock and 30% per sector
- Compares the target portfolio with current holdings
- Produces BUY, BUY MORE, HOLD, SELL PARTIAL, and SELL ALL actions
- Shows signals for the latest completed trading day and provisional latest close

## Files in this repo

- `streamlit_app.py` — main Streamlit dashboard implementation
- `ind_nifty200listWithSector.csv` — required universe file with Nifty 200 symbols and sectors
- `current_holdings_template.csv` — optional holdings template used by the dashboard
- `requirements.txt` — Python dependencies for running the app
- `README.md` — this file

## Setup

Install dependencies into a Python environment using the repository requirements.

```bash
python3 -m pip install -r requirements.txt
```

## Run locally

```bash
streamlit run streamlit_app.py
```

Then open the local Streamlit URL that appears in the terminal.

## Data files

- `ind_nifty200listWithSector.csv` is required and should contain at least:
  - `Symbol`
  - `Sector`

- `current_holdings_template.csv` is optional and can provide current holdings with:
  - `Symbol`
  - `Quantity`
  - `AveragePrice`

The dashboard also includes a file uploader for holdings if you prefer not to use the template file.

## Notes

- The official signal is based on the last completed trading day of the month.
- The app also supports a provisional live watchlist mode for the most recent daily close, but that mode is not the official monthly rebalance signal.
- The app uses Yahoo Finance to fetch historical prices, so internet access is required.
