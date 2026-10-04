# Clickstream Analytics Agent

Ask questions about clickstream data in plain English. A Gemini model writes a read-only DuckDB SQL query, the app runs it against a local database, and Streamlit shows the answer with a chart, the SQL, and the raw data.

## Features

- **Chat over your data:** each answer goes through three visible steps (generate SQL → run query → summarize), with timings for each.
- **Answer / SQL / Data tabs:** a short answer with the key number in bold, notes on any assumptions the model made, and a chart picked automatically (bar or line). The Data tab formats percentages, dates and numbers.
- **Follow-up suggestions:** three suggested next questions after each answer. Click one to ask it.
- **Overview dashboard:** KPI cards with sparklines (sessions, daily active users, purchase conversion, average session length), a conversion funnel and sessions per day. It shows on the start screen and from the sidebar's **Overview** button.
- **Multiple chats:** each conversation is listed in the sidebar and titled automatically after its first answer.
- **Feedback and export:** thumbs up/down, copy the SQL, download results as CSV.
- **Light and dark themes**, set in `.streamlit/config.toml`.

## Data

`clickstream.duckdb` contains two tables:

| Table | Grain | Rows | Highlights |
|---|---|---|---|
| `fact_sessions` | one row per session | 71,849 | user, start/end time, events, pages viewed, converted, duration |
| `fact_daily_metrics` | one row per day | 92 | daily active users, sessions, funnel counts (page view → click → add to cart → purchase), conversion rate |

The data covers Jul 1 – Sep 30, 2026 (92 consecutive days).

## Getting started

Requires Python 3.10+ (developed on 3.14) and a [Gemini API key](https://aistudio.google.com/apikey).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then put your key in GEMINI_API_KEY
streamlit run app.py
```

The app opens at http://localhost:8501.

## Configuration

| Setting | Where | Default |
|---|---|---|
| `GEMINI_API_KEY` | `.env` or `.streamlit/secrets.toml` | required |
| `GEMINI_MODEL` | `.env` | `gemini-3.5-flash-lite` |
| Theme (colors, fonts, light/dark) | `.streamlit/config.toml` | indigo accent, Inter font |

Neither `.env` nor `.streamlit/secrets.toml` is committed (both are in `.gitignore`).

## Safety and limits

- The database opens **read-only**, and only queries starting with `SELECT` or `WITH` are run. Anything else is blocked and shown as an error.
- Results are capped at 200 rows.
- Gemini calls time out after 90 seconds and show a "Query failed" message instead of hanging.
- Query results are cached for an hour.

## Project structure

```
app.py                  Streamlit app: LLM pipeline, chat UI, overview dashboard
clickstream.duckdb      Sample clickstream database
.streamlit/config.toml  Theme
requirements.txt        Python dependencies
.env.example            Template for environment variables
```

## Built with

[Streamlit](https://streamlit.io) · [DuckDB](https://duckdb.org) · [Google Gen AI SDK](https://github.com/googleapis/python-genai) · [Altair](https://altair-viz.github.io) · [pandas](https://pandas.pydata.org)
