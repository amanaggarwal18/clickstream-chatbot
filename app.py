import json
import os
import re
import time
import uuid

import altair as alt
import duckdb
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "clickstream.duckdb")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
MAX_ROWS = 200
LLM_TIMEOUT_MS = 90_000
MAX_BAR_ROWS = 15

# Chart accents per theme mode (match .streamlit/config.toml primary colors).
PALETTE = {
    "light": {"top": "#4F46E5", "bar": "#C7D2FE", "text": "#111827"},
    "dark": {"top": "#818CF8", "bar": "#3730A3", "text": "#E5E7EB"},
}

STARTERS = {
    ":material/trending_up: Traffic": [
        "How many sessions were there per day?",
        "Which day had the most active users?",
    ],
    ":material/shopping_cart: Conversion": [
        "What is the purchase conversion rate by day?",
        "Where do sessions drop off in the funnel?",
    ],
    ":material/group: Users": [
        "Top 5 users by number of sessions",
        "How long is an average session?",
    ],
}

st.set_page_config(
    page_title="Clickstream Analytics Agent",
    page_icon=":material/insights:",
    layout="wide",
)

# Native buttons center their label; left-align the sidebar chat list like a nav menu.
st.html(
    """
    <style>
    .st-key-chatlist button, .st-key-chatlist button > div {justify-content: flex-start; width: 100%;}
    .st-key-chatlist button p {white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
    </style>
    """
)


# ---------- Data + LLM ----------
@st.cache_resource
def get_con():
    return duckdb.connect(DB_PATH, read_only=True)


def get_api_key() -> str | None:
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.getenv("GEMINI_API_KEY")


@st.cache_resource
def get_client():
    key = get_api_key()
    return genai.Client(api_key=key, http_options=types.HttpOptions(timeout=LLM_TIMEOUT_MS)) if key else None


@st.cache_data
def get_tables() -> list[str]:
    return [t for (t,) in get_con().sql("SHOW TABLES").fetchall()]


@st.cache_data
def get_schema() -> str:
    con = get_con()
    parts = []
    for t in get_tables():
        cols = con.sql(f"DESCRIBE {t}").fetchall()
        col_txt = ", ".join(f"{c[0]} {c[1]}" for c in cols)
        sample = con.sql(f"SELECT * FROM {t} LIMIT 2").df().to_string(index=False)
        parts.append(f"TABLE {t} ({col_txt})\nSample rows:\n{sample}")
    return "\n\n".join(parts)


@st.cache_data
def describe(table: str) -> pd.DataFrame:
    return get_con().sql(f"DESCRIBE {table}").df()[["column_name", "column_type"]]


@st.cache_data(ttl="1h")
def table_counts() -> dict[str, int]:
    return {t: get_con().cursor().sql(f"SELECT count(*) FROM {t}").fetchone()[0] for t in get_tables()}


@st.cache_data(ttl="1h")
def date_range() -> tuple | None:
    if "fact_sessions" not in get_tables():
        return None
    return get_con().cursor().sql("SELECT min(dt), max(dt) FROM fact_sessions").fetchone()


@st.cache_data(ttl="1h")
def load_daily() -> pd.DataFrame | None:
    if "fact_daily_metrics" not in get_tables():
        return None
    return get_con().cursor().sql("SELECT * FROM fact_daily_metrics ORDER BY dt").df()


def ask_llm(prompt: str) -> str:
    return get_client().interactions.create(model=MODEL, input=prompt).output_text


def clean_sql(text: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    return (m.group(1) if m else text).strip().rstrip(";")


def is_safe(sql: str) -> bool:
    s = sql.strip().lower()
    return s.startswith(("select", "with")) and ";" not in s


def tables_in(sql: str) -> list[str]:
    return [t for t in get_tables() if re.search(rf"\b{t}\b", sql, re.I)]


def generate_sql(question: str, history: list[dict]) -> str:
    convo = "\n".join(
        f"{m['role']}: {m['content']}" for m in history[-6:] if m.get("content")
    )
    prompt = f"""You are a DuckDB SQL expert for a clickstream analytics database.
Write ONE read-only DuckDB SQL query that answers the user's question.
Use readable snake_case column aliases (e.g. conversion_rate, sessions).
Return only the SQL, no explanation.
If the question cannot be answered from the schema, return exactly: CANNOT_ANSWER

Schema:
{get_schema()}

Recent conversation:
{convo}

Question: {question}"""
    return clean_sql(ask_llm(prompt))


def summarize(question: str, sql: str, df: pd.DataFrame) -> dict:
    prompt = f"""You are a clickstream analytics assistant. Using the query result, return ONLY a JSON object:
{{
  "answer": "1-2 plain-language sentences answering the question. Bold the key number with **. Don't mention SQL.",
  "title": "3-5 word title for this conversation, sentence case",
  "chart": {{
    "type": "bar" | "line" | "none",
    "label_col": "result column for categories (bar) or dates (line)",
    "value_col": "numeric result column to plot",
    "format": "percent" | "number" | "currency" | "seconds",
    "caption": "short definition of the metric and date range, e.g. 'purchased sessions ÷ total sessions · Sep 21 – Oct 3'"
  }},
  "notes": ["0-2 short notes on definitions or assumptions the query made, e.g. 'Conversion = sessions with a purchase ÷ all sessions'"],
  "follow_ups": ["three short follow-up questions (max 6 words each) answerable from the schema"]
}}
Use "bar" for a handful of categories, "line" for a time series, "none" for a single value or wide tables.

Schema:
{get_schema()}

Question: {question}
SQL: {sql}
Result ({len(df)} rows, first 50 shown):
{df.head(50).to_string(index=False)}"""
    raw = ask_llm(prompt)
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        return json.loads(m.group(0)) if m else {"answer": raw}
    except json.JSONDecodeError:
        return {"answer": raw}


@st.cache_data(ttl="1h", max_entries=200, show_spinner=False)
def run_query(sql: str) -> pd.DataFrame:
    # A cursor per query: the cached connection is shared across sessions/threads.
    return get_con().cursor().sql(f"SELECT * FROM ({sql}) LIMIT {MAX_ROWS}").df()


# ---------- State ----------
def new_chat() -> str:
    cid = uuid.uuid4().hex[:8]
    st.session_state.chats[cid] = {"title": "New chat", "messages": []}
    st.session_state.active = cid
    return cid


if "chats" not in st.session_state:
    st.session_state.chats = {}
    new_chat()
st.session_state.setdefault("pending", None)

chat = st.session_state.chats[st.session_state.active]


def on_new_chat():
    if chat["messages"]:
        new_chat()


def set_active(cid: str):
    st.session_state.active = cid


def ask(question: str):
    st.session_state.pending = question


def on_pick(key: str):
    if q := st.session_state.get(key):
        ask(q)


def on_feedback(key: str):
    if st.session_state.get(key) is not None:
        st.toast("Thanks for the feedback!", icon=":material/favorite:")


# ---------- Formatting ----------
def palette() -> dict:
    return PALETTE["dark" if st.context.theme.type == "dark" else "light"]


def humanize(col: str) -> str:
    return col.replace("_", " ").strip().capitalize()


def guess_kind(col: str, values: pd.Series) -> str:
    name = col.lower()
    if re.search(r"rate|pct|percent|share|ratio|conversion", name) and values.dropna().between(0, 1).all():
        return "percent"
    if re.search(r"duration|seconds|secs", name):
        return "seconds"
    if re.search(r"revenue|amount|price|value_usd|spend", name):
        return "currency"
    return "number"


def fmt(v: float, kind: str, scale_pct: bool = True) -> str:
    if pd.isna(v):
        return "–"
    if kind == "percent":
        return f"{v * 100 if scale_pct else v:.1f}%"
    if kind == "currency":
        return f"${v:,.0f}" if abs(v) >= 100 else f"${v:,.2f}"
    if kind == "seconds":
        return f"{v / 60:,.1f} min" if v >= 120 else f"{v:,.0f}s"
    return f"{v:,.0f}" if float(v).is_integer() or abs(v) >= 100 else f"{v:,.2f}"


def column_config(df: pd.DataFrame) -> dict:
    """Readable headers plus percent/date/number formatting for the Data tab."""
    cfg = {}
    for c in df.columns:
        s, label = df[c], humanize(c)
        if pd.api.types.is_datetime64_any_dtype(s):
            is_date = (s.dropna().dt.normalize() == s.dropna()).all()
            cfg[c] = (
                st.column_config.DateColumn(label, format="MMM D, YYYY") if is_date
                else st.column_config.DatetimeColumn(label, format="MMM D, h:mm a")
            )
        elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            kind = guess_kind(c, s)
            if kind == "percent":
                cfg[c] = st.column_config.ProgressColumn(label, format="percent", min_value=0, max_value=1)
            elif kind == "currency":
                cfg[c] = st.column_config.NumberColumn(label, format="dollar")
            else:
                cfg[c] = st.column_config.NumberColumn(label, format="localized")
        else:
            cfg[c] = st.column_config.TextColumn(label)
    return cfg


# ---------- Charts ----------
def bar_chart(data: pd.DataFrame, kind: str) -> alt.LayerChart:
    """Horizontal bars with value labels; the top value is highlighted."""
    colors = palette()
    if "text" not in data:
        scale_pct = data["value"].abs().max() <= 1
        data = data.assign(text=[fmt(v, kind, scale_pct) for v in data["value"]])
    vmax = data["value"].max()
    data = data.assign(is_top=data["value"] == vmax)
    # Reserve room for category labels (web fonts load after Vega measures text)
    # and headroom on the right so value labels never clip.
    longest = data["label"].str.len().max()
    headroom = 1.15 + 0.035 * data["text"].str.len().max()
    base = alt.Chart(data).encode(
        y=alt.Y(
            "label:N", sort=None, title=None,
            axis=alt.Axis(labelFontSize=13, ticks=False, domain=False, labelPadding=8, labelLimit=220, minExtent=min(220, 9 * longest + 12)),
        ),
        x=alt.X("value:Q", axis=None, scale=alt.Scale(domain=[0, vmax * headroom if vmax > 0 else 1])),
    )
    bars = base.mark_bar(cornerRadius=5, height={"band": 0.62}).encode(
        color=alt.condition("datum.is_top", alt.value(colors["top"]), alt.value(colors["bar"])),
        tooltip=[alt.Tooltip("label:N", title="Label"), alt.Tooltip("text:N", title="Value")],
    )
    labels = base.mark_text(align="left", dx=8, fontSize=13, fontWeight=600, color=colors["text"]).encode(text="text:N")
    return (bars + labels).properties(height=max(110, 44 * len(data))).configure_view(stroke=None)


def trend_chart(data: pd.DataFrame, x: str, y: str, y_title: str) -> alt.LayerChart:
    """Daily bars on a time axis with a 7-day rolling average line on top."""
    colors = palette()
    data = data.assign(avg_7d=data[y].rolling(7, min_periods=1).mean().round(0))
    base = alt.Chart(data).encode(x=alt.X(f"{x}:T", title=None, axis=alt.Axis(format="%b %-d", tickCount=6, grid=False)))
    tooltip = [
        alt.Tooltip(f"{x}:T", title="Day", format="%a, %b %-d"),
        alt.Tooltip(f"{y}:Q", title=y_title, format=","),
        alt.Tooltip("avg_7d:Q", title="7-day avg", format=","),
    ]
    bars = base.mark_bar(color=colors["bar"]).encode(
        y=alt.Y(f"{y}:Q", title=y_title, axis=alt.Axis(grid=True, tickCount=4)), tooltip=tooltip
    )
    line = base.mark_line(color=colors["top"], strokeWidth=2.5, interpolate="monotone").encode(y="avg_7d:Q", tooltip=tooltip)
    return (bars + line).properties(height=230).configure_view(stroke=None)


def render_chart(spec: dict | None, df: pd.DataFrame | None, key: str):
    if not spec or df is None or df.empty:
        return
    kind, label, value = spec.get("type"), spec.get("label_col"), spec.get("value_col")
    if kind not in ("bar", "line") or label not in df or value not in df:
        return
    if not pd.api.types.is_numeric_dtype(df[value]):
        return
    caption = spec.get("caption")
    alt_text = caption or f"{humanize(value)} by {humanize(label)}"

    with st.container(border=True):
        if kind == "line":
            st.line_chart(
                df, x=label, y=value, height=260, alt=alt_text,
                x_label=humanize(label), y_label=humanize(value),
            )
        else:
            data = (
                df[[label, value]].dropna().head(MAX_BAR_ROWS)
                .rename(columns={label: "label", value: "value"})
                .astype({"label": str})
            )
            if data.empty:
                return
            st.altair_chart(bar_chart(data, spec.get("format", "number")), key=f"chart_{key}", alt=alt_text)
        if caption:
            st.caption(caption)


# ---------- Data overview ----------
def render_kpis(d: pd.DataFrame):
    last, prev = d.iloc[-1], d.iloc[:-1]
    sessions = int(d["total_sessions"].sum())
    conv = d["sessions_with_purchase"].sum() / sessions
    duration = (d["avg_session_duration_seconds"] * d["total_sessions"]).sum() / sessions
    last_day = last["dt"].strftime("%b %-d")

    with st.container(horizontal=True):
        st.metric(
            "Sessions", f"{sessions:,}",
            delta=f"{last['total_sessions'] / prev['total_sessions'].mean() - 1:+.0%}",
            delta_description=f"{last_day} vs. daily avg",
            chart_data=d["total_sessions"].tolist(), chart_type="bar",
            border=True, icon=":material/touch_app:",
            help="All sessions across the tracked days.",
        )
        st.metric(
            "Daily active users", f"{d['daily_active_users'].mean():,.0f}",
            delta=int(last["daily_active_users"] - prev["daily_active_users"].mean()),
            delta_description=f"{last_day} vs. avg",
            chart_data=d["daily_active_users"].tolist(),
            border=True, icon=":material/group:",
            help="Average distinct users per day.",
        )
        st.metric(
            "Purchase conversion", f"{conv:.1%}",
            delta=f"{(last['purchase_conversion_rate'] - conv) * 100:+.1f} pts",
            delta_description=f"{last_day} vs. overall",
            chart_data=d["purchase_conversion_rate"].tolist(), chart_type="area",
            border=True, icon=":material/shopping_cart_checkout:",
            help="Sessions with a purchase ÷ all sessions.",
        )
        st.metric(
            "Avg. session", fmt(duration, "seconds"),
            delta=fmt(last["avg_session_duration_seconds"], "seconds"),
            delta_description=f"on {last_day}", delta_color="off", delta_arrow="off",
            chart_data=d["avg_session_duration_seconds"].tolist(),
            border=True, icon=":material/timer:",
            help="Session-weighted average duration.",
        )


def render_overview(key: str):
    d = load_daily()
    if d is None or d.empty:
        st.caption("No daily metrics table found.")
        return
    render_kpis(d)

    funnel_col, daily_col = st.columns([1, 1])
    with funnel_col.container(border=True, height="stretch"):
        st.markdown("**:material/filter_alt: Conversion funnel**")
        sessions = d["total_sessions"].sum()
        stages = {
            "Page view": d["sessions_with_page_view"].sum(),
            "Click": d["sessions_with_click"].sum(),
            "Add to cart": d["sessions_with_add_to_cart"].sum(),
            "Purchase": d["sessions_with_purchase"].sum(),
        }
        funnel = pd.DataFrame(
            {
                "label": list(stages),
                "value": list(stages.values()),
                "text": [f"{v:,} · {v / sessions:.0%}" for v in stages.values()],
            }
        )
        st.altair_chart(bar_chart(funnel, "number"), key=f"funnel_{key}", alt="Sessions reaching each funnel stage")
        st.caption("Sessions reaching each stage · % of all sessions")

    with daily_col.container(border=True, height="stretch"):
        st.markdown("**:material/calendar_month: Sessions per day**")
        st.altair_chart(
            trend_chart(d, "dt", "total_sessions", "Sessions"),
            key=f"daily_{key}", alt="Daily sessions with a 7-day rolling average line",
        )
        first, last = d["dt"].iloc[0], d["dt"].iloc[-1]
        st.caption(f"{first:%b %-d} – {last:%b %-d, %Y} · {len(d)} days · line = 7-day average")


@st.dialog("Data overview", width="large", icon=":material/dashboard:")
def overview_dialog():
    render_overview("dialog")


@st.dialog("Schema", width="medium", icon=":material/schema:")
def schema_dialog():
    counts = table_counts()
    for t in get_tables():
        st.markdown(f"**`{t}`** :gray[· {counts[t]:,} rows]")
        st.dataframe(
            describe(t), hide_index=True, alt=f"Columns of {t}",
            column_config={"column_name": "Column", "column_type": "Type"},
        )


# ---------- Messages ----------
def secs(x: float | None) -> str:
    return f"{x:.1f} s" if x is not None else "–"


def status_label(msg: dict) -> str:
    tables = " ".join(f"`{t}`" for t in msg.get("tables", [])) or "data"
    n = len(msg["df"]) if msg.get("df") is not None else 0
    total = sum(x for x in (msg.get("sql_s"), msg.get("ms", 0) / 1000, msg.get("sum_s")) if x)
    return f"Queried {tables} · {n:,} row{'s' if n != 1 else ''} · {total:.1f} s"


def render_steps(msg: dict):
    """Replay the pipeline as a collapsed step timeline."""
    with st.status(status_label(msg), type="compact", state="complete"):
        with st.status(f"Generated SQL · {secs(msg.get('sql_s'))}", type="step", state="complete"):
            st.code(msg["sql"], language="sql")
        with st.status(f"Ran query · {msg.get('ms', 0):,} ms", type="step", state="complete"):
            st.caption(f"{len(msg['df']):,} rows from {', '.join(msg.get('tables', [])) or 'the database'}")
        with st.status(f"Summarized · {secs(msg.get('sum_s'))}", type="step", state="complete"):
            st.caption(f"Answered with `{MODEL}`")


def render_headline(df: pd.DataFrame):
    """Single-row results read best as metric cards."""
    if len(df) != 1:
        return
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][:4]
    if not nums:
        return
    with st.container(horizontal=True):
        for c in nums:
            st.metric(humanize(c), fmt(df[c].iloc[0], guess_kind(c, df[c])), border=True)


def render_assistant(msg: dict, key: str, is_last: bool):
    if msg.get("error"):
        with st.status(msg.get("status", "Something went wrong"), type="compact", state="error"):
            if msg.get("sql"):
                st.code(msg["sql"], language="sql")
            st.caption(msg["error"])
        st.markdown(msg["content"])
        return

    if not msg.get("sql"):
        st.markdown(msg["content"])
        return

    df = msg["df"]
    render_steps(msg)

    show_sql = st.session_state.get("show_sql", False)
    labels = [":material/lightbulb: Answer", ":material/code: SQL", f":material/table_rows: Data · {len(df):,}"]
    t_answer, t_sql, t_data = st.tabs(labels, default=labels[1] if show_sql else labels[0], key=f"tabs_{key}_{show_sql}")
    with t_answer:
        render_headline(df)
        st.markdown(msg["content"])
        if notes := msg.get("notes"):
            st.caption("  \n".join(f":material/info: {n}" for n in notes))
        render_chart(msg.get("chart"), df, key)
    with t_sql:
        st.code(msg["sql"], language="sql")
    with t_data:
        st.dataframe(df, hide_index=True, column_config=column_config(df), alt="Query result")

    with st.container(horizontal=True, vertical_alignment="center"):
        fb_key = f"fb_{key}"
        msg["feedback"] = st.feedback("thumbs", key=fb_key, on_change=on_feedback, args=(fb_key,))
        with st.popover("Copy SQL", icon=":material/content_copy:", key=f"copy_{key}"):
            st.caption("Use the copy icon in the top-right of the block.")
            st.code(msg["sql"], language="sql")
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode(),
            file_name="clickstream_result.csv",
            mime="text/csv",
            icon=":material/download:",
            key=f"csv_{key}",
            on_click="ignore",
        )

    if is_last and msg.get("follow_ups"):
        pick_key = f"followups_{key}"
        st.pills(
            "Follow up", msg["follow_ups"][:3], key=pick_key, on_change=on_pick, args=(pick_key,),
            format_func=lambda q: f":material/subdirectory_arrow_right: {q}",
        )


def answer(question: str, history: list[dict]) -> dict:
    """Run the NL → SQL → result → summary pipeline as a live step timeline."""
    reply = {"role": "assistant", "content": ""}
    sql = None
    with st.status(":shimmer[Thinking]", type="compact", expanded=True) as status:
        step = st.status(":shimmer[Generating SQL]", type="step")
        try:
            with step:
                t0 = time.perf_counter()
                sql = generate_sql(question, history)
                sql_s = time.perf_counter() - t0
                if sql.strip() == "CANNOT_ANSWER":
                    step.update(label="No matching data", state="complete")
                    status.update(label="No matching data", state="complete", expanded=False)
                    reply["content"] = "I can't answer that from the available clickstream data."
                    return reply
                if not is_safe(sql):
                    raise ValueError("Generated SQL was not a read-only query, so it wasn't run.")
                st.code(sql, language="sql")
                step.update(label=f"Generated SQL · {secs(sql_s)}", state="complete")

            step = st.status(":shimmer[Running query]", type="step")
            with step:
                t0 = time.perf_counter()
                df = run_query(sql)
                ms = round((time.perf_counter() - t0) * 1000)
                st.caption(f"{len(df):,} rows")
                step.update(label=f"Ran query · {ms:,} ms", state="complete")

            step = st.status(":shimmer[Summarizing]", type="step")
            with step:
                t0 = time.perf_counter()
                summary = summarize(question, sql, df)
                sum_s = time.perf_counter() - t0
                step.update(label=f"Summarized · {secs(sum_s)}", state="complete")

            reply.update(
                sql=sql, df=df, ms=ms, sql_s=sql_s, sum_s=sum_s, tables=tables_in(sql),
                content=summary.get("answer", ""),
                chart=summary.get("chart"),
                notes=[n for n in summary.get("notes", []) if isinstance(n, str)][:2],
                follow_ups=[q for q in summary.get("follow_ups", []) if isinstance(q, str)],
                title=summary.get("title"),
            )
            status.update(label=status_label(reply), state="complete", expanded=False)
        except Exception as e:
            step.update(state="error")
            status.update(label="Query failed", state="error")
            reply.update(
                content="Something went wrong while answering that.",
                error=f"`{e}`", status="Query failed", sql=sql,
            )
    return reply


# ---------- Sidebar ----------
with st.sidebar:
    with st.container(horizontal=True, vertical_alignment="center"):
        st.header("Chats", width="stretch")
        st.button("New", icon=":material/add:", on_click=on_new_chat)

    with st.container(key="chatlist", gap=None):
        for cid, c in reversed(st.session_state.chats.items()):
            active = cid == st.session_state.active
            st.button(
                c["title"],
                key=f"chat_{cid}",
                icon=":material/chat_bubble:" if active else ":material/chat_bubble_outline:",
                type="secondary" if active else "tertiary",
                width="stretch",
                on_click=set_active,
                args=(cid,),
            )

    st.space("medium")
    st.subheader("Data source")
    counts = table_counts()
    with st.container(horizontal=True, gap="small"):
        for t in get_tables():
            st.badge(f"{t} · {counts[t]:,}", icon=":material/table:", color="gray")
    if rng := date_range():
        st.caption(f":material/date_range: {rng[0]:%b %-d} – {rng[1]:%b %-d, %Y}")
    with st.container(horizontal=True, gap="small"):
        if st.button("Overview", icon=":material/dashboard:", width="stretch"):
            overview_dialog()
        if st.button("Schema", icon=":material/schema:", width="stretch"):
            schema_dialog()

    st.space("large")
    st.toggle("Show SQL by default", key="show_sql")
    st.caption(f":material/smart_toy: `{MODEL}` · :material/lock: read-only DuckDB")


# ---------- Main ----------
st.title("Clickstream Analytics Agent")

if get_client() is None:
    st.error("`GEMINI_API_KEY` not set. Add it to `.env` or `.streamlit/secrets.toml`.", icon=":material/key:")
    st.stop()

cid = st.session_state.active
messages = chat["messages"]
for i, m in enumerate(messages):
    avatar = ":material/person:" if m["role"] == "user" else ":material/insights:"
    with st.chat_message(m["role"], avatar=avatar):
        if m["role"] == "user":
            st.markdown(m["content"])
        else:
            render_assistant(m, key=f"{cid}_{i}", is_last=i == len(messages) - 1)

typed = st.chat_input(
    "Ask a follow-up…" if messages else "Ask about sessions, conversions, users…",
    submit_mode="disable",
)
question = typed or st.session_state.pending
st.session_state.pending = None

if not messages and not question:
    st.caption(
        "Ask questions about your clickstream data in plain English — "
        "I write the SQL, run it, and explain the result. Here's where things stand:"
    )
    render_overview("home")

    st.space("small")
    st.subheader("Try asking")
    for col, (topic, qs) in zip(st.columns(len(STARTERS)), STARTERS.items()):
        with col.container(border=True, height="stretch"):
            st.markdown(f"**{topic}**")
            for j, q in enumerate(qs):
                st.button(
                    q, key=f"starter_{cid}_{topic}_{j}", type="tertiary",
                    icon=":material/arrow_outward:", on_click=ask, args=(q,),
                )

if question:
    history = list(messages)
    messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(question)
    with st.chat_message("assistant", avatar=":material/insights:"):
        reply = answer(question, history)
    if chat["title"] == "New chat":
        chat["title"] = (reply.pop("title", None) or question)[:40]
    messages.append(reply)
    st.rerun()
