"""Additional commodity tracking tab for Solar EPC Cost Intelligence.

FRED series are public price indexes, not vendor prices. Optional uploads are
explicitly tagged as user-provided data; no fabricated price history is used.
"""
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

SERIES = {
    "Steel — iron and steel": ("WPU101", "PPI index", "BLS/FRED"),
    "Silicon — silicon metal": (None, "USD / metric ton", "Upload benchmark CSV"),
    "Silicon — polysilicon": (None, "USD / kg", "Upload benchmark CSV"),
    "Aluminum — mill shapes": ("WPU102501", "PPI index", "BLS/FRED"),
    "Plastics — resins": ("WPU066", "PPI index", "BLS/FRED"),
    "Fuel — diesel": ("WPU057303", "PPI index", "BLS/FRED"),
    "Glass — flat glass": ("WPU1311", "PPI index", "BLS/FRED"),
    "Battery — LFP pack": (None, "USD / kWh", "Upload benchmark CSV"),
    "Battery — NMC pack": (None, "USD / kWh", "Upload benchmark CSV"),
}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred(series_id):
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    resp = requests.get(url, params={"id": series_id}, timeout=25)
    resp.raise_for_status()
    raw = pd.read_csv(io.StringIO(resp.text))
    if raw.shape[1] < 2:
        raise ValueError("FRED returned no observations")
    result = pd.DataFrame({"date": pd.to_datetime(raw.iloc[:, 0], errors="coerce"),
                           "value": pd.to_numeric(raw.iloc[:, 1], errors="coerce")})
    return result.dropna().sort_values("date").drop_duplicates("date")

def read_uploaded(uploaded, choice):
    if uploaded is None:
        return pd.DataFrame(columns=["date", "value"])
    data = pd.read_csv(uploaded)
    if not {"date", "value"}.issubset(data.columns):
        raise ValueError("CSV requires date,value columns (optional commodity column).")
    if "commodity" in data.columns:
        data = data.loc[data["commodity"].astype(str) == choice]
    out = pd.DataFrame({"date": pd.to_datetime(data["date"], errors="coerce"),
                        "value": pd.to_numeric(data["value"], errors="coerce")})
    return out.dropna().sort_values("date").drop_duplicates("date")

def make_forecast(data, months, annual_growth, uncertainty):
    """Transparent compound-growth planning scenario; not a market price forecast."""
    latest = float(data.iloc[-1]["value"])
    dates = pd.date_range(data.iloc[-1]["date"] + pd.offsets.MonthBegin(1), periods=months, freq="MS")
    t = np.arange(1, months + 1, dtype=float) / 12
    base = latest * np.power(max(0.01, 1 + annual_growth / 100), t)
    lower = latest * np.power(max(0.01, 1 + (annual_growth - uncertainty) / 100), t)
    upper = latest * np.power(max(0.01, 1 + (annual_growth + uncertainty) / 100), t)
    return pd.DataFrame({"date": dates, "base": base, "lower": lower, "upper": upper})

def render_expanded_commodities(history_years=3, forecast_horizon=12):
    st.subheader("Additional commodity indices & battery costs")
    st.caption("Live public BLS/FRED history where available; uploaded benchmarks for silicon and LFP/NMC. The curves ahead are adjustable procurement scenarios, not observed futures prices.")
    choice = st.selectbox("Commodity", list(SERIES), key="expanded_commodity")
    series_id, unit, source = SERIES[choice]
    upload = st.file_uploader("Optional benchmark history CSV (date,value; optional commodity)", type="csv", key="expanded_upload")
    if upload is not None:
        try:
            history = read_uploaded(upload, choice)
            source = "User-uploaded benchmark"
        except Exception as exc:
            st.error(f"Invalid CSV: {exc}")
            return
    elif series_id:
        try:
            history = fetch_fred(series_id)
        except (requests.RequestException, ValueError, pd.errors.ParserError) as exc:
            st.warning(f"Public series unavailable right now: {exc}")
            return
    else:
        history = pd.DataFrame(columns=["date", "value"])
    st.caption(f"Source: {source} | Unit: {unit}" + (f" | [FRED series](https://fred.stlouisfed.org/series/{series_id})" if series_id else ""))
    if history.empty:
        st.info("No verified historical time series loaded for this grade/chemistry. Upload a licensed or supplier benchmark CSV to activate its history and scenarios.")
        st.download_button("Download CSV template", "date,value\n2025-01-01,\n", file_name="commodity_history_template.csv", mime="text/csv", key="expanded_template")
        return
    history = history[history["value"] > 0].copy()
    if history.empty:
        st.warning("All values are zero or negative; cannot model percentage changes.")
        return
    display_hist = history.loc[history["date"] >= history["date"].max() - pd.DateOffset(years=history_years)]
    a, b, c = st.columns(3)
    latest = float(history.iloc[-1]["value"])
    a.metric("Latest benchmark", f"{latest:,.2f} {unit}")
    b.metric("Latest observation", history.iloc[-1]["date"].strftime("%b %Y"))
    yoy_base = history.loc[history["date"] <= history.iloc[-1]["date"] - pd.DateOffset(months=12)]
    c.metric("12-month observed change", f"{(latest / float(yoy_base.iloc[-1]['value']) - 1)*100:+.1f}%" if not yoy_base.empty else "Not available")
    a1, b1 = st.columns(2)
    growth = a1.slider("Base annual change (%)", -40, 40, 0, 1, key="expanded_growth")
    spread = b1.slider("Upside/downside range (percentage points)", 0, 50, 10, 1, key="expanded_spread")
    fc = make_forecast(history, forecast_horizon, growth, spread)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=display_hist["date"], y=display_hist["value"], name="Historical benchmark", mode="lines+markers"))
    bridge = pd.DataFrame({"date": [history.iloc[-1]["date"]], "base": [latest], "lower": [latest], "upper": [latest]})
    curve = pd.concat([bridge, fc], ignore_index=True)
    fig.add_trace(go.Scatter(x=curve["date"], y=curve["upper"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=curve["date"], y=curve["lower"], line=dict(width=0), fill="tonexty", name="Scenario range"))
    fig.add_trace(go.Scatter(x=curve["date"], y=curve["base"], mode="lines", line=dict(dash="dash"), name="Base scenario"))
    fig.update_layout(xaxis_title=None, yaxis_title=unit, hovermode="x unified", legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    st.info("BLS producer-price indices are not dollar prices. For purchases, combine these escalation signals with current supplier quotes. Scenario curves are assumption-based—not BloombergNEF, CME, or analyst forecasts.")
    st.download_button("Export historical benchmark CSV", history.to_csv(index=False), "expanded_commodity_history.csv", "text/csv", key="expanded_hist_download")
    st.download_button("Export scenario CSV", fc.to_csv(index=False), "expanded_commodity_scenarios.csv", "text/csv", key="expanded_fc_download")
    with st.expander("Commodity benchmark definitions"):
        st.markdown("""- **Silicon**: choose silicon metal or polysilicon by purity and procurement use; public USGS annual statistics are not a live transaction-price series.
- **Plastic**: the broad resin PPI does not distinguish PP, PE, PVC, PET, or ABS; use vendor-specific uploads for individual resin grades.
- **Fuel**: diesel producer-price index is not retail $/gallon or a fuel forward contract.
- **Glass**: flat-glass PPI is not a solar module glass quotation.
- **LFP / NMC**: pack and cell cost differ; upload price observations with consistent geography, application, and $/kWh basis.
- **Steel / aluminum**: PPI direction is not a live traded futures price or a fabrication quotation.""")
