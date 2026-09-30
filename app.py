
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="Solar EPC Cost Intelligence V3", page_icon="☀️", layout="wide")

BENCHMARKS = {
    "Steel — finished products": ("WPUSISTEEL2", "Index 1982=100", "BLS/FRED", "monthly"),
    "Steel — structural shapes/plate": ("WPU101704", "Index Jun 1982=100", "BLS/FRED", "monthly"),
    "Copper wire & cable": ("WPU10260314", "Index Dec 1986=100", "BLS/FRED", "monthly"),
    "Construction labor — wages": ("ECICONWAG", "Index Dec 2005=100", "BLS/FRED", "quarterly"),
    "Switchgear & switchboards": ("WPU117522", "Index Jun 2005=100", "BLS/FRED", "monthly"),
    "Switchgear — excluding relays/ducts": ("WPU11752201A", "Index Dec 2007=100", "BLS/FRED", "monthly"),
    "Transformers — power & distribution": ("WPU117409", "Index Dec 1999=100", "BLS/FRED", "monthly"),
    "Electronic components — inverter context": ("PCU33443344", "Index Dec 1984=100", "BLS/FRED", "monthly"),
    "PV/semiconductor trade context": ("ID8541", "Index Dec 2017=100", "BLS/FRED", "monthly"),
}

DEFAULT_QUOTES = pd.DataFrame([
    ["Module","U.S.-made","Example vendor quote",0.00,"$/Wdc","Replace with actual quote"],
    ["Module","Non-U.S.","Example vendor quote",0.00,"$/Wdc","Replace with actual quote"],
    ["Inverter","Utility-scale string/central","Example vendor quote",0.00,"$/Wac","Replace with actual quote"],
    ["Tracker / racking","Project-specific","Example vendor quote",0.00,"$/Wdc","Replace with actual quote"],
    ["MV equipment","Project-specific","Example vendor quote",0.00,"$/project","Replace with actual quote"],
    ["BOS","Electrical BOS","Example vendor quote",0.00,"$/Wdc","Replace with actual quote"],
], columns=["category","subcategory","source","price","unit","note"])

DEFAULT_PROJECTS = pd.DataFrame([
    ["Example 5 MWdc",5.0,0.28,0.07,0.10,0.12,0.08,0.15,0.20],
], columns=["project","mw_dc","modules","inverters","racking_steel","wire_copper_al","gear_transformers","labor","other_bos"])

DEFAULT_CME = pd.DataFrame({
    "contract_month": pd.to_datetime(["2026-10-01","2026-11-01","2026-12-01","2027-01-01","2027-03-01","2027-07-01"]),
    "last_usd_per_short_ton": [1285,1323,1330,1333,1279,1190],
    "volume": [24,65,12,2,7,5],
    "source_date": ["2026-09-25"]*6,
    "contract": ["HRCV6","HRCX6","HRCZ6","HRCF7","HRCH7","HRCN7"],
})

@st.cache_data(ttl=3600)
def fred(series):
    url=f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    r=requests.get(url,timeout=20)
    r.raise_for_status()
    d=pd.read_csv(io.StringIO(r.text))
    d.columns=["date","value"]
    d["date"]=pd.to_datetime(d["date"])
    d["value"]=pd.to_numeric(d["value"],errors="coerce")
    return d.dropna().sort_values("date")

def pct(a,b):
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return (a/b-1)*100

def latest_period_change(d):
    return pct(d.iloc[-1]["value"], d.iloc[-2]["value"]) if len(d) > 1 else np.nan

def yoy_change(d, frequency):
    lag = 12 if frequency=="monthly" else 4
    return pct(d.iloc[-1]["value"], d.iloc[-1-lag]["value"]) if len(d) > lag else np.nan

def three_year_change(d, frequency):
    lag = 36 if frequency=="monthly" else 12
    return pct(d.iloc[-1]["value"], d.iloc[-1-lag]["value"]) if len(d) > lag else np.nan

def trend_label(v):
    if pd.isna(v): return "Insufficient data"
    if v > 2: return "Rising"
    if v < -2: return "Falling"
    return "Stable / mixed"

def seasonal_trend_forecast(df, frequency="monthly", periods=12):
    d=df.copy().sort_values("date")
    n_recent = 60 if frequency=="monthly" else 24
    recent=d.tail(min(n_recent, len(d))).copy()
    y=recent["value"].values.astype(float)
    x=np.arange(len(y), dtype=float)
    if len(y) < 4:
        return pd.DataFrame()
    slope, intercept=np.polyfit(x,y,1)
    trend=intercept+slope*x
    resid=y-trend
    if frequency=="monthly":
        keys=recent["date"].dt.month.values
        future_dates=pd.date_range(recent["date"].max()+pd.offsets.MonthBegin(1), periods=periods, freq="MS")
        future_keys=future_dates.month
    else:
        keys=recent["date"].dt.quarter.values
        future_dates=pd.date_range(recent["date"].max()+pd.offsets.QuarterBegin(startingMonth=1), periods=periods, freq="QS")
        future_keys=future_dates.quarter
    season_map={}
    for k in sorted(set(keys)):
        vals=resid[keys==k]
        season_map[int(k)] = float(np.mean(vals)) if len(vals) else 0.0
    fx=np.arange(len(y), len(y)+periods, dtype=float)
    base=intercept+slope*fx
    seasonal=np.array([season_map.get(int(k),0.0) for k in future_keys])
    forecast=base+seasonal
    sigma=float(np.nanstd(resid,ddof=1)) if len(resid)>2 else 0.0
    h=np.arange(1,periods+1)
    band=1.96*sigma*np.sqrt(1+h/max(len(y),1))
    return pd.DataFrame({
        "date": future_dates,
        "forecast": forecast,
        "lower": forecast-band,
        "upper": forecast+band,
    })

def parse_cme(uploaded):
    if uploaded is None:
        return DEFAULT_CME.copy()
    d=pd.read_csv(uploaded)
    req={"contract_month","last_usd_per_short_ton"}
    if not req.issubset(d.columns):
        raise ValueError("CME CSV requires contract_month and last_usd_per_short_ton.")
    d["contract_month"]=pd.to_datetime(d["contract_month"])
    d["last_usd_per_short_ton"]=pd.to_numeric(d["last_usd_per_short_ton"],errors="coerce")
    if "volume" in d.columns:
        d["volume"]=pd.to_numeric(d["volume"],errors="coerce")
    return d.dropna(subset=["contract_month","last_usd_per_short_ton"]).sort_values("contract_month")

st.title("Solar EPC Cost & Commodity Intelligence — V3")
st.caption("Three-year history, forecast scenarios, forward pricing, vendor quotes and project exposure for solar preconstruction.")

with st.sidebar:
    st.header("Dashboard controls")
    history_years=st.selectbox("Historical window", [1,3,5], index=1)
    forecast_horizon=st.selectbox("Forecast horizon", [6,12,18,24], index=1)
    quote_upload=st.file_uploader("Upload component/vendor quotes CSV",type="csv")
    project_upload=st.file_uploader("Upload project cost mix CSV",type="csv")
    cme_upload=st.file_uploader("Upload refreshed CME HRC curve CSV",type="csv")

quotes=pd.read_csv(quote_upload) if quote_upload else DEFAULT_QUOTES.copy()
projects=pd.read_csv(project_upload) if project_upload else DEFAULT_PROJECTS.copy()
try:
    cme=parse_cme(cme_upload)
except Exception as e:
    st.error(str(e))
    cme=DEFAULT_CME.copy()

tabs=st.tabs([
    "Executive dashboard","3-Year history + forecast","3-Year comparison",
    "Commodities","Labor","Modules","Inverters","Gear & transformers",
    "BOS","Project exposure","Quote tracker","Sources"
])

with tabs[0]:
    st.subheader("Market pulse")
    pulse_names=[
        "Steel — structural shapes/plate","Copper wire & cable","Construction labor — wages",
        "Switchgear & switchboards","Transformers — power & distribution"
    ]
    cols=st.columns(len(pulse_names))
    rows=[]
    for col,name in zip(cols,pulse_names):
        sid,unit,source,freq=BENCHMARKS[name]
        try:
            d=fred(sid)
            latest=d.iloc[-1]["value"]
            lp=latest_period_change(d)
            yoy=yoy_change(d,freq)
            t3=three_year_change(d,freq)
            sig=trend_label(yoy)
            with col:
                st.metric(name.split(" — ")[0],f"{latest:.1f}",f"{lp:+.1f}% latest period")
                st.caption(f"{sig} • {d.iloc[-1]['date'].strftime('%b %Y')}")
            fc_periods = forecast_horizon if freq=="monthly" else max(1,int(np.ceil(forecast_horizon/3)))
            fc=seasonal_trend_forecast(d,freq,fc_periods)
            f6 = pct(fc.iloc[min(len(fc)-1, 5 if freq=="monthly" else 1)]["forecast"], latest) if len(fc) else np.nan
            f12 = pct(fc.iloc[min(len(fc)-1, 11 if freq=="monthly" else 3)]["forecast"], latest) if len(fc) else np.nan
            rows.append([name,latest,yoy,t3,f6,f12,sig,d.iloc[-1]["date"].date()])
        except Exception:
            pass
    st.info("Forecasts are model outputs from historical BLS/FRED indexes. They are not supplier quotes. Futures, where available, are shown separately.")
    summary=pd.DataFrame(rows,columns=["Cost driver","Latest index","YoY %","3Y %","Forecast ~6M %","Forecast ~12M %","Direction","Data through"])
    for c in ["YoY %","3Y %","Forecast ~6M %","Forecast ~12M %"]:
        summary[c]=summary[c].round(1)
    st.dataframe(summary,use_container_width=True,hide_index=True)

with tabs[1]:
    st.subheader("Historical actuals + forecast")
    choice=st.selectbox("Cost driver",list(BENCHMARKS.keys()),index=1,key="hist_fc_choice")
    sid,unit,source,freq=BENCHMARKS[choice]
    d=fred(sid)
    periods=forecast_horizon if freq=="monthly" else max(1,int(np.ceil(forecast_horizon/3)))
    fc=seasonal_trend_forecast(d,freq,periods)
    n = (12 if history_years==1 else 36 if history_years==3 else 60) if freq=="monthly" else (4 if history_years==1 else 12 if history_years==3 else 20)
    hist=d.tail(n)
    latest=d.iloc[-1]["value"]
    yoy=yoy_change(d,freq)
    t3=three_year_change(d,freq)
    a,b,c,dcol=st.columns(4)
    a.metric("Latest index",f"{latest:.2f}")
    b.metric("YoY change","—" if pd.isna(yoy) else f"{yoy:+.1f}%")
    c.metric("3-year change","—" if pd.isna(t3) else f"{t3:+.1f}%")
    f_end=pct(fc.iloc[-1]["forecast"],latest) if len(fc) else np.nan
    dcol.metric(f"{forecast_horizon}M model change","—" if pd.isna(f_end) else f"{f_end:+.1f}%")

    fig=go.Figure()
    fig.add_trace(go.Scatter(x=hist["date"],y=hist["value"],mode="lines+markers",name="Historical actual"))
    if len(fc):
        fig.add_trace(go.Scatter(x=fc["date"],y=fc["upper"],mode="lines",line=dict(width=0),showlegend=False))
        fig.add_trace(go.Scatter(x=fc["date"],y=fc["lower"],mode="lines",fill="tonexty",line=dict(width=0),name="Illustrative forecast band"))
        fig.add_trace(go.Scatter(x=fc["date"],y=fc["forecast"],mode="lines+markers",line=dict(dash="dash"),name="Statistical forecast"))
        fig.add_vline(x=d.iloc[-1]["date"],line_dash="dot",annotation_text="Forecast starts",annotation_position="top")
    fig.update_layout(title=f"{choice}: history + forecast",yaxis_title=unit,xaxis_title=None,hovermode="x unified")
    st.plotly_chart(fig,use_container_width=True)

    if choice.startswith("Steel") and len(cme):
        st.markdown("#### CME HRC market forward curve")
        fig2=px.line(cme,x="contract_month",y="last_usd_per_short_ton",markers=True)
        fig2.update_layout(yaxis_title="USD / short ton",xaxis_title=None,hovermode="x unified")
        st.plotly_chart(fig2,use_container_width=True)
        st.caption("CME HRC futures are market-observed forward prices, not the same unit as the BLS steel index and not a guaranteed future spot price.")

    st.warning("The forecast is a transparent statistical extrapolation based on recent trend + seasonality. Use it for planning ranges, not as a guaranteed procurement price.")

with tabs[2]:
    st.subheader("3-Year normalized comparison")
    st.caption("All selected series are rebased to 100 at the start so you can compare percentage movement across different indexes.")
    default_sel=["Steel — structural shapes/plate","Copper wire & cable","Construction labor — wages","Switchgear & switchboards","Transformers — power & distribution"]
    selected=st.multiselect("Compare cost drivers",list(BENCHMARKS.keys()),default=default_sel)
    fig=go.Figure()
    compare_rows=[]
    for name in selected:
        sid,unit,source,freq=BENCHMARKS[name]
        try:
            d=fred(sid)
            n=36 if freq=="monthly" else 12
            v=d.tail(n).copy()
            if len(v):
                v["normalized"]=v["value"]/v.iloc[0]["value"]*100
                fig.add_trace(go.Scatter(x=v["date"],y=v["normalized"],mode="lines",name=name))
                compare_rows.append([name,pct(v.iloc[-1]["value"],v.iloc[0]["value"])])
        except Exception:
            pass
    fig.update_layout(yaxis_title="Rebased index (start = 100)",xaxis_title=None,hovermode="x unified")
    st.plotly_chart(fig,use_container_width=True)
    cr=pd.DataFrame(compare_rows,columns=["Cost driver","3Y move %"]).sort_values("3Y move %",ascending=False)
    if len(cr):
        cr["3Y move %"]=cr["3Y move %"].round(1)
    st.dataframe(cr,use_container_width=True,hide_index=True)

with tabs[3]:
    st.subheader("Commodities")
    choice=st.selectbox("Commodity benchmark",["Steel — finished products","Steel — structural shapes/plate","Copper wire & cable"],key="commodity")
    sid,unit,_,freq=BENCHMARKS[choice]
    d=fred(sid)
    n=36 if history_years==3 else (12 if history_years==1 else 60)
    fig=px.line(d.tail(n),x="date",y="value",markers=True,title=f"{choice} — {history_years}Y history")
    fig.update_layout(yaxis_title=unit,xaxis_title=None,hovermode="x unified")
    st.plotly_chart(fig,use_container_width=True)
    st.warning("Aluminum conductor remains quote-driven in this version because the dedicated public aluminum-wire series is stale. Add current vendor quotes in Quote Tracker.")

with tabs[4]:
    st.subheader("Construction labor escalation")
    sid,unit,_,freq=BENCHMARKS["Construction labor — wages"]
    d=fred(sid)
    fig=px.line(d.tail(12),x="date",y="value",markers=True,title="Construction wages — 3-year quarterly history")
    fig.update_layout(yaxis_title=unit,xaxis_title=None,hovermode="x unified")
    st.plotly_chart(fig,use_container_width=True)
    st.metric("Year-over-year labor escalation","—" if pd.isna(yoy_change(d,freq)) else f"{yoy_change(d,freq):+.1f}%")
    st.caption("Quarterly labor data are compared QoQ and YoY rather than using monthly-period labels.")

with tabs[5]:
    st.subheader("Modules — U.S. vs non-U.S.")
    st.info("Actual module quote history is kept separate from broad public market indexes. Upload vendor quote history as you collect it.")
    mq=quotes[quotes["category"].astype(str).str.lower()=="module"].copy()
    st.data_editor(mq,use_container_width=True,hide_index=True,num_rows="dynamic")
    sid,unit,_,freq=BENCHMARKS["PV/semiconductor trade context"]
    d=fred(sid)
    fig=px.line(d.tail(36),x="date",y="value",markers=True,title="PV / semiconductor trade-price context — 3 years")
    fig.update_layout(yaxis_title=unit,xaxis_title=None)
    st.plotly_chart(fig,use_container_width=True)

with tabs[6]:
    st.subheader("Inverters")
    st.warning("The public electronics index is context only; actual utility-scale inverter bids should be tracked in $/Wac or $/kWac.")
    sid,unit,_,freq=BENCHMARKS["Electronic components — inverter context"]
    d=fred(sid)
    fig=px.line(d.tail(36),x="date",y="value",markers=True,title="Electronic components — 3-year context")
    fig.update_layout(yaxis_title=unit,xaxis_title=None)
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe(quotes[quotes["category"].astype(str).str.lower()=="inverter"],use_container_width=True,hide_index=True)

with tabs[7]:
    st.subheader("Gear & transformers")
    equip=st.selectbox("Equipment",["Switchgear & switchboards","Switchgear — excluding relays/ducts","Transformers — power & distribution"],key="gear")
    sid,unit,_,freq=BENCHMARKS[equip]
    d=fred(sid)
    fc=seasonal_trend_forecast(d,freq,forecast_horizon)
    hist=d.tail(36)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=hist["date"],y=hist["value"],mode="lines+markers",name="Actual"))
    if len(fc):
        fig.add_trace(go.Scatter(x=fc["date"],y=fc["forecast"],mode="lines+markers",line=dict(dash="dash"),name="Forecast"))
    fig.update_layout(title=f"{equip}: 3-year history + {forecast_horizon}M forecast",yaxis_title=unit,xaxis_title=None,hovermode="x unified")
    st.plotly_chart(fig,use_container_width=True)

with tabs[8]:
    st.subheader("Balance of System (BOS)")
    st.markdown("""
Track these as project/vendor quote histories:
- Tracker/racking/piles
- DC wire and connectors
- AC cable / MV cable
- Combiner equipment / disconnects
- Switchgear / relays
- Transformers
- Conduit / trench / duct bank
- Grounding
- SCADA / DAS / metering
- Fencing / gates / security
- Roads / aggregate / drainage
- Freight / logistics
    """)
    bos=quotes[quotes["category"].astype(str).str.lower().isin(["bos","tracker / racking","mv equipment"])].copy()
    st.dataframe(bos,use_container_width=True,hide_index=True)

with tabs[9]:
    st.subheader("Project / portfolio cost exposure")
    edited=st.data_editor(projects,use_container_width=True,hide_index=True,num_rows="dynamic")
    costcols=["modules","inverters","racking_steel","wire_copper_al","gear_transformers","labor","other_bos"]
    if len(edited):
        e=edited.copy()
        for c in ["mw_dc"]+costcols:
            e[c]=pd.to_numeric(e[c],errors="coerce").fillna(0)
        e["total_$/Wdc"]=e[costcols].sum(axis=1)
        e["total_$"]=e["total_$/Wdc"]*e["mw_dc"]*1_000_000
        st.dataframe(e[["project","mw_dc","total_$/Wdc","total_$"]],use_container_width=True,hide_index=True)
        long=e.melt(id_vars=["project","mw_dc"],value_vars=costcols,var_name="component",value_name="$/Wdc")
        fig=px.bar(long,x="project",y="$/Wdc",color="component",title="Project cost stack")
        st.plotly_chart(fig,use_container_width=True)
        st.download_button("Download project exposure CSV",e.to_csv(index=False).encode(),"solar_project_cost_exposure.csv","text/csv")

with tabs[10]:
    st.subheader("Vendor / component quote tracker")
    edited_quotes=st.data_editor(quotes,use_container_width=True,hide_index=True,num_rows="dynamic")
    st.download_button("Download quote tracker",edited_quotes.to_csv(index=False).encode(),"component_quotes.csv","text/csv")
    st.caption("Recommended next fields: vendor, project, quote date, valid-through date, lead time, quantity, Incoterm/freight, tariff assumption and domestic-content status.")

with tabs[11]:
    st.subheader("Source map")
    st.dataframe(pd.DataFrame([
        ["Structural steel","BLS/FRED","WPU101704","Auto","Historical + model forecast"],
        ["Copper wire & cable","BLS/FRED","WPU10260314","Auto","Historical + model forecast"],
        ["Construction labor","BLS/FRED","ECICONWAG","Auto","Quarterly history + model forecast"],
        ["Switchgear","BLS/FRED","WPU117522","Auto","Historical + model forecast"],
        ["Transformers","BLS/FRED","WPU117409","Auto","Historical + model forecast"],
        ["HRC forward curve","CME Group","HRC futures","CSV/licensed feed","Market forward curve"],
        ["Modules U.S./non-U.S.","Vendor quote database","—","Manual now","Actual $/Wdc history"],
        ["Inverters","Vendor quote database","—","Manual now","Actual $/Wac or $/kWac history"],
        ["Aluminum conductor","Vendor/licensed metal feed","—","Manual now","Actual $/ft or material quote"],
        ["BOS","Project/vendor database","—","Manual now","Actual project pricing"],
    ],columns=["Component","Source","Series / feed","Refresh","Use"]),use_container_width=True,hide_index=True)
    st.caption("Public indexes are escalation benchmarks. Vendor/EPC quotes establish project-specific pricing.")

st.divider()
st.caption("Planning dashboard only. Forecasts are statistical scenarios, not guaranteed commodity or equipment prices.")
