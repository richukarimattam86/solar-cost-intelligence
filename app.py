
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="Solar EPC Cost Intelligence", page_icon="☀️", layout="wide")

BENCHMARKS = {
    "Steel — finished products": ("WPUSISTEEL2", "Index 1982=100", "BLS/FRED", "Direct public benchmark"),
    "Steel — structural shapes/plate": ("WPU101704", "Index Jun 1982=100", "BLS/FRED", "Direct public benchmark"),
    "Copper wire & cable": ("WPU10260314", "Index Dec 1986=100", "BLS/FRED", "Direct public benchmark"),
    "Construction labor — wages": ("ECICONWAG", "Index Dec 2005=100", "BLS/FRED", "Broad construction labor benchmark"),
    "Switchgear & switchboards": ("WPU117522", "Index Jun 2005=100", "BLS/FRED", "Direct public equipment benchmark"),
    "Switchgear — excluding relays/ducts": ("WPU11752201A", "Index Dec 2007=100", "BLS/FRED", "Direct public equipment benchmark"),
    "Transformers — power & distribution": ("WPU117409", "Index Dec 1999=100", "BLS/FRED", "Direct public equipment benchmark"),
    "Electronic components — inverter context": ("PCU33443344", "Index Dec 1984=100", "BLS/FRED", "Proxy only — not an inverter quote"),
    "PV/semiconductor trade context": ("ID8541", "Index Dec 2017=100", "BLS/FRED", "Proxy only — not a module $/W quote"),
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

@st.cache_data(ttl=3600)
def fred(series):
    url=f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    r=requests.get(url,timeout=20); r.raise_for_status()
    d=pd.read_csv(io.StringIO(r.text))
    d.columns=["date","value"]; d["date"]=pd.to_datetime(d["date"])
    d["value"]=pd.to_numeric(d["value"],errors="coerce")
    return d.dropna().sort_values("date")

def changes(d):
    v=d["value"].dropna()
    latest=v.iloc[-1]
    one=v.iloc[-2] if len(v)>1 else np.nan
    three=v.iloc[-4] if len(v)>3 else np.nan
    twelve=v.iloc[-13] if len(v)>12 else np.nan
    def pc(old): return (latest/old-1)*100 if pd.notna(old) and old else np.nan
    return latest,pc(one),pc(three),pc(twelve)

def signal(p1,p3,p12):
    vals=[x for x in [p1,p3,p12] if pd.notna(x)]
    if not vals:return "Insufficient data"
    score=np.mean([np.sign(x) for x in vals])
    if score>.35:return "Rising"
    if score<-.35:return "Falling"
    return "Mixed / stable"

st.title("Solar EPC Cost & Commodity Intelligence")
st.caption("Procurement dashboard for utility-scale / C&I solar: commodities, labor, equipment, vendor quotes and project exposure.")

with st.sidebar:
    st.header("Portfolio controls")
    months=st.slider("History shown",12,120,36,6)
    st.caption("Public benchmarks refresh from FRED/BLS. Vendor pricing is intentionally maintained separately.")
    quote_upload=st.file_uploader("Upload vendor/component quotes CSV",type="csv")
    project_upload=st.file_uploader("Upload project cost mix CSV",type="csv")

quotes=pd.read_csv(quote_upload) if quote_upload else DEFAULT_QUOTES.copy()
projects=pd.read_csv(project_upload) if project_upload else DEFAULT_PROJECTS.copy()

tabs=st.tabs(["Executive dashboard","Commodities","Labor","Modules","Inverters","Gear & transformers","BOS","Project exposure","Quote tracker","Sources"])

with tabs[0]:
    st.subheader("Market pulse")
    cards=[]
    for name in ["Steel — structural shapes/plate","Copper wire & cable","Construction labor — wages","Switchgear & switchboards","Transformers — power & distribution"]:
        sid,unit,source,quality=BENCHMARKS[name]
        try:
            d=fred(sid); latest,p1,p3,p12=changes(d)
            cards.append((name,latest,p1,p3,p12,signal(p1,p3,p12),d.iloc[-1]["date"]))
        except Exception:
            cards.append((name,np.nan,np.nan,np.nan,np.nan,"Unavailable",pd.NaT))
    cols=st.columns(len(cards))
    for col,row in zip(cols,cards):
        name,latest,p1,p3,p12,sig,dt=row
        with col:
            st.metric(name.split(" — ")[0], "—" if pd.isna(latest) else f"{latest:.1f}", None if pd.isna(p1) else f"{p1:+.1f}% latest period")
            st.caption(f"{sig} • {'' if pd.isna(dt) else dt.strftime('%b %Y')}")
    st.info("Indexes are best used to measure escalation and direction. They are not equivalent to a supplier's delivered $/W, $/ft, $/kW or $/project quote.")
    rows=[]
    for name,(sid,unit,source,quality) in BENCHMARKS.items():
        try:
            d=fred(sid); latest,p1,p3,p12=changes(d)
            rows.append([name,latest,p1,p3,p12,signal(p1,p3,p12),d.iloc[-1]["date"].date(),quality])
        except Exception:
            pass
    pulse=pd.DataFrame(rows,columns=["Cost driver","Latest index","Latest-period %","~3-period %","~12-period %","Trend","Data through","Benchmark type"])
    st.dataframe(pulse,use_container_width=True,hide_index=True)

with tabs[1]:
    st.subheader("Commodity & conductor benchmarks")
    choice=st.selectbox("Benchmark",["Steel — finished products","Steel — structural shapes/plate","Copper wire & cable"])
    sid,unit,source,quality=BENCHMARKS[choice]
    d=fred(sid); view=d.tail(months)
    latest,p1,p3,p12=changes(d)
    a,b,c,dcol=st.columns(4)
    a.metric("Latest",f"{latest:.2f}",f"{p1:+.1f}% latest period")
    b.metric("~3-period change",f"{p3:+.1f}%")
    c.metric("~12-period change",f"{p12:+.1f}%")
    dcol.metric("Direction",signal(p1,p3,p12))
    fig=px.line(view,x="date",y="value",markers=True,title=choice)
    fig.update_layout(yaxis_title=unit,xaxis_title=None,hovermode="x unified")
    st.plotly_chart(fig,use_container_width=True)
    st.warning("Aluminum wire: the dedicated BLS/FRED aluminum-wire series ended in 2017. Do not use it as a current price signal. Add current aluminum conductor/vendor quotes in Quote Tracker; a licensed metal feed can be connected later.")

with tabs[2]:
    st.subheader("Construction labor escalation")
    sid,unit,_,_=BENCHMARKS["Construction labor — wages"]
    d=fred(sid); latest,p1,p3,p12=changes(d)
    fig=px.line(d.tail(max(20,months//3)),x="date",y="value",markers=True,title="Private construction wages — Employment Cost Index")
    fig.update_layout(yaxis_title=unit,xaxis_title=None,hovermode="x unified")
    st.plotly_chart(fig,use_container_width=True)
    st.metric("Approx. year-over-year wage escalation",f"{p3:+.1f}%" if len(d)<13 else f"{p3:+.1f}% (rough quarterly comparison)")
    st.caption("This is a national construction-wage benchmark. Project labor should also account for prevailing wage, union/open-shop conditions, state/region, per diem, productivity and schedule.")

with tabs[3]:
    st.subheader("PV modules — U.S. vs non-U.S.")
    st.info("Public indexes do not cleanly equal bankable module $/W quotes by origin. This page therefore separates market/trade context from actual vendor quotes.")
    sid,unit,_,_=BENCHMARKS["PV/semiconductor trade context"]
    d=fred(sid)
    fig=px.line(d.tail(months),x="date",y="value",markers=True,title="PV / photosensitive semiconductor trade-price context")
    fig.update_layout(yaxis_title=unit,xaxis_title=None)
    st.plotly_chart(fig,use_container_width=True)
    mq=quotes[quotes["category"].astype(str).str.lower()=="module"].copy()
    st.data_editor(mq,use_container_width=True,hide_index=True,num_rows="dynamic")
    st.markdown("**Recommended quote fields for production:** manufacturer, model, wattage, domestic-content status, origin, Incoterm, tariff assumptions, freight, safe-harbor/domestic-content eligibility, quote date, validity, quantity and $/Wdc.")

with tabs[4]:
    st.subheader("Inverter pricing")
    st.warning("Electronic-component indexes are contextual proxies only; they are not utility-scale inverter prices.")
    sid,unit,_,_=BENCHMARKS["Electronic components — inverter context"]
    d=fred(sid)
    fig=px.line(d.tail(months),x="date",y="value",markers=True,title="Electronic-component manufacturing PPI — contextual proxy")
    fig.update_layout(yaxis_title=unit,xaxis_title=None)
    st.plotly_chart(fig,use_container_width=True)
    iq=quotes[quotes["category"].astype(str).str.lower()=="inverter"].copy()
    st.dataframe(iq,use_container_width=True,hide_index=True)
    st.markdown("Track actual inverter bids in **$/Wac or $/kWac**, plus manufacturer, model, quantity, transformer inclusion, MV skid inclusion, freight, warranty, lead time and quote validity.")

with tabs[5]:
    st.subheader("Switchgear & transformers")
    equip=st.selectbox("Equipment",["Switchgear & switchboards","Switchgear — excluding relays/ducts","Transformers — power & distribution"])
    sid,unit,_,_=BENCHMARKS[equip]
    d=fred(sid); latest,p1,p3,p12=changes(d)
    fig=px.line(d.tail(months),x="date",y="value",markers=True,title=equip)
    fig.update_layout(yaxis_title=unit,xaxis_title=None)
    st.plotly_chart(fig,use_container_width=True)
    c1,c2,c3=st.columns(3); c1.metric("Latest",f"{latest:.2f}",f"{p1:+.1f}%")
    c2.metric("~3-period",f"{p3:+.1f}%"); c3.metric("~12-period",f"{p12:+.1f}%")
    st.caption("For project decisions, combine index escalation with actual OEM quotes and lead times. Transformer and switchgear pricing can move differently from raw steel/copper.")

with tabs[6]:
    st.subheader("Balance of System (BOS)")
    st.markdown("""
Use this section as the roll-up for costs that do not belong cleanly to modules/inverters:
- Racking / trackers / piles
- DC wire, homeruns and connectors
- AC cable and MV cable
- Combiner equipment / disconnects
- Switchboards / switchgear / relays
- Transformers
- Conduit, trenching and duct bank
- Grounding
- SCADA / DAS / communications
- Weather station / metering
- Fencing / gates / security
- Roads, aggregate, drainage and civil materials
- Freight / logistics
    """)
    bos=quotes[quotes["category"].astype(str).str.lower().isin(["bos","tracker / racking","mv equipment"])].copy()
    st.dataframe(bos,use_container_width=True,hide_index=True)

with tabs[7]:
    st.subheader("Project / portfolio cost exposure")
    st.caption("Enter cost components as $/Wdc. The tool converts them to project dollars and shows the portfolio cost mix.")
    edited=st.data_editor(projects,use_container_width=True,hide_index=True,num_rows="dynamic")
    costcols=["modules","inverters","racking_steel","wire_copper_al","gear_transformers","labor","other_bos"]
    if len(edited):
        e=edited.copy()
        for c in ["mw_dc"]+costcols:e[c]=pd.to_numeric(e[c],errors="coerce").fillna(0)
        e["total_$/Wdc"]=e[costcols].sum(axis=1)
        e["total_$"]=e["total_$/Wdc"]*e["mw_dc"]*1_000_000
        st.dataframe(e[["project","mw_dc","total_$/Wdc","total_$"]],use_container_width=True,hide_index=True)
        long=e.melt(id_vars=["project","mw_dc"],value_vars=costcols,var_name="component",value_name="$/Wdc")
        fig=px.bar(long,x="project",y="$/Wdc",color="component",title="Project cost stack")
        st.plotly_chart(fig,use_container_width=True)
        st.download_button("Download project exposure CSV",e.to_csv(index=False).encode(),"solar_project_cost_exposure.csv","text/csv")

with tabs[8]:
    st.subheader("Vendor / market quote tracker")
    st.caption("Use this for actual $/W, $/kW, $/ft, $/ton or lump-sum supplier pricing. Public indexes remain separate.")
    edited_quotes=st.data_editor(quotes,use_container_width=True,hide_index=True,num_rows="dynamic")
    st.download_button("Download quote tracker",edited_quotes.to_csv(index=False).encode(),"component_quotes.csv","text/csv")
    st.markdown("For better bid-leveling later, add columns for **vendor, project, quote date, valid-through date, lead time, quantity, Incoterm/freight, tariff assumption, domestic-content status and notes**.")

with tabs[9]:
    st.subheader("Data-source map")
    source_table=pd.DataFrame([
        ["Structural steel","BLS/FRED","WPU101704","Automatic","Public index"],
        ["Copper wire & cable","BLS/FRED","WPU10260314","Automatic","Public index"],
        ["Aluminum wire/cable","Vendor / licensed metals feed","—","Manual now","Dedicated BLS series is stale"],
        ["Construction labor","BLS/FRED","ECICONWAG","Automatic","National construction wages"],
        ["Modules — U.S.","Vendor / market quote dataset","—","Manual now","Track actual $/Wdc + domestic content"],
        ["Modules — non-U.S.","Vendor / market quote dataset","—","Manual now","Track origin/tariff/freight"],
        ["Inverters","Vendor quote + BLS proxy","PCU33443344","Mixed","Proxy is not inverter $/W"],
        ["Switchgear","BLS/FRED","WPU117522","Automatic","Public equipment index"],
        ["Transformers","BLS/FRED","WPU117409","Automatic","Public equipment index"],
        ["BOS","Project/vendor database","—","Manual now","Project-specific cost stack"],
        ["HRC forward curve","CME Group","HRC futures","CSV/licensed feed","Market forward pricing"],
    ],columns=["Component","Preferred source","Series / feed","Refresh","Use"])
    st.dataframe(source_table,use_container_width=True,hide_index=True)
    st.info("Best practice: public indexes measure escalation; actual supplier/EPC quotes establish project pricing. Keep both and compare them.")

st.divider()
st.caption("Procurement planning tool. Public indexes, market futures and proxy series are not supplier quotes. Validate major procurement decisions against current vendor offers and authorized market-data sources.")
