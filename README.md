# Solar EPC Cost & Commodity Intelligence v2

Expanded from the U.S. Steel Market Tracker into a solar preconstruction/procurement dashboard.

## Included
- Steel / structural steel
- Copper wire & cable
- Construction labor escalation
- Modules: U.S. vs non-U.S. quote tracking
- Inverter quote tracking + public proxy context
- Switchgear
- Power/distribution transformers
- BOS categories
- Project $/Wdc cost stack
- Portfolio exposure
- Vendor quote tracker
- CSV exports

## Automatic public sources
The app retrieves FRED CSV feeds for BLS/FRED series including:
- WPUSISTEEL2 — finished steel
- WPU101704 — structural steel shapes/plate
- WPU10260314 — copper wire & cable
- ECICONWAG — construction wages
- WPU117522 — switchgear/switchboards
- WPU117409 — transformers
- PCU33443344 — electronic components (context only)
- ID8541 — PV/semiconductor trade context (context only)

## Important source design
Not every solar component has a reliable free public $/W series.
Modules, inverters, aluminum conductor and BOS should therefore combine:
1. public benchmark indexes for escalation/context; and
2. actual vendor/EPC quote history for project pricing.

This avoids presenting a broad proxy index as if it were an actual supplier price.

## Run
pip install -r requirements.txt
streamlit run app.py
