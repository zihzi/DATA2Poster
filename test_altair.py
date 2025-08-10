import json
import altair as alt   # pip install altair

# ❶ load or define your spec
spec = json.load(open("test_altair.json"))

# ❷ wrap the dict in an Altair top-level object
#     • alt.VegaLite(spec)                  ← simplest
#     • alt.Chart.from_dict(spec)           ← identical, can skip schema validation
chart = alt.Chart.from_dict(spec)
chart.save("test_altair_chart.png")

