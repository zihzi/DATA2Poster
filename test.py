# import pandas as pd
# import json
import altair as alt
# import altair_saver
# df = pd.read_csv("data/cas_cancer_nc.csv")
# df.drop(columns=["id"],inplace=True)
# df.drop(columns=["ICD-10"],inplace=True)
# df.to_csv("data/cas_cancer_nc.csv", index=False)

import pandas as pd


dic ={
  "$schema": "https://vega.github.io/schema/vega-lite/v5.17.0.json",
  "config": {
    "axis": {
      "labelFontSize": 12,
      "titleFontSize": 14
    },
    "title": {
      "anchor": "start",
      "fontSize": 16
    },
    "view": {
      "continuousHeight": 300,
      "continuousWidth": 300
    }
  },
  "data": {
    "url": "https://raw.githubusercontent.com/zihzi/DATA2Poster/refs/heads/main/data/Movies.csv"
  },
  "encoding": {
    "color": {
      "field": "Genre",
      "legend": {
        "title": "Genre"
      },
      "scale": {
        "domain": [
          "Drama",
          "Thriller"
        ],
        "range": [
          "#1E6091",
          "#168AAD"
        ]
      },
      "type": "nominal"
    },
    "tooltip": [
      {
        "field": "Genre",
        "title": "Genre",
        "type": "nominal"
      },
      {
        "field": "Worldwide Gross",
        "format": "$,",
        "title": "Total Worldwide Gross",
        "type": "quantitative"
      }
    ],
    "x": {
      "axis": {
        "labelAngle": 0
      },
      "field": "Genre",
      "title": "Genre",
      "type": "nominal"
    },
    "y": {
      "axis": {
        "format": "$,",
    
      },
      "field": "Worldwide Gross",
      "title": "Total Worldwide Gross (USD)",
      "type": "quantitative"
    }
  },
  "height": 300,
  "mark": {
    "type": "bar"
  },
  "title": "Total Worldwide Gross for Drama vs. Thriller Genres",
  "width": 400
}
chart = alt.Chart.from_dict(dic)
chart.save('chart.json')
chart.save('chart.png')