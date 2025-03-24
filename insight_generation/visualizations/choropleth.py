import altair as alt
from .fact_visualizer import FactVisualizer
from vega_datasets import data

class Choropleth(FactVisualizer):
    def vis_base_chart(self, df, subject, x_type, y_type, c_type, orient) -> alt.Chart:
        states = alt.topo_feature(data.us_10m.url, 'states')
        return alt.Chart(states).mark_geoshape().encode(
            color=y_type,
            stroke=alt.value('red'),
            strokeWidth=alt.value(0)
        ).transform_lookup(
            lookup='id',
            from_=alt.LookupData(df, 'id', [subject['measure'], subject['breakdown']])
        ).project(
            type='albersUsa'
        ).properties(
            width=450,
            height=300
        )

    def vis_proportion_fact(self, base, subject, x):
        return alt.layer(
            base.encode(
                opacity=alt.condition(
                    alt.datum[subject["breakdown"]] == x,
                    alt.value(1),
                    alt.value(0.3)
                ),
                strokeWidth=alt.condition(
                    alt.datum[subject["breakdown"]] == x,
                    alt.value(2),
                    alt.value(0)
                ),
            )
        ).to_json()
    
    def vis_extremum_fact(self, base, subject, x):
        return alt.layer(
            base.encode(
                opacity=alt.condition(
                    alt.datum[subject["breakdown"]] == x,
                    alt.value(1),
                    alt.value(0.3)
                ),
                strokeWidth=alt.condition(
                    alt.datum[subject["breakdown"]] == x,
                    alt.value(2),
                    alt.value(0)
                ),
            )
        ).to_json()
    
    def vis_rank_fact(self, base, subject, x1, x2, x3):
        return alt.layer(
            base.encode(
                opacity=alt.condition(
                    (alt.datum[subject["breakdown"]] == x1) | (alt.datum[subject["breakdown"]] == x2) | (alt.datum[subject["breakdown"]] == x3),
                    alt.value(1),
                    alt.value(0.3)
                ),
                strokeWidth=alt.condition(
                    (alt.datum[subject["breakdown"]] == x1) | (alt.datum[subject["breakdown"]] == x2) | (alt.datum[subject["breakdown"]] == x3),
                    alt.value(2),
                    alt.value(0)
                ),
            )
        ).to_json()
    
    def vis_outlier_fact(self, base, subject, x):
        return None

    def vis_difference_fact(self, base, subject, x1, x2):
        return None
    
    def vis_trend_fact(self, base, subject, x1, x2, trend, span):
        return None
    
    def vis_association_fact(self, base, subject, correlation):
        return None
    
    def vis_value_fact(self, base, subject, x) -> str:
        return alt.layer(
            base.encode(
                opacity=alt.condition(
                    alt.datum[subject["breakdown"]] == x,
                    alt.value(1),
                    alt.value(0.3)
                ),
                strokeWidth=alt.condition(
                    alt.datum[subject["breakdown"]] == x,
                    alt.value(2),
                    alt.value(0)
                ),
            )
        ).to_json()
