import altair as alt
from .fact_visualizer import FactVisualizer

fade_opacity = 0.15

class Radial(FactVisualizer):
    def vis_base_chart(self, df, subject, x_type, y_type, c_type, orient) -> alt.Chart:
        base = alt.Chart(df).encode(
            alt.Theta(y_type).stack(True),
            alt.Radius(subject['measure']).scale(type="sqrt", zero=True, rangeMin=20),
            color=f"{subject['breakdown']}:N",
            order=alt.Order(subject['measure'], sort="ascending"),
        )
        return alt.layer(
            base.mark_arc(innerRadius=20, stroke="#fff"),
            base.mark_text(radiusOffset=10).encode(text=y_type),
        ).properties(width=450, height=300)

    def vis_proportion_fact(self, base, subject, x):
        return alt.layer(
            base.encode(
                opacity=alt.condition(
                    alt.datum[subject["breakdown"]] == x,
                    alt.value(1),
                    alt.value(fade_opacity)
                )
            )
        ).to_json()
    
    def vis_extremum_fact(self, base, subject, x):
        return alt.layer(
            base.encode(
                opacity=alt.condition(
                    alt.datum[subject["breakdown"]] == x,
                    alt.value(1),
                    alt.value(fade_opacity)
                )
            )
        ).to_json()

    def vis_difference_fact(self, base, subject, x1, x2):
        return alt.layer(
            base.encode(
                opacity=alt.condition(
                    (alt.datum[subject["breakdown"]] == x1) | (alt.datum[subject["breakdown"]] == x2),
                    alt.value(1),
                    alt.value(fade_opacity)
                )
            )
        ).to_json()
    
    def vis_rank_fact(self, base, subject, x1, x2, x3):
        return None
    
    def vis_outlier_fact(self, base, subject, x):
        return None
    
    def vis_trend_fact(self, base, subject, x1, x2, trend, span):
        return None
    
    def vis_association_fact(self, base, subject, correlation):
        return None
    
    def vis_value_fact(self, base, subject, x):
        return alt.layer(
            base.encode(
                opacity=alt.condition(
                    alt.datum[subject["breakdown"]] == x,
                    alt.value(1),
                    alt.value(fade_opacity)
                )
            )
        ).to_json()
