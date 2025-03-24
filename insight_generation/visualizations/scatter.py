import altair as alt
from .fact_visualizer import FactVisualizer

fade_opacity = 0.15

class Scatter(FactVisualizer):
    x_type = None
    y_type = None

    def vis_base_chart(self, df, subject, x_type=None, y_type=None, c_type=None, orient="verticle") -> alt.Chart:
        if self.x_type is None:
            self.x_type = x_type
        if self.y_type is None:
            self.y_type = y_type
        encodings = {
            "x": self.x_type,
            "y": self.y_type,
        } if subject["series"] is None else {
            "x": self.x_type,
            "y": self.y_type,    
            "color": alt.Color(
                c_type,
                legend=alt.Legend(symbolStrokeWidth=0, symbolOpacity=1),
            ),
        }
        return alt.Chart(df).mark_circle().encode(**encodings).properties(width=450, height=300)


    def vis_outlier_fact(self, base, subject, x, y, name):
        if base == None:
            return None
        if subject.get("subspace_pair", None) is None:
            outlier_cond = {
                "and": [
                    alt.datum[subject["breakdown"]] == name,
                    alt.datum[subject["measure"]] == x,
                    alt.datum[subject["measure2"]] == y,
                ]
            }
            spec = alt.layer(
                base.encode(opacity=alt.value(0.1)),
                base.mark_circle(opacity=1, filled=True, color="red").transform_filter(outlier_cond),
                base.mark_text(dx=5, align="left", color="red", fontWeight="bold", text=name).transform_filter(outlier_cond),
            )
        else:
            k, v = subject["subspace_pair"]
            outlier_cond = {
                "and": [
                    alt.datum[k] == v,
                    alt.datum[subject["breakdown"]] == name,
                    alt.datum[subject["measure"]] == x,
                    alt.datum[subject["measure2"]] == y,
                ]
            }
            spec = alt.layer(
                base.encode(opacity=alt.value(0.1)),
                base.mark_circle(opacity=1).transform_filter(alt.datum[k] == v),
                base.mark_text(dx=5, align="left", color="red", fontWeight="bold", text=name).transform_filter(outlier_cond),
            )
        return spec.to_json()

    def vis_difference_fact(self, base, subject, x):
        raise AssertionError("Scatter plot does not support difference fact")
    
    def vis_proportion_fact(self, base, subject, x):
        raise AssertionError("Scatter plot does not support proportion fact")
    
    def vis_rank_fact(self, base, subject, x):
        raise AssertionError("Scatter plot does not support rank fact")
    
    def vis_extremum_fact(self, base, subject, x):
        raise AssertionError("Scatter plot does not support extremum fact")
    
    def vis_trend_fact(self, base, subject, x):
        raise AssertionError("Scatter plot does not support trend fact")
    
    def vis_association_fact(self, base, subject, correlation):
        if base == None:
            return None
        if subject.get("subspace_pair", None) is None:
            return alt.layer(
                base,
                base.transform_regression(subject["measure"], subject["measure2"]).mark_line(color="red"),
                base.transform_regression(subject["measure"], subject["measure2"]).mark_text(
                    dx=5, align="left", color="red", fontWeight="bold", text=f"ρ = {correlation:.2f}"
                ).transform_filter(alt.datum[subject["measure"]] > base.data[subject["measure"]].quantile(0.5)),
            ).to_json()
        else:
            k, v = subject["subspace_pair"]
            series_base = self.vis_base_chart(base.data.loc[base.data[k] == v], subject)
            return alt.layer(
                base.encode(opacity=alt.value(0.1)),
                series_base,
                series_base.transform_regression(subject["measure"], subject["measure2"]).mark_line(color="red").encode(color=alt.Color(legend=None)),
                series_base.transform_regression(subject["measure"], subject["measure2"]).mark_text(
                    dx=5, align="left", color="red", fontWeight="bold", text=f"ρ = {correlation:.2f}"
                ).encode(color=alt.Color(legend=None)).transform_filter(alt.datum[subject["measure"]] > series_base.data[subject["measure"]].quantile(0.5)),
            ).to_json()
        
    def vis_value_fact(self, base, subject, x):
        raise AssertionError("Scatter plot does not support value fact")
