import altair as alt
from .bar import Bar
from .line import Line
from .scatter import Scatter
from .radial import Radial
from .choropleth import Choropleth
from .fact_visualizer import FactVisualizer

STRATEGIES = {
    "bar": Bar,
    "line": Line,
    "scatter": Scatter,
    "pie": Radial,
    "map": Choropleth
}

class Visualizer:
    def __init__(self, df, subject, x_type, y_type, c_type, orient):
        self._df = df
        self._subject = subject
        self._x_type = x_type
        self._y_type = y_type
        self._c_type = c_type
        self._strategy = self._get_strategy(subject["visualization"])
        self._base_chart = self._strategy.vis_base_chart(df, subject, x_type, y_type, c_type, orient)
            
    def _get_strategy(self, chart_type) -> FactVisualizer:
        assert chart_type in STRATEGIES, "Unsupported chart type"
        return STRATEGIES[chart_type]()
    
    def get_chart_type(self):
        return self._subject["visualization"]

    def get_base_chart(self, json=True) -> alt.Chart:
        if not json:
            return self._base_chart
        return self._base_chart.to_json()
    
    def get_fact_visualized_chart(self, fact_type, *args):
        base = self.get_base_chart(json=False)
        if fact_type == "difference":
            return self._strategy.vis_difference_fact(base, *args)
        elif fact_type == "proportion":
            return self._strategy.vis_proportion_fact(base, *args)
        elif fact_type == "rank":
            return self._strategy.vis_rank_fact(base, *args)
        elif fact_type == "extremum":
            return self._strategy.vis_extremum_fact(base, *args)
        elif fact_type == "outlier":
            return self._strategy.vis_outlier_fact(base, *args)
        elif fact_type == "trend":
            return self._strategy.vis_trend_fact(base, *args)
        elif fact_type == "association":
            return self._strategy.vis_association_fact(base, *args)
        elif fact_type == "value":
            return self._strategy.vis_value_fact(base, *args)
        else:
            raise ValueError(f"Unsupported fact type: {fact_type}")
