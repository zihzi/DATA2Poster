import altair as alt
from .fact_visualizer import FactVisualizer
from .utils import month_patch

angle = {
    "increasing": 315,
    "decreasing": 45,
    "flat": 0,
    "wavering": 0,
}
dx = {
    "increasing": 120,
    "decreasing": -40,
    "flat": 0,
    "wavering": 0,
}
dy = {
    "increasing": -40,
    "decreasing": -40,
    "flat": -80,
    "wavering": -80,
}
align = {
    "increasing": "right",
    "decreasing": "left",
    "flat": "left",
    "wavering": "left",
}
arrow = {
    "increasing": "-->",
    "decreasing": "-->",
    "flat": "-->",
    "wavering": "~~>",
}


class Bar(FactVisualizer):
    def vis_base_chart(self, df, subject, x_type, y_type, c_type, orient) -> alt.Chart:
        offset_encodings = {"yOffset": c_type} if orient == "horizontal" else {"xOffset": c_type}
        if isinstance(c_type, str):
            c_type = alt.Color(c_type)
        encodings = {
            "x": x_type,
            "y": y_type,
        } if subject["series"] is None else {
            "color": c_type.legend(symbolStrokeWidth=0, symbolOpacity=1),
            "x": x_type,
            "y": y_type,
            **offset_encodings
        }
        return alt.Chart(df).mark_bar().encode(**encodings).properties(width=450, height=300)


    def vis_difference_fact(self, base, subject, x1, x2):
        if base == None:
            return None
        if subject.get("subspace_pair", None) is None:
            spec = alt.layer(
                base.encode(opacity=alt.value(0.3)),
                base.mark_bar(stroke="red", strokeWidth=3).transform_filter(
                    {
                        "or": [
                            alt.datum[subject["breakdown"]] == x1,
                            alt.datum[subject["breakdown"]] == x2,
                        ]
                    }
                ),
            )
        else:
            k, v = subject["subspace_pair"]
            spec = alt.layer(
                base.encode(opacity=alt.value(0.3)),
                base.mark_bar(opacity=1).transform_filter(alt.datum[k] == v),
                base.mark_bar(stroke="red", strokeWidth=3).transform_filter(
                    {
                        "and": [
                            alt.datum[k] == v,
                            {
                                "or": [
                                    alt.datum[subject["breakdown"]] == x1,
                                    alt.datum[subject["breakdown"]] == x2,
                                ]
                            },
                        ]
                    }
                ),
            )
        return spec.to_json()


    def vis_proportion_fact(self, base, subject, x):
        if base == None:
            return None
        if subject.get("subspace_pair", None) is None:
            spec = alt.layer(
                base.encode(opacity=alt.value(0.3)),
                base.mark_bar(stroke="red", strokeWidth=3).transform_filter(
                    alt.datum[subject["breakdown"]] == x
                ),
            )
        else:
            k, v = subject["subspace_pair"]
            spec = alt.layer(
                base.encode(opacity=alt.value(0.3)),
                base.mark_bar(opacity=1).transform_filter(alt.datum[k] == v),
                base.mark_bar(stroke="red", strokeWidth=3).transform_filter(
                    {
                        "and": [
                            alt.datum[k] == v,
                            alt.datum[subject["breakdown"]] == x,
                        ]
                    }
                ),
            )
        return spec.to_json()


    def vis_rank_fact(self, base, subject, x1, x2, x3):
        if base == None:
            return None
        if subject.get("subspace_pair", None) is None:
            spec = alt.layer(
                base.encode(opacity=alt.value(0.3)),
                base.mark_bar(stroke="red", strokeWidth=3).transform_filter(
                    {
                        "or": [
                            alt.datum[subject["breakdown"]] == x1,
                            alt.datum[subject["breakdown"]] == x2,
                            alt.datum[subject["breakdown"]] == x3,
                        ]
                    }
                ),
            )
        else:
            k, v = subject["subspace_pair"]
            spec = alt.layer(
                base.encode(opacity=alt.value(0.3)),
                base.mark_bar(opacity=1).transform_filter(alt.datum[k] == v),
                base.mark_bar(stroke="red", strokeWidth=3).transform_filter(
                    {
                        "and": [
                            alt.datum[k] == v,
                            {
                                "or": [
                                    alt.datum[subject["breakdown"]] == x1,
                                    alt.datum[subject["breakdown"]] == x2,
                                    alt.datum[subject["breakdown"]] == x3,
                                ]
                            },
                        ]
                    }
                ),
            )
        return spec.to_json()


    def vis_extremum_fact(self, base, subject, x):
        if base == None:
            return None
        if subject.get("subspace_pair", None) is None:
            spec = alt.layer(
                base.encode(opacity=alt.value(0.3)),
                base.mark_bar(stroke="red", strokeWidth=3).transform_filter(
                    alt.datum[subject["breakdown"]] == x
                ),
            )
        else:
            k, v = subject["subspace_pair"]
            spec = alt.layer(
                base.encode(opacity=alt.value(0.3)),
                base.mark_bar(opacity=1).transform_filter(alt.datum[k] == v),
                base.mark_bar(stroke="red", strokeWidth=3).transform_filter(
                    {
                        "and": [
                            alt.datum[k] == v,
                            alt.datum[subject["breakdown"]] == x,
                        ]
                    }
                ),
            )
        return spec.to_json()


    def vis_outlier_fact(self, base, subject, x):
        if base == None:
            return None
        if subject.get("subspace_pair", None) is None:
            spec = alt.layer(
                base.encode(opacity=alt.value(0.3)),
                base.mark_bar(stroke="red", strokeWidth=3).transform_filter(
                    alt.datum[subject["breakdown"]] == x
                ),
            )
        else:
            k, v = subject["subspace_pair"]
            spec = alt.layer(
                base.encode(opacity=alt.value(0.3)),
                base.mark_bar(opacity=1).transform_filter(alt.datum[k] == v),
                base.mark_bar(stroke="red", strokeWidth=3).transform_filter(
                    {
                        "and": [
                            alt.datum[k] == v,
                            alt.datum[subject["breakdown"]] == x,
                        ]
                    }
                ),
            )
        return spec.to_json()

    
    def vis_trend_fact(self, base, subject, x1, x2, trend, span):
        if base == None:
            return None
        arrowEl = base.mark_text(
            dx=dx[trend],
            dy=dy[trend],
            align=align[trend],
            baseline="bottom",
            fontSize=36,
            font="Fira Code",
            angle=angle[trend],
        ).encode(
            text=alt.value(arrow[trend]),
            color=alt.value("red"),
        )
        if subject.get("subspace_pair", None) is None:
            if span == 1:
                spec = alt.layer(
                    base,
                    arrowEl.transform_filter(
                        {
                            "and": [
                                alt.datum[subject["breakdown"]] == x1,
                            ]
                        }
                    ),
                )
            else:
                spec = alt.layer(
                    base.encode(opacity=alt.value(0.3)),
                    base.mark_bar(stroke="red", strokeWidth=3).transform_filter(
                        {
                            "and": [
                                alt.datum[subject["breakdown"]] >= x1,
                                alt.datum[subject["breakdown"]] <= x2,
                            ] if not isinstance(x1, str) else
                                [
                                    alt.datum["month_number"] >= month_patch(x1),
                                    alt.datum["month_number"] <= month_patch(x2),
                                ]
                        }
                    ),
                    arrowEl.transform_filter(
                        {
                            "and": [
                                alt.datum[subject["breakdown"]] == x1,
                            ]
                        }
                    ),
                )
        else:
            k, v = subject["subspace_pair"]
            spec = alt.layer(
                base.encode(opacity=alt.value(0.3)),
                base.mark_bar(opacity=1).transform_filter(alt.datum[k] == v),
                base.mark_bar(stroke="red", strokeWidth=3).transform_filter(
                    {
                        "and": [
                            alt.datum[k] == v,
                            alt.datum[subject["breakdown"]] >= x1,
                            alt.datum[subject["breakdown"]] <= x2,
                        ] if not isinstance(x1, str) else
                            [
                                alt.datum[k] == v,
                                alt.datum["month_number"] >= month_patch(x1),
                                alt.datum["month_number"] <= month_patch(x2),
                            ]
                    }
                ),
                arrowEl.transform_filter(
                    {
                        "and": [
                            alt.datum[k] == v,
                            alt.datum[subject["breakdown"]] == x1,
                        ]
                    }
                ),
            )
        return spec.to_json()
    

    def vis_association_fact(self, base, subject, x):
        raise AssertionError("Bar plot does not support association fact")
    
    def vis_value_fact(self, base, subject, x) -> str:
        if base == None:
            return None
        if subject.get("subspace_pair", None) is None:
            spec = alt.layer(
                base.encode(opacity=alt.value(0.3)),
                base.mark_bar(stroke="red", strokeWidth=3).transform_filter(
                    alt.datum[subject["breakdown"]] == x
                ),
            )
        else:
            k, v = subject["subspace_pair"]
            spec = alt.layer(
                base.encode(opacity=alt.value(0.3)),
                base.mark_bar(stroke="red", strokeWidth=3).transform_filter(
                    {
                        "and": [
                            alt.datum[k] == v,
                            alt.datum[subject["breakdown"]] == x,
                        ]
                    }
                ),
            )
        return spec.to_json()
    