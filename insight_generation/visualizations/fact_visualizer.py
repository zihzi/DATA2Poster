import altair as alt

from abc import ABC, abstractmethod

class FactVisualizer(ABC):
    @abstractmethod
    def vis_base_chart(self, df, subject, x_type, y_type, c_type, orient) -> alt.Chart:
        raise NotImplementedError
    
    @abstractmethod
    def vis_difference_fact(self, base_chart, *args) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def vis_proportion_fact(self, base_chart, *args) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def vis_rank_fact(self, base_chart, *args) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def vis_extremum_fact(self, base_chart, *args) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def vis_outlier_fact(self, base_chart, *args) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def vis_trend_fact(self, base_chart, *args) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def vis_association_fact(self, base_chart, *args) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def vis_value_fact(self, base_chart, *args) -> str:
        raise NotImplementedError