import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import kendalltau
from scipy.signal import find_peaks

class FactTypeGenerator:
    def __init__(self, df):
        """Initialize the fact type generator with a DataFrame."""
        self.df = df

    def value_C(self, subspace, measure, agg, str):
        """Calculate a derived value."""
        if agg == "sum":
            value = self.df[measure].sum()
        elif agg == "mean":
            value = self.df[measure].mean()
        elif agg == "max":
            value = self.df[measure].max()
        elif agg == "min":
            value = self.df[measure].min()     
        return f"The {agg} of '{measure}' is {value} when '{subspace}' is '{str}'."
    
    def value_N(self, subspace, measure, agg, str):
        """Calculate a derived value."""
        if agg == "sum":
            value = self.df[measure].sum()
        elif agg == "mean":
            value = self.df[measure].mean()
        elif agg == "max":
            value = self.df[measure].max()
        elif agg == "min":
            value = self.df[measure].min()   
        return f"The {agg} of '{measure}' is {value} when '{subspace}' is {str} than mean."

    def difference_C(self, subspace, breakdown, measure, agg, str):
        """Calculate the difference between focus values in a breakdown."""
        if agg == "sum":
            grouped = self.df.groupby(breakdown)[measure].sum()     
            # Convert the series to a NumPy array for processing
            values = grouped.values
            intervals = grouped.index
            for i in range(len(values) - 1):
                v1 = values[i]
                v2 = values[i + 1]
                x1 = intervals[i]
                x2 = intervals[i + 1]
                if v1 > 0 and v2 > 0 and max(v1/v2, v2/v1) >= 1.5:
                    ratio = max(v1/v2, v2/v1)
                    if v1 > v2:
                        return f"The {agg} '{measure}' of '{breakdown}'={x1} is {ratio:.2f} times more than that of '{breakdown}'={x2} when '{subspace}' is '{str}'."
                    else:
                        return f"The {agg} '{measure}' of '{breakdown}'={x1} is {ratio:.2f} times less than that of '{breakdown}'={x2} when '{subspace}' is '{str}'."
                else:
                    return f"No significant difference."
        elif agg == "mean":
            grouped = self.df.groupby(breakdown)[measure].mean()     
            # Convert the series to a NumPy array for processing
            values = grouped.values
            intervals = grouped.index
            for i in range(len(values) - 1):
                v1 = values[i]
                v2 = values[i + 1]
                x1 = intervals[i]
                x2 = intervals[i + 1]
                if v1 > 0 and v2 > 0 and max(v1/v2, v2/v1) >= 1.5:
                    ratio = max(v1/v2, v2/v1)
                    if v1 > v2:
                        return f"The {agg} '{measure}' of '{breakdown}'={x1} is {ratio:.2f} times more than that of '{breakdown}'={x2} when '{subspace}' is '{str}'."
                    else:
                        return f"The {agg} '{measure}' of '{breakdown}'={x1} is {ratio:.2f} times less than that of '{breakdown}'={x2} when '{subspace}' is '{str}'."
                else:
                    return f"No significant difference."
        elif agg == "max":
            grouped = self.df.groupby(breakdown)[measure].max()     
            # Convert the series to a NumPy array for processing
            values = grouped.values
            intervals = grouped.index
            for i in range(len(values) - 1):
                v1 = values[i]
                v2 = values[i + 1]
                x1 = intervals[i]
                x2 = intervals[i + 1]
                if v1 > 0 and v2 > 0 and max(v1/v2, v2/v1) >= 1.5:
                    ratio = max(v1/v2, v2/v1)
                    if v1 > v2:
                        return f"The {agg} '{measure}' of '{breakdown}'={x1} is {ratio:.2f} times more than that of '{breakdown}'={x2} when '{subspace}' is '{str}'."
                    else:
                        return f"The {agg} '{measure}' of '{breakdown}'={x1} is {ratio:.2f} times less than that of '{breakdown}'={x2} when '{subspace}' is '{str}'."
                else:
                    return f"No significant difference."
        elif agg == "min": 
            grouped = self.df.groupby(breakdown)[measure].min()     
            # Convert the series to a NumPy array for processing
            values = grouped.values
            intervals = grouped.index
            for i in range(len(values) - 1):
                v1 = values[i]
                v2 = values[i + 1]
                x1 = intervals[i]
                x2 = intervals[i + 1]
                if v1 > 0 and v2 > 0 and max(v1/v2, v2/v1) >= 1.5:
                    ratio = max(v1/v2, v2/v1)
                    if v1 > v2:
                        return f"The {agg} '{measure}' of '{breakdown}'={x1} is {ratio:.2f} times more than that of '{breakdown}'={x2} when '{subspace}' is '{str}'."
                    else:
                        return f"The {agg} '{measure}' of '{breakdown}'={x1} is {ratio:.2f} times less than that of '{breakdown}'={x2} when '{subspace}' is '{str}'."
                else:
                    return f"No significant difference."
                
    def difference_N(self, subspace, breakdown, measure, agg, str):
        """Calculate the difference between focus values in a breakdown."""
        if agg == "sum":
            grouped = self.df.groupby(breakdown)[measure].sum()     
            # Convert the series to a NumPy array for processing
            values = grouped.values
            intervals = grouped.index
            for i in range(len(values) - 1):
                v1 = values[i]
                v2 = values[i + 1]
                x1 = intervals[i]
                x2 = intervals[i + 1]
                if v1 > 0 and v2 > 0 and max(v1/v2, v2/v1) >= 1.5:
                    ratio = max(v1/v2, v2/v1)
                    if v1 > v2:
                        return f"The {agg} '{measure}' of '{breakdown}'={x1} is {ratio:.2f} times more than that of '{breakdown}'={x2} when '{subspace}' is {str} than mean."
                    else:
                        return f"The {agg} '{measure}' of '{breakdown}'={x1} is {ratio:.2f} times less than that of '{breakdown}'={x2} when '{subspace}' is {str} than mean."
                else:
                    return f"No significant difference."
        elif agg == "mean":
            grouped = self.df.groupby(breakdown)[measure].mean()     
            # Convert the series to a NumPy array for processing
            values = grouped.values
            intervals = grouped.index
            for i in range(len(values) - 1):
                v1 = values[i]
                v2 = values[i + 1]
                x1 = intervals[i]
                x2 = intervals[i + 1]
                if v1 > 0 and v2 > 0 and max(v1/v2, v2/v1) >= 1.5:
                    ratio = max(v1/v2, v2/v1)
                    if v1 > v2:
                        return f"The {agg} '{measure}' of '{breakdown}'={x1} is {ratio:.2f} times more than that of '{breakdown}'={x2} when '{subspace}' is {str} than mean."
                    else:
                        return f"The {agg} '{measure}' of '{breakdown}'={x1} is {ratio:.2f} times less than that of '{breakdown}'={x2} when '{subspace}' is {str} than mean."
                else:
                    return f"No significant difference."
        elif agg == "max":
            grouped = self.df.groupby(breakdown)[measure].max()     
            # Convert the series to a NumPy array for processing
            values = grouped.values
            intervals = grouped.index
            for i in range(len(values) - 1):
                v1 = values[i]
                v2 = values[i + 1]
                x1 = intervals[i]
                x2 = intervals[i + 1]
                if v1 > 0 and v2 > 0 and max(v1/v2, v2/v1) >= 1.5:
                    ratio = max(v1/v2, v2/v1)
                    if v1 > v2:
                        return f"The {agg} '{measure}' of '{breakdown}'={x1} is {ratio:.2f} times more than that of '{breakdown}'={x2} when '{subspace}' is {str} than mean."
                    else:
                        return f"The {agg} '{measure}' of '{breakdown}'={x1} is {ratio:.2f} times less than that of '{breakdown}'={x2} when '{subspace}' is {str} than mean."
                else:
                    return f"No significant difference."
        elif agg == "min": 
            grouped = self.df.groupby(breakdown)[measure].min()     
            # Convert the series to a NumPy array for processing
            values = grouped.values
            intervals = grouped.index
            for i in range(len(values) - 1):
                v1 = values[i]
                v2 = values[i + 1]
                x1 = intervals[i]
                x2 = intervals[i + 1]
                if v1 > 0 and v2 > 0 and max(v1/v2, v2/v1) >= 1.5:
                    ratio = max(v1/v2, v2/v1)
                    if v1 > v2:
                        return f"The {agg} '{measure}' of '{breakdown}'={x1} is {ratio:.2f} times more than that of '{breakdown}'={x2} when '{subspace}' is {str} than mean."
                    else:
                        return f"The {agg} '{measure}' of '{breakdown}'={x1} is {ratio:.2f} times less than that of '{breakdown}'={x2} when '{subspace}' is {str} than mean."
                else:
                    return f"No significant difference."
    
    def proportion_C(self, subspace, breakdown, measure, agg, str):
        """Calculate the proportion for a specific focus."""
        # May return NULL, need to eliminate
        if agg == "sum":
            grouped = self.df.groupby(breakdown)[measure].sum()
            total = grouped.sum()
            # Convert the series to a NumPy array for processing
            values = grouped.values
            intervals = grouped.index
            for i in range(len(values)):
                proportion = values[i] / total if total != 0 else 0
                if proportion >= 0.5:
                    return f"The '{breakdown}'={intervals[i]} accounts for {proportion:.2%} of the {agg} '{measure}' when '{subspace}' is '{str}'."
        elif agg == "mean":
            grouped = self.df.groupby(breakdown)[measure].mean()
            total = grouped.sum()
            # Convert the series to a NumPy array for processing
            values = grouped.values
            intervals = grouped.index
            for i in range(len(values)):
                proportion = values[i] / total if total != 0 else 0
                if proportion >= 0.5:
                    return f"The '{breakdown}'={intervals[i]} accounts for {proportion:.2%} of the {agg} '{measure}' when '{subspace}' is '{str}'."
        elif agg == "max":
            grouped = self.df.groupby(breakdown)[measure].max()
            total = grouped.sum()
            # Convert the series to a NumPy array for processing
            values = grouped.values
            intervals = grouped.index
            for i in range(len(values)):
                proportion = values[i] / total if total != 0 else 0
                if proportion >= 0.5:
                    return f"The '{breakdown}'={intervals[i]} accounts for {proportion:.2%} of the {agg} '{measure}' when '{subspace}' is '{str}'."
        elif agg == "min":
            grouped = self.df.groupby(breakdown)[measure].min()
            total = grouped.sum()
            # Convert the series to a NumPy array for processing
            values = grouped.values
            intervals = grouped.index
            for i in range(len(values)):
                proportion = values[i] / total if total != 0 else 0
                if proportion >= 0.5:
                    return f"The '{breakdown}'={intervals[i]} accounts for {proportion:.2%} of the {agg} '{measure}' when '{subspace}' is '{str}'."

    def proportion_N(self, subspace, breakdown, measure, agg, str):
        """Calculate the proportion for a specific focus."""
        # May return NULL, need to eliminate
        if agg == "sum":
            grouped = self.df.groupby(breakdown)[measure].sum()
            total = grouped.sum()
            # Convert the series to a NumPy array for processing
            values = grouped.values
            intervals = grouped.index
            for i in range(len(values)):
                proportion = values[i] / total if total != 0 else 0
                if proportion >= 0.5:
                    return f"The '{breakdown}'={intervals[i]} accounts for {proportion:.2%} of the {agg} '{measure}' when '{subspace}' is {str} than mean."
        elif agg == "mean":
            grouped = self.df.groupby(breakdown)[measure].mean()
            total = grouped.sum()
            # Convert the series to a NumPy array for processing
            values = grouped.values
            intervals = grouped.index
            for i in range(len(values)):
                proportion = values[i] / total if total != 0 else 0
                if proportion >= 0.5:
                    return f"The '{breakdown}'={intervals[i]} accounts for {proportion:.2%} of the {agg} '{measure}' when '{subspace}' is {str} than mean."
        elif agg == "max":
            grouped = self.df.groupby(breakdown)[measure].max()
            total = grouped.sum()
            # Convert the series to a NumPy array for processing
            values = grouped.values
            intervals = grouped.index
            for i in range(len(values)):
                proportion = values[i] / total if total != 0 else 0
                if proportion >= 0.5:
                    return f"The '{breakdown}'={intervals[i]} accounts for {proportion:.2%} of the {agg} '{measure}' when '{subspace}' is {str} than mean."
        elif agg == "min":
            grouped = self.df.groupby(breakdown)[measure].min()
            total = grouped.sum()
            # Convert the series to a NumPy array for processing
            values = grouped.values
            intervals = grouped.index
            for i in range(len(values)):
                proportion = values[i] / total if total != 0 else 0
                if proportion >= 0.5:
                    return f"The '{breakdown}'={intervals[i]} accounts for {proportion:.2%} of the {agg} '{measure}' when '{subspace}' is {str} than mean."

    def overall_trend_C(self, subspace, breakdown, measure, agg, str):
        """Calculate a trend (increasing or decreasing)."""
        if agg == "sum":
            grouped = self.df.groupby(breakdown)[measure].sum()
            # Convert the series to a NumPy array for processing
            values = grouped.values
            intervals = grouped.index

            # Find peaks and valleys
            peaks, _ = find_peaks(values)  # Peaks
            valleys, _ = find_peaks(-values)  # Valleys (inverted peaks)

            # Combine peaks and valleys into a single list with their indices
            extrema_indices = np.sort(np.concatenate((peaks, valleys)))

            # Extract the values of peaks and valleys for comparison
            extrema_values = values[extrema_indices]

            # Determine overall trend
            if len(extrema_values) < 2:
                return "No clear trend."  # Not enough extrema to determine trend

            if np.all(np.diff(extrema_values) > 0):
                return f"The increasing trend of {agg} '{measure}' over '{breakdown}' when '{subspace}' is '{str}'."
            elif np.all(np.diff(extrema_values) < 0):
                return f"The decreasing trend of {agg} '{measure}' over '{breakdown}' when '{subspace}' is '{str}'."
            else:
                return f"The wavering trend of {agg} '{measure}' over '{breakdown}' when '{subspace}' is '{str}'."
        elif agg == "mean":
            grouped = self.df.groupby(breakdown)[measure].mean()
            # Convert the series to a NumPy array for processing
            values = grouped.values
            intervals = grouped.index

            # Find peaks and valleys
            peaks, _ = find_peaks(values)  # Peaks
            valleys, _ = find_peaks(-values)  # Valleys (inverted peaks)

            # Combine peaks and valleys into a single list with their indices
            extrema_indices = np.sort(np.concatenate((peaks, valleys)))

            # Extract the values of peaks and valleys for comparison
            extrema_values = values[extrema_indices]

            # Determine overall trend
            if len(extrema_values) < 2:
                return "No clear trend."  # Not enough extrema to determine trend

            if np.all(np.diff(extrema_values) > 0):
                return f"The increasing trend of {agg} '{measure}' over '{breakdown}' when '{subspace}' is '{str}'."
            elif np.all(np.diff(extrema_values) < 0):
                return f"The decreasing trend of {agg} '{measure}' over '{breakdown}' when '{subspace}' is '{str}'."
            else:
                return f"The wavering trend of {agg} '{measure}' over '{breakdown}' when '{subspace}' is '{str}'."
        elif agg == "max":
            grouped = self.df.groupby(breakdown)[measure].max()
            # Convert the series to a NumPy array for processing
            values = grouped.values
            intervals = grouped.index

            # Find peaks and valleys
            peaks, _ = find_peaks(values)  # Peaks
            valleys, _ = find_peaks(-values)  # Valleys (inverted peaks)

            # Combine peaks and valleys into a single list with their indices
            extrema_indices = np.sort(np.concatenate((peaks, valleys)))

            # Extract the values of peaks and valleys for comparison
            extrema_values = values[extrema_indices]

            # Determine overall trend
            if len(extrema_values) < 2:
                return "No clear trend."  # Not enough extrema to determine trend

            if np.all(np.diff(extrema_values) > 0):
                return f"The increasing trend of {agg} '{measure}' over '{breakdown}' when '{subspace}' is '{str}'."
            elif np.all(np.diff(extrema_values) < 0):
                return f"The decreasing trend of {agg} '{measure}' over '{breakdown}' when '{subspace}' is '{str}'."
            else:
                return f"The wavering trend of {agg} '{measure}' over '{breakdown}' when '{subspace}' is '{str}'."
       
        elif agg == "min":
            grouped = self.df.groupby(breakdown)[measure].min()
            # Convert the series to a NumPy array for processing
            values = grouped.values
            intervals = grouped.index

            # Find peaks and valleys
            peaks, _ = find_peaks(values)  # Peaks
            valleys, _ = find_peaks(-values)  # Valleys (inverted peaks)

            # Combine peaks and valleys into a single list with their indices
            extrema_indices = np.sort(np.concatenate((peaks, valleys)))

            # Extract the values of peaks and valleys for comparison
            extrema_values = values[extrema_indices]

            # Determine overall trend
            if len(extrema_values) < 2:
                return "No clear trend."  # Not enough extrema to determine trend

            if np.all(np.diff(extrema_values) > 0):
                return f"The increasing trend of {agg} '{measure}' over '{breakdown}' when '{subspace}' is '{str}'."
            elif np.all(np.diff(extrema_values) < 0):
                return f"The decreasing trend of {agg} '{measure}' over '{breakdown}' when '{subspace}' is '{str}'."
            else:
                return f"The wavering trend of {agg} '{measure}' over '{breakdown}' when '{subspace}' is '{str}'."
 
    def overall_trend_N(self, subspace, breakdown, measure, agg, str):
        """Calculate a trend (increasing or decreasing)."""
        if agg == "sum":
            grouped = self.df.groupby(breakdown)[measure].sum()
            # Convert the series to a NumPy array for processing
            values = grouped.values
            intervals = grouped.index

            # Find peaks and valleys
            peaks, _ = find_peaks(values)  # Peaks
            valleys, _ = find_peaks(-values)  # Valleys (inverted peaks)

            # Combine peaks and valleys into a single list with their indices
            extrema_indices = np.sort(np.concatenate((peaks, valleys)))

            # Extract the values of peaks and valleys for comparison
            extrema_values = values[extrema_indices]

            # Determine overall trend
            if len(extrema_values) < 2:
                return "No clear trend."  # Not enough extrema to determine trend

            if np.all(np.diff(extrema_values) > 0):
                return f"The increasing trend of {agg} '{measure}' over '{breakdown}' when '{subspace}' is {str} than mean."
            elif np.all(np.diff(extrema_values) < 0):
                return f"The decreasing trend of {agg} '{measure}' over '{breakdown}' when '{subspace}' is {str} than mean."
            else:
                return f"The wavering trend of {agg} '{measure}' over '{breakdown}' when '{subspace}' is {str} than mean."
        elif agg == "mean":
            grouped = self.df.groupby(breakdown)[measure].mean()
            # Convert the series to a NumPy array for processing
            values = grouped.values
            intervals = grouped.index

            # Find peaks and valleys
            peaks, _ = find_peaks(values)  # Peaks
            valleys, _ = find_peaks(-values)  # Valleys (inverted peaks)

            # Combine peaks and valleys into a single list with their indices
            extrema_indices = np.sort(np.concatenate((peaks, valleys)))

            # Extract the values of peaks and valleys for comparison
            extrema_values = values[extrema_indices]

            # Determine overall trend
            if len(extrema_values) < 2:
                return "No clear trend."  # Not enough extrema to determine trend

            if np.all(np.diff(extrema_values) > 0):
                return f"The increasing trend of {agg} '{measure}' over '{breakdown}' when '{subspace}' is {str} than mean."
            elif np.all(np.diff(extrema_values) < 0):
                return f"The decreasing trend of {agg} '{measure}' over '{breakdown}' when '{subspace}' is {str} than mean."
            else:
                return f"The wavering trend of {agg} '{measure}' over '{breakdown}' when '{subspace}' is {str} than mean."
        elif agg == "max":
            grouped = self.df.groupby(breakdown)[measure].max()
            # Convert the series to a NumPy array for processing
            values = grouped.values
            intervals = grouped.index

            # Find peaks and valleys
            peaks, _ = find_peaks(values)  # Peaks
            valleys, _ = find_peaks(-values)  # Valleys (inverted peaks)

            # Combine peaks and valleys into a single list with their indices
            extrema_indices = np.sort(np.concatenate((peaks, valleys)))

            # Extract the values of peaks and valleys for comparison
            extrema_values = values[extrema_indices]

            # Determine overall trend
            if len(extrema_values) < 2:
                return "No clear trend."  # Not enough extrema to determine trend

            if np.all(np.diff(extrema_values) > 0):
                return f"The increasing trend of {agg} '{measure}' over '{breakdown}' when '{subspace}' is {str} than mean."
            elif np.all(np.diff(extrema_values) < 0):
                return f"The decreasing trend of {agg} '{measure}' over '{breakdown}' when '{subspace}' is {str} than mean."
            else:
                return f"The wavering trend of {agg} '{measure}' over '{breakdown}' when '{subspace}' is {str} than mean."
       
        elif agg == "min":
            grouped = self.df.groupby(breakdown)[measure].min()
            # Convert the series to a NumPy array for processing
            values = grouped.values
            intervals = grouped.index

            # Find peaks and valleys
            peaks, _ = find_peaks(values)  # Peaks
            valleys, _ = find_peaks(-values)  # Valleys (inverted peaks)

            # Combine peaks and valleys into a single list with their indices
            extrema_indices = np.sort(np.concatenate((peaks, valleys)))

            # Extract the values of peaks and valleys for comparison
            extrema_values = values[extrema_indices]

            # Determine overall trend
            if len(extrema_values) < 2:
                return "No clear trend."  # Not enough extrema to determine trend

            if np.all(np.diff(extrema_values) > 0):
                return f"The increasing trend of {agg} '{measure}' over '{breakdown}' when '{subspace}' is {str} than mean."
            elif np.all(np.diff(extrema_values) < 0):
                return f"The decreasing trend of {agg} '{measure}' over '{breakdown}' when '{subspace}' is {str} than mean."
            else:
                return f"The wavering trend of {agg} '{measure}' over '{breakdown}' when '{subspace}' is {str} than mean."
    
    def segment_trend_C(self, subspace, breakdown, measure, agg, str):  
        trend = []
        if agg == "sum":
            grouped = self.df.groupby(breakdown)[measure].sum()
            # Convert the series to a NumPy array for processing
            values = grouped.values
            intervals = grouped.index
            for i in range(len(values) - 1):
                # Apply Kendall's Tau to check statistical significance
                tau, p_value = kendalltau([values[i]], [values[i + 1]])

                # Check p-value for significance (commonly 0.05)
                if p_value <= 0.05:
                    if tau > 0:
                        trend.append(f"The increasing trend of {agg} '{measure}' over '{breakdown}' when '{subspace}' is '{str}' from {intervals[i]} to {intervals[i + 1]}.")
                    elif tau < 0:
                        trend.append(f"The decreasing trend of {agg} '{measure}' over '{breakdown}' when '{subspace}' is '{str}' from {intervals[i]} to {intervals[i + 1]}.")
            if trend != []:
                return trend
        elif agg == "mean":
            grouped = self.df.groupby(breakdown)[measure].mean()
            # Convert the series to a NumPy array for processing
            values = grouped.values
            intervals = grouped.index
            for i in range(len(values) - 1):
                # Apply Kendall's Tau to check statistical significance
                tau, p_value = kendalltau([values[i]], [values[i + 1]])

                # Check p-value for significance (commonly 0.05)
                if p_value <= 0.05:
                    if tau > 0:
                        trend.append(f"The increasing trend of {agg} '{measure}' over '{breakdown}' when '{subspace}' is '{str}' from {intervals[i]} to {intervals[i + 1]}.")
                    elif tau < 0:
                        trend.append(f"The decreasing trend of {agg} '{measure}' over '{breakdown}' when '{subspace}' is '{str}' from {intervals[i]} to {intervals[i + 1]}.")
            if trend != []:
                return trend
        elif agg == "max":
            grouped = self.df.groupby(breakdown)[measure].max()
            # Convert the series to a NumPy array for processing
            values = grouped.values
            intervals = grouped.index
            for i in range(len(values) - 1):
                # Apply Kendall's Tau to check statistical significance
                tau, p_value = kendalltau([values[i]], [values[i + 1]])

                # Check p-value for significance (commonly 0.05)
                if p_value <= 0.05:
                    if tau > 0:
                        trend.append(f"The increasing trend of {agg} '{measure}' over '{breakdown}' when '{subspace}' is '{str}' from {intervals[i]} to {intervals[i + 1]}.")
                    elif tau < 0:
                        trend.append(f"The decreasing trend of {agg} '{measure}' over '{breakdown}' when '{subspace}' is '{str}' from {intervals[i]} to {intervals[i + 1]}.")
            if trend != []:
                return trend
        elif agg == "min":
            grouped = self.df.groupby(breakdown)[measure].min()
            # Convert the series to a NumPy array for processing
            values = grouped.values
            intervals = grouped.index
            for i in range(len(values) - 1):
                # Apply Kendall's Tau to check statistical significance
                tau, p_value = kendalltau([values[i]], [values[i + 1]])

                # Check p-value for significance (commonly 0.05)
                if p_value <= 0.05:
                    if tau > 0:
                        trend.append(f"The increasing trend of {agg} '{measure}' over '{breakdown}' when '{subspace}' is '{str}' from {intervals[i]} to {intervals[i + 1]}.")
                    elif tau < 0:
                        trend.append(f"The decreasing trend of {agg} '{measure}' over '{breakdown}' when '{subspace}' is '{str}' from {intervals[i]} to {intervals[i + 1]}.")
            if trend != []:
                return trend
    
    def segment_trend_N(self, subspace, breakdown, measure, agg, str): 
        trend = []
        if agg == "sum":
            grouped = self.df.groupby(breakdown)[measure].sum()
            # Convert the series to a NumPy array for processing
            values = grouped.values
            intervals = grouped.index
            for i in range(len(values) - 1):
                # Apply Kendall's Tau to check statistical significance
                tau, p_value = kendalltau([values[i]], [values[i + 1]])

                # Check p-value for significance (commonly 0.05)
                if p_value <= 0.05:
                    if tau > 0:
                        trend.append(f"The increasing trend of {agg} '{measure}' over '{breakdown}' when '{subspace}' is {str} than mean from {intervals[i]} to {intervals[i + 1]}.")
                    elif tau < 0:
                        trend.append(f"The decreasing trend of {agg} '{measure}' over '{breakdown}' when '{subspace}' is {str} than mean from {intervals[i]} to {intervals[i + 1]}.")
            if trend != []:
                return trend
        elif agg == "mean":
            grouped = self.df.groupby(breakdown)[measure].mean()
            # Convert the series to a NumPy array for processing
            values = grouped.values
            intervals = grouped.index
            for i in range(len(values) - 1):
                # Apply Kendall's Tau to check statistical significance
                tau, p_value = kendalltau([values[i]], [values[i + 1]])

                # Check p-value for significance (commonly 0.05)
                if p_value <= 0.05:
                    if tau > 0:
                        trend.append(f"The increasing trend of {agg} '{measure}' over '{breakdown}' when '{subspace}' is {str} than mean from {intervals[i]} to {intervals[i + 1]}.")
                    elif tau < 0:
                        trend.append(f"The decreasing trend of {agg} '{measure}' over '{breakdown}' when '{subspace}' is {str} than mean from {intervals[i]} to {intervals[i + 1]}.")
            if trend != []:
                return trend
        elif agg == "max":
            grouped = self.df.groupby(breakdown)[measure].max()
            # Convert the series to a NumPy array for processing
            values = grouped.values
            intervals = grouped.index
            for i in range(len(values) - 1):
                # Apply Kendall's Tau to check statistical significance
                tau, p_value = kendalltau([values[i]], [values[i + 1]])

                # Check p-value for significance (commonly 0.05)
                if p_value <= 0.05:
                    if tau > 0:
                        trend.append(f"The increasing trend of {agg} '{measure}' over '{breakdown}' when '{subspace}' is {str} than mean from {intervals[i]} to {intervals[i + 1]}.")
                    elif tau < 0:
                        trend.append(f"The decreasing trend of {agg} '{measure}' over '{breakdown}' when '{subspace}' is {str} than mean from {intervals[i]} to {intervals[i + 1]}.")
            if trend != []:
                return trend
        elif agg == "min":
            grouped = self.df.groupby(breakdown)[measure].min()
            # Convert the series to a NumPy array for processing
            values = grouped.values
            intervals = grouped.index
            for i in range(len(values) - 1):
                # Apply Kendall's Tau to check statistical significance
                tau, p_value = kendalltau([values[i]], [values[i + 1]])

                # Check p-value for significance (commonly 0.05)
                if p_value <= 0.05:
                    if tau > 0:
                        trend.append(f"The increasing trend of {agg} '{measure}' over '{breakdown}' when '{subspace}' is {str} than mean from {intervals[i]} to {intervals[i + 1]}.")
                    elif tau < 0:
                        trend.append(f"The decreasing trend of {agg} '{measure}' over '{breakdown}' when '{subspace}' is {str} than mean from {intervals[i]} to {intervals[i + 1]}.")
            if trend != []:
                return trend

    def categorization_C(self, subspace, breakdown, str):
        """Count the number of categories in a breakdown."""
        counts = self.df[breakdown].unique()
        # focus is the most frequent category(has the most records)
        focus = self.df[breakdown].value_counts().idxmax()
        return f"There are {len(counts)} categories of '{breakdown}' which are {counts}, when '{subspace}' is '{str}', among which '{focus}' is the most frequent category."
    
    def categorization_N(self, subspace, breakdown, str):
        """Count the number of categories in a breakdown."""
        counts = self.df[breakdown].unique()
        # focus is the most frequent category(has the most records)
        focus = self.df[breakdown].value_counts().idxmax()
        return f"There are {len(counts)} categories of '{breakdown}' which are {counts}, when '{subspace}' is {str} than mean, among which '{focus}' is the most frequent category."
    
    def distribution_C(self, subspace, breakdown, measure, agg, str):
        """Calculate the distribution over a breakdown."""
        distribution = pd.DataFrame()
        if agg == "sum":
            distribution = self.df.groupby(breakdown)[measure].sum()
            if len(distribution) <= 1:
                return f"The distribution of the {agg} '{measure}' over '{breakdown}' when '{subspace}' is '{str}' is empty."
            else:
                u = distribution.mean()
                std = distribution.std()
                statistic_d, pvalue_p = stats.kstest(distribution, 'norm', (u, std))
                if pvalue_p < 0.05:
                    # focus when distribution is not normal 
                    return f"The distribution of the {agg} '{measure}' over '{breakdown}' when '{subspace}' is '{str}' is not normal."
        elif agg == "mean":
            distribution = self.df.groupby(breakdown)[measure].mean()
            if len(distribution) <= 1:
                return f"The distribution of the {agg} '{measure}' over '{breakdown}' when '{subspace}' is '{str}' is empty."
            else:
                u = distribution.mean()
                std = distribution.std()
                statistic_d, pvalue_p = stats.kstest(distribution, 'norm', (u, std))
                if pvalue_p < 0.05:
                    # focus when distribution is not normal 
                    return f"The distribution of the {agg} '{measure}' over '{breakdown}' when '{subspace}' is '{str}' is not normal."
        elif agg == "max":
            distribution = self.df.groupby(breakdown)[measure].max()
            if len(distribution) <= 1:
                return f"The distribution of the {agg} '{measure}' over '{breakdown}' when '{subspace}' is '{str}' is empty."
            else:
                u = distribution.mean()
                std = distribution.std()
                statistic_d, pvalue_p = stats.kstest(distribution, 'norm', (u, std))
                if pvalue_p < 0.05:
                    # focus when distribution is not normal 
                    return f"The distribution of the {agg} '{measure}' over '{breakdown}' when '{subspace}' is '{str}' is not normal."
        elif agg == "min":
            distribution = self.df.groupby(breakdown)[measure].min()  
            if len(distribution) <= 1:
                return f"The distribution of the {agg} '{measure}' over '{breakdown}' when '{subspace}' is '{str}' is empty."
            else:
                u = distribution.mean()
                std = distribution.std()
                statistic_d, pvalue_p = stats.kstest(distribution, 'norm', (u, std))
                if pvalue_p < 0.05:
                    # focus when distribution is not normal 
                    return f"The distribution of the {agg} '{measure}' over '{breakdown}' when '{subspace}' is '{str}' is not normal." 
        return f"The normal distribution of the {agg} '{measure}' over '{breakdown}' when '{subspace}' is '{str}'."
    
    def distribution_N(self, subspace, breakdown, measure, agg, str):
        """Calculate the distribution over a breakdown."""
        distribution = pd.DataFrame()
        if agg == "sum":
            distribution = self.df.groupby(breakdown)[measure].sum()
            if len(distribution) <= 1:
                return f"The distribution of the {agg} '{measure}' over '{breakdown}' when '{subspace}' is {str} than mean is empty."
            else:
                u = distribution.mean()
                std = distribution.std()
                statistic_d, pvalue_p = stats.kstest(distribution, 'norm', (u, std))
                if pvalue_p < 0.05:
                    # focus when distribution is not normal 
                    return f"The distribution of the {agg} '{measure}' over '{breakdown}' when '{subspace}' is {str} than mean is not normal."
        elif agg == "mean":
            distribution = self.df.groupby(breakdown)[measure].mean()
            if len(distribution) <= 1:
                return f"The distribution of the {agg} '{measure}' over '{breakdown}' when '{subspace}' is {str} than mean is empty."
            else:
                u = distribution.mean()
                std = distribution.std()
                statistic_d, pvalue_p = stats.kstest(distribution, 'norm', (u, std))
                if pvalue_p < 0.05:
                    # focus when distribution is not normal 
                    return f"The distribution of the {agg} '{measure}' over '{breakdown}' when '{subspace}' is {str} than mean is not normal."
        elif agg == "max":
            distribution = self.df.groupby(breakdown)[measure].max()
            if len(distribution) <= 1:
                return f"The distribution of the {agg} '{measure}' over '{breakdown}' when '{subspace}' is {str} than mean is empty."
            else:
                u = distribution.mean()
                std = distribution.std()
                statistic_d, pvalue_p = stats.kstest(distribution, 'norm', (u, std))
                if pvalue_p < 0.05:
                    # focus when distribution is not normal 
                    return f"The distribution of the {agg} '{measure}' over '{breakdown}' when '{subspace}' is {str} than mean is not normal."
        elif agg == "min":
            distribution = self.df.groupby(breakdown)[measure].min()  
            if len(distribution) <= 1:
                return f"The distribution of the {agg} '{measure}' over '{breakdown}' when '{subspace}' is {str} than mean is empty."
            else:
                u = distribution.mean()
                std = distribution.std()
                statistic_d, pvalue_p = stats.kstest(distribution, 'norm', (u, std))
                if pvalue_p < 0.05:
                    # focus when distribution is not normal 
                    return f"The distribution of the {agg} '{measure}' over '{breakdown}' when '{subspace}' is {str} than mean is not normal." 
        return f"The normal distribution of the {agg} '{measure}' over '{breakdown}' when '{subspace}' is {str} than mean."

    def rank_C(self, subspace, breakdown, measure, agg, str, top_n=3):
        """Rank items in a breakdown by measure."""
        if agg == "sum":
            ranked = self.df.groupby(breakdown)[measure].sum().sort_values(ascending=False)
            focus = ranked.head(top_n).index.tolist()
        elif agg == "mean":
            ranked = self.df.groupby(breakdown)[measure].mean().sort_values(ascending=False)
            focus = ranked.head(top_n).index.tolist()
        elif agg == "max":
            ranked = self.df.groupby(breakdown)[measure].max().sort_values(ascending=False)
            focus = ranked.head(top_n).index.tolist()
        elif agg == "min":
            ranked = self.df.groupby(breakdown)[measure].min().sort_values(ascending=False)
            focus = ranked.head(top_n).index.tolist()
        return f"In the {agg} '{measure}' ranking of different '{breakdown}', the top three are {focus} when '{subspace}' is '{str}'."
    
    def rank_N(self, subspace, breakdown, measure, agg, str, top_n=3):
        """Rank items in a breakdown by measure."""
        if agg == "sum":
            ranked = self.df.groupby(breakdown)[measure].sum().sort_values(ascending=False)
            focus = ranked.head(top_n).index.tolist()
        elif agg == "mean":
            ranked = self.df.groupby(breakdown)[measure].mean().sort_values(ascending=False)
            focus = ranked.head(top_n).index.tolist()
        elif agg == "max":
            ranked = self.df.groupby(breakdown)[measure].max().sort_values(ascending=False)
            focus = ranked.head(top_n).index.tolist()
        elif agg == "min":
            ranked = self.df.groupby(breakdown)[measure].min().sort_values(ascending=False)
            focus = ranked.head(top_n).index.tolist()
        return f"In the {agg} '{measure}' ranking of different '{breakdown}', the top three are {focus} when '{subspace}' is {str} than mean."

    def association_C(self, subspace, measure, str):
        """Calculate the Pearson correlation coefficient."""
        # May return NULL, need to eliminate
        for i in range(len(measure)):
            for j in range(i + 1, len(measure)):
                measure1 = measure[i]
                measure2 = measure[j]
                correlation = self.df[[measure1, measure2]].corr(method = 'pearson').iloc[0, 1]
                level = "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.3 else "weak"
                if correlation < 0:
                    level = "negative " + level 
                return f"There is a {level} relationship between {measure1} and {measure2} when '{subspace}' is '{str}', with pearson correlation coefficient of {correlation:.2f}."
    
    def association_N(self, subspace, measure, str):
        """Calculate the Pearson correlation coefficient."""
        # May return NULL, need to eliminate
        for i in range(len(measure)):
            for j in range(i + 1, len(measure)):
                measure1 = measure[i]
                measure2 = measure[j]
                correlation = self.df[[measure1, measure2]].corr(method = 'pearson').iloc[0, 1]
                level = "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.3 else "weak"
                if correlation < 0:
                    level = "negative " + level
                return f"There is a {level} relationship between {measure1} and {measure2} when '{subspace}' is '{str}' than mean, with pearson correlation coefficient of {correlation:.2f}."
            
    def extreme_C(self, subspace, breakdown, measure, agg, str, extreme_type):
        """Find the maximum or minimum value of a measure."""
        if agg == "sum":
            if extreme_type == "maximum":
               extreme = self.df.groupby(breakdown)[measure].sum().max()
               extreme_index = self.df.groupby(breakdown)[measure].sum().idxmax()
            elif extreme_type == "minimum":
                extreme = self.df.groupby(breakdown)[measure].sum().min()
                extreme_index = self.df.groupby(breakdown)[measure].sum().idxmin()
        elif agg == "mean":
            if extreme_type == "maximum":
                extreme = self.df.groupby(breakdown)[measure].mean().max()
                extreme_index = self.df.groupby(breakdown)[measure].mean().idxmax()
            elif extreme_type == "minimum":
                extreme = self.df.groupby(breakdown)[measure].mean().min()
                extreme_index = self.df.groupby(breakdown)[measure].mean().idxmin()
        elif agg == "max":
            if extreme_type == "maximum":
                extreme = self.df.groupby(breakdown)[measure].max().max()
                extreme_index = self.df.groupby(breakdown)[measure].max().idxmax()
            elif extreme_type == "minimum":
                extreme = self.df.groupby(breakdown)[measure].max().min()
                extreme_index = self.df.groupby(breakdown)[measure].max().idxmin()
        elif agg == "min":
            if extreme_type == "maximum":
                extreme = self.df.groupby(breakdown)[measure].min().max()
                extreme_index = self.df.groupby(breakdown)[measure].min().idxmax()
            elif extreme_type == "minimum":
                extreme = self.df.groupby(breakdown)[measure].min().min()
                extreme_index = self.df.groupby(breakdown)[measure].min().idxmin()              
        return f"The {extreme_type} value of the {agg} '{measure}' is {extreme} from '{breakdown}'={extreme_index} when '{subspace}' is '{str}'."
    
    def extreme_N(self, subspace, breakdown, measure, agg, str, extreme_type):
        """Find the maximum or minimum value of a measure."""
        if agg == "sum":
            if extreme_type == "maximum":
               extreme = self.df.groupby(breakdown)[measure].sum().max()
               extreme_index = self.df.groupby(breakdown)[measure].sum().idxmax()
            elif extreme_type == "minimum":
                extreme = self.df.groupby(breakdown)[measure].sum().min()
                extreme_index = self.df.groupby(breakdown)[measure].sum().idxmin()
        elif agg == "mean":
            if extreme_type == "maximum":
                extreme = self.df.groupby(breakdown)[measure].mean().max()
                extreme_index = self.df.groupby(breakdown)[measure].mean().idxmax()
            elif extreme_type == "minimum":
                extreme = self.df.groupby(breakdown)[measure].mean().min()
                extreme_index = self.df.groupby(breakdown)[measure].mean().idxmin()
        elif agg == "max":
            if extreme_type == "maximum":
                extreme = self.df.groupby(breakdown)[measure].max().max()
                extreme_index = self.df.groupby(breakdown)[measure].max().idxmax()
            elif extreme_type == "minimum":
                extreme = self.df.groupby(breakdown)[measure].max().min()
                extreme_index = self.df.groupby(breakdown)[measure].max().idxmin()
        elif agg == "min":
            if extreme_type == "maximum":
                extreme = self.df.groupby(breakdown)[measure].min().max()
                extreme_index = self.df.groupby(breakdown)[measure].min().idxmax()
            elif extreme_type == "minimum":
                extreme = self.df.groupby(breakdown)[measure].min().min()
                extreme_index = self.df.groupby(breakdown)[measure].min().idxmin() 
        return f"The {extreme_type} value of the {agg} '{measure}' is {extreme} from '{breakdown}'={extreme_index} when '{subspace}' is {str} than mean."

    def outlier_C(self, subspace, breakdown, measure, agg, str, threshold=1.5):
        """Identify outliers using the IQR method."""
        outliers = ""
        if agg == "sum":
            group_df = self.df.groupby(breakdown)[measure].sum()
            Q1 = group_df.quantile(0.25)
            Q3 = group_df.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            for value in group_df.values:
                if value < lower_bound or value > upper_bound:
                    idx = (list(group_df).index(value))
                    outliers = group_df.index.tolist()[idx]
                    return f"The {agg} '{measure}' of '{outliers}' is an outlier when compare with that of other '{breakdown}' when '{subspace}' is '{str}'."
        elif agg == "mean":
            group_df = self.df.groupby(breakdown)[measure].mean()
            Q1 = group_df.quantile(0.25)
            Q3 = group_df.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            for value in group_df.values:
                if value < lower_bound or value > upper_bound:
                    idx = (list(group_df).index(value))
                    outliers = group_df.index.tolist()[idx]
                    return f"The {agg} '{measure}' of '{outliers}' is an outlier when compare with that of other '{breakdown}' when '{subspace}' is '{str}'."
        elif agg == "max":
            group_df = self.df.groupby(breakdown)[measure].max()
            Q1 = group_df.quantile(0.25)
            Q3 = group_df.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            for value in group_df.values:
                if value < lower_bound or value > upper_bound:
                    idx = (list(group_df).index(value))
                    outliers = group_df.index.tolist()[idx]
                    return f"The {agg} '{measure}' of '{outliers}' is an outlier when compare with that of other '{breakdown}' when '{subspace}' is '{str}'."
        elif agg == "min":
            group_df = self.df.groupby(breakdown)[measure].min()
            Q1 = group_df.quantile(0.25)
            Q3 = group_df.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            for value in group_df.values:
                if value < lower_bound or value > upper_bound:
                    idx = (list(group_df).index(value))
                    outliers = group_df.index.tolist()[idx]
                    return f"The {agg} '{measure}' of '{outliers}' is an outlier when compare with that of other '{breakdown}' when '{subspace}' is '{str}'."
        return f"There is no outlier in the {agg} '{measure}' when '{subspace}' is '{str}'."
    
    def outlier_N(self, subspace, breakdown, measure, agg, str, threshold=1.5):
        """Identify outliers using the IQR method."""
        outliers = ""
        if agg == "sum":
            group_df = self.df.groupby(breakdown)[measure].sum()
            Q1 = group_df.quantile(0.25)
            Q3 = group_df.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            for value in group_df.values:
                if value < lower_bound or value > upper_bound:
                    idx = (list(group_df).index(value))
                    outliers = group_df.index.tolist()[idx]
                    return f"The {agg} '{measure}' of '{outliers}' is an outlier when compare with that of other '{breakdown}' when '{subspace}' is {str} than mean."
        elif agg == "mean":
            group_df = self.df.groupby(breakdown)[measure].mean()
            Q1 = group_df.quantile(0.25)
            Q3 = group_df.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            for value in group_df.values:
                if value < lower_bound or value > upper_bound:
                    idx = (list(group_df).index(value))
                    outliers = group_df.index.tolist()[idx]
                    return f"The {agg} '{measure}' of '{outliers}' is an outlier when compare with that of other '{breakdown}' when '{subspace}' is {str} than mean."
        elif agg == "max":
            group_df = self.df.groupby(breakdown)[measure].max()
            Q1 = group_df.quantile(0.25)
            Q3 = group_df.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            for value in group_df.values:
                if value < lower_bound or value > upper_bound:
                    idx = (list(group_df).index(value))
                    outliers = group_df.index.tolist()[idx]
                    return f"The {agg} '{measure}' of '{outliers}' is an outlier when compare with that of other '{breakdown}' when '{subspace}' is {str} than mean."
        elif agg == "min":
            group_df = self.df.groupby(breakdown)[measure].min()
            Q1 = group_df.quantile(0.25)
            Q3 = group_df.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            for value in group_df.values:
                if value < lower_bound or value > upper_bound:
                    idx = (list(group_df).index(value))
                    outliers = group_df.index.tolist()[idx]
                    return f"The {agg} '{measure}' of '{outliers}' is an outlier when compare with that of other '{breakdown}' when '{subspace}' is {str} than mean."
        return f"There is no outlier in the {agg} '{measure}' when '{subspace}' is {str} than mean."
    


def fact_generator(columns_from_gpt, df, user_selected_column):
    fact = []
    # Extract dataFrame by user_selected_column and the related columns from gpt 
    related_column = []
    for column in columns_from_gpt["related_columns"]:
        related_column.append(column["name"])
    df_for_cal = df[[user_selected_column] + related_column]   
    # Produce columns for subspace emumeration
    columns_dic = {columns_from_gpt["selected_column"]["name"]: columns_from_gpt["selected_column"]["dtype"]}
    for column in columns_from_gpt["related_columns"]:
        columns_dic[column["name"]] = column["dtype"]
    # subspace = list(columns_dic.keys())  # Any column can be part of subspace
    subspace = [col for col, dtype in columns_dic.items() if dtype == "C" or dtype == "T"]# subspace exclude dtype=="number"
    breakdown_C = [col for col, dtype in columns_dic.items() if dtype == "C"]
    breakdown_T = [col for col, dtype in columns_dic.items() if dtype == "T"]    
    measure = [col for col, dtype in columns_dic.items() if dtype == "N"]
    # Iterate over all columns for subspace
    result_C = []
    result_T = []
    for sub in subspace:
        for brk in breakdown_C:
            if brk == sub:  # Skip if breakdown column is the same as subspace
                continue
            for msr in measure:
                if msr == sub or msr == brk:  # Skip if measure column is the same as subspace or breakdown
                    continue
                result_C.append({
                    "subspace": sub,
                    "breakdown": brk,
                    "measure": msr
                })
        for brk in breakdown_T:
            if brk == sub:  # Skip if breakdown column is the same as subspace
                continue
            for msr in measure:
                if msr == sub or msr == brk:  # Skip if measure column is the same as subspace or breakdown
                    continue
                result_T.append({
                    "subspace": sub,
                    "breakdown": brk,
                    "measure": msr
                }) 
    # Calculate data fact in each subspace, when breakdown is "category"
    for res in result_C: 
        df_for_category = df_for_cal
        df_for_time = df_for_cal
        df_greater_than_mean = df_for_cal
        df_less_than_mean = df_for_cal
        for col in list(columns_dic.items()):  # example: col= [0:"Cylinders" 1:"number"]
            # When subspace is "category"
            if col[0] == res["subspace"] and col[1] == "C":
                unique_values = df_for_cal[res["subspace"]].unique()
                for value in unique_values:
                    df_for_category = df_for_cal[df_for_cal[res["subspace"]] == value]
                    fact_type_category = FactTypeGenerator(df_for_category)
                    # value fact is only calculated in result_C, since value fact isn't related to breakdown. No need to calculate again in result_T.
                    # fact.append(fact_type_category.value_C(res["subspace"], res["measure"], "sum",value))
                    # fact.append(fact_type_category.value_C(res["subspace"], res["measure"], "mean",value))
                    # fact.append(fact_type_category.value_C(res["subspace"], res["measure"], "max",value))
                    # fact.append(fact_type_category.value_C(res["subspace"], res["measure"], "min",value))
                    fact.append(fact_type_category.difference_C(res["subspace"], res["breakdown"], res["measure"], "sum", value))
                    fact.append(fact_type_category.difference_C(res["subspace"], res["breakdown"], res["measure"], "mean", value))
                    fact.append(fact_type_category.difference_C(res["subspace"], res["breakdown"], res["measure"], "max", value))
                    fact.append(fact_type_category.difference_C(res["subspace"], res["breakdown"], res["measure"], "min", value))
                    fact.append(fact_type_category.proportion_C(res["subspace"], res["breakdown"], res["measure"], "sum", value))
                    fact.append(fact_type_category.proportion_C(res["subspace"], res["breakdown"], res["measure"], "mean", value))
                    fact.append(fact_type_category.proportion_C(res["subspace"], res["breakdown"], res["measure"], "max", value))
                    fact.append(fact_type_category.proportion_C(res["subspace"], res["breakdown"], res["measure"], "min", value))
                    fact.append(fact_type_category.categorization_C(res["subspace"], res["breakdown"],value))
                    fact.append(fact_type_category.distribution_C(res["subspace"], res["breakdown"], res["measure"], "sum", value))
                    fact.append(fact_type_category.distribution_C(res["subspace"], res["breakdown"], res["measure"], "mean", value))
                    fact.append(fact_type_category.distribution_C(res["subspace"], res["breakdown"], res["measure"], "max", value))
                    fact.append(fact_type_category.distribution_C(res["subspace"], res["breakdown"], res["measure"], "min", value))
                    fact.append(fact_type_category.rank_C(res["subspace"], res["breakdown"], res["measure"], "sum", value))
                    fact.append(fact_type_category.rank_C(res["subspace"], res["breakdown"], res["measure"], "mean", value))
                    fact.append(fact_type_category.rank_C(res["subspace"], res["breakdown"], res["measure"], "max", value))
                    fact.append(fact_type_category.rank_C(res["subspace"], res["breakdown"], res["measure"], "min", value))
                    fact.append(fact_type_category.association_C(res["subspace"], measure, value))
                    fact.append(fact_type_category.extreme_C(res["subspace"], res["breakdown"], res["measure"], "sum", value, "maximum"))
                    fact.append(fact_type_category.extreme_C(res["subspace"], res["breakdown"], res["measure"], "sum", value, "minimum"))
                    fact.append(fact_type_category.extreme_C(res["subspace"], res["breakdown"], res["measure"], "mean", value, "maximum"))  
                    fact.append(fact_type_category.extreme_C(res["subspace"], res["breakdown"], res["measure"], "mean", value, "minimum"))
                    fact.append(fact_type_category.extreme_C(res["subspace"], res["breakdown"], res["measure"], "max", value, "maximum"))
                    fact.append(fact_type_category.extreme_C(res["subspace"], res["breakdown"], res["measure"], "max", value, "minimum"))
                    fact.append(fact_type_category.extreme_C(res["subspace"], res["breakdown"], res["measure"], "min", value, "maximum"))
                    fact.append(fact_type_category.extreme_C(res["subspace"], res["breakdown"], res["measure"], "min", value, "minimum"))
                    fact.append(fact_type_category.outlier_C(res["subspace"], res["breakdown"], res["measure"], "sum", value))
                    fact.append(fact_type_category.outlier_C(res["subspace"], res["breakdown"], res["measure"], "mean", value))
                    fact.append(fact_type_category.outlier_C(res["subspace"], res["breakdown"], res["measure"], "max", value))
                    fact.append(fact_type_category.outlier_C(res["subspace"], res["breakdown"], res["measure"], "min", value))
                                       
            # When subspace is "time"
            elif col[0] == res["subspace"] and col[1] == "T":
                unique_values = df_for_cal[res["subspace"]].unique()
                for value in unique_values:
                    df_for_time = df_for_cal[df_for_cal[res["subspace"]] == value]
                    fact_type_time = FactTypeGenerator(df_for_time)
                    # value fact is only calculated in result_C, since value fact isn't related to breakdown. No need to calculate again in result_T.
                    # fact.append(fact_type_time.value_C(res["subspace"], res["measure"], "sum",value))
                    # fact.append(fact_type_time.value_C(res["subspace"], res["measure"], "mean",value))
                    # fact.append(fact_type_time.value_C(res["subspace"], res["measure"], "max",value))
                    # fact.append(fact_type_time.value_C(res["subspace"], res["measure"], "min",value))
                    fact.append(fact_type_time.difference_C(res["subspace"], res["breakdown"], res["measure"], "sum", value))
                    fact.append(fact_type_time.difference_C(res["subspace"], res["breakdown"], res["measure"], "mean", value))
                    fact.append(fact_type_time.difference_C(res["subspace"], res["breakdown"], res["measure"], "max", value))
                    fact.append(fact_type_time.difference_C(res["subspace"], res["breakdown"], res["measure"], "min", value))
                    fact.append(fact_type_time.proportion_C(res["subspace"], res["breakdown"], res["measure"], "sum", value))
                    fact.append(fact_type_time.proportion_C(res["subspace"], res["breakdown"], res["measure"], "mean", value))
                    fact.append(fact_type_time.proportion_C(res["subspace"], res["breakdown"], res["measure"], "max", value))
                    fact.append(fact_type_time.proportion_C(res["subspace"], res["breakdown"], res["measure"], "min", value))
                    fact.append(fact_type_time.categorization_C(res["subspace"], res["breakdown"],value))
                    fact.append(fact_type_time.distribution_C(res["subspace"], res["breakdown"], res["measure"], "sum", value))
                    fact.append(fact_type_time.distribution_C(res["subspace"], res["breakdown"], res["measure"], "mean", value))
                    fact.append(fact_type_time.distribution_C(res["subspace"], res["breakdown"], res["measure"], "max", value))
                    fact.append(fact_type_time.distribution_C(res["subspace"], res["breakdown"], res["measure"], "min", value))
                    fact.append(fact_type_time.rank_C(res["subspace"], res["breakdown"], res["measure"], "sum", value))
                    fact.append(fact_type_time.rank_C(res["subspace"], res["breakdown"], res["measure"], "mean", value))
                    fact.append(fact_type_time.rank_C(res["subspace"], res["breakdown"], res["measure"], "max", value))
                    fact.append(fact_type_time.rank_C(res["subspace"], res["breakdown"], res["measure"], "min", value))
                    fact.append(fact_type_time.association_C(res["subspace"], measure, value))
                    fact.append(fact_type_time.extreme_C(res["subspace"], res["breakdown"], res["measure"], "sum", value, "maximum")) 
                    fact.append(fact_type_time.extreme_C(res["subspace"], res["breakdown"], res["measure"], "sum", value, "minimum"))
                    fact.append(fact_type_time.extreme_C(res["subspace"], res["breakdown"], res["measure"], "mean", value, "maximum"))
                    fact.append(fact_type_time.extreme_C(res["subspace"], res["breakdown"], res["measure"], "mean", value, "minimum"))
                    fact.append(fact_type_time.extreme_C(res["subspace"], res["breakdown"], res["measure"], "max", value, "maximum"))
                    fact.append(fact_type_time.extreme_C(res["subspace"], res["breakdown"], res["measure"], "max", value, "minimum"))
                    fact.append(fact_type_time.extreme_C(res["subspace"], res["breakdown"], res["measure"], "min", value, "maximum"))
                    fact.append(fact_type_time.extreme_C(res["subspace"], res["breakdown"], res["measure"], "min", value, "minimum"))
                    fact.append(fact_type_time.outlier_C(res["subspace"], res["breakdown"], res["measure"], "sum", value))
                    fact.append(fact_type_time.outlier_C(res["subspace"], res["breakdown"], res["measure"], "mean", value))
                    fact.append(fact_type_time.outlier_C(res["subspace"], res["breakdown"], res["measure"], "max", value))
                    fact.append(fact_type_time.outlier_C(res["subspace"], res["breakdown"], res["measure"], "min", value))
                   
            # When subspace is "number"     
            # elif col[0] == res["subspace"] and col[1] == "N":
            #     df_greater_than_mean = df_for_cal[df_for_cal[res["subspace"]] >= df_for_cal[res["subspace"]].mean()]
            #     df_less_than_mean = df_for_cal[df_for_cal[res["subspace"]] < df_for_cal[res["subspace"]].mean()]   
            #     fact_type_greater_than_mean = FactTypeGenerator(df_greater_than_mean)
            #     fact_type_less_than_mean = FactTypeGenerator(df_less_than_mean)
            #     # value fact is only calculated in result_C, since value fact isn't related to breakdown. No need to calculate again in result_T.
            #     fact.append(fact_type_greater_than_mean.value_N(res["subspace"], res["measure"], "sum", "greater"))
            #     fact.append(fact_type_greater_than_mean.value_N(res["subspace"], res["measure"], "mean", "greater"))
            #     fact.append(fact_type_greater_than_mean.value_N(res["subspace"], res["measure"], "max", "greater"))
            #     fact.append(fact_type_greater_than_mean.value_N(res["subspace"], res["measure"], "min", "greater"))
            #     fact.append(fact_type_less_than_mean.value_N(res["subspace"], res["measure"], "sum", "less"))
            #     fact.append(fact_type_less_than_mean.value_N(res["subspace"], res["measure"], "mean", "less"))
            #     fact.append(fact_type_less_than_mean.value_N(res["subspace"], res["measure"], "max", "less"))
            #     fact.append(fact_type_less_than_mean.value_N(res["subspace"], res["measure"], "min", "less"))
            #     fact.append(fact_type_greater_than_mean.difference_N(res["subspace"], res["breakdown"], res["measure"], "sum", "greater"))
            #     fact.append(fact_type_greater_than_mean.difference_N(res["subspace"], res["breakdown"], res["measure"], "mean", "greater"))
            #     fact.append(fact_type_greater_than_mean.difference_N(res["subspace"], res["breakdown"], res["measure"], "max", "greater"))
            #     fact.append(fact_type_greater_than_mean.difference_N(res["subspace"], res["breakdown"], res["measure"], "min", "greater"))
            #     fact.append(fact_type_less_than_mean.difference_N(res["subspace"], res["breakdown"], res["measure"], "sum", "less"))
            #     fact.append(fact_type_less_than_mean.difference_N(res["subspace"], res["breakdown"], res["measure"], "mean", "less"))
            #     fact.append(fact_type_less_than_mean.difference_N(res["subspace"], res["breakdown"], res["measure"], "max", "less"))
            #     fact.append(fact_type_less_than_mean.difference_N(res["subspace"], res["breakdown"], res["measure"], "min", "less"))
            #     fact.append(fact_type_greater_than_mean.proportion_N(res["subspace"], res["breakdown"], res["measure"], "sum", "greater"))
            #     fact.append(fact_type_greater_than_mean.proportion_N(res["subspace"], res["breakdown"], res["measure"], "mean", "greater"))
            #     fact.append(fact_type_greater_than_mean.proportion_N(res["subspace"], res["breakdown"], res["measure"], "max", "greater"))
            #     fact.append(fact_type_greater_than_mean.proportion_N(res["subspace"], res["breakdown"], res["measure"], "min", "greater"))
            #     fact.append(fact_type_less_than_mean.proportion_N(res["subspace"], res["breakdown"], res["measure"], "sum", "less"))
            #     fact.append(fact_type_less_than_mean.proportion_N(res["subspace"], res["breakdown"], res["measure"], "mean", "less"))
            #     fact.append(fact_type_less_than_mean.proportion_N(res["subspace"], res["breakdown"], res["measure"], "max", "less"))
            #     fact.append(fact_type_less_than_mean.proportion_N(res["subspace"], res["breakdown"], res["measure"], "min", "less"))
            #     fact.append(fact_type_greater_than_mean.categorization_N(res["subspace"], res["breakdown"],"greater"))
            #     fact.append(fact_type_less_than_mean.categorization_N(res["subspace"], res["breakdown"],"less"))
            #     fact.append(fact_type_greater_than_mean.distribution_N(res["subspace"], res["breakdown"], res["measure"], "sum", "greater"))
            #     fact.append(fact_type_greater_than_mean.distribution_N(res["subspace"], res["breakdown"], res["measure"], "mean", "greater"))
            #     fact.append(fact_type_greater_than_mean.distribution_N(res["subspace"], res["breakdown"], res["measure"], "max", "greater"))
            #     fact.append(fact_type_greater_than_mean.distribution_N(res["subspace"], res["breakdown"], res["measure"], "min", "greater"))
            #     fact.append(fact_type_less_than_mean.distribution_N(res["subspace"], res["breakdown"], res["measure"], "sum", "less"))
            #     fact.append(fact_type_less_than_mean.distribution_N(res["subspace"], res["breakdown"], res["measure"], "mean", "less"))
            #     fact.append(fact_type_less_than_mean.distribution_N(res["subspace"], res["breakdown"], res["measure"], "max", "less"))
            #     fact.append(fact_type_less_than_mean.distribution_N(res["subspace"], res["breakdown"], res["measure"], "min", "less"))
            #     fact.append(fact_type_greater_than_mean.rank_N(res["subspace"], res["breakdown"], res["measure"], "sum", "greater"))
            #     fact.append(fact_type_greater_than_mean.rank_N(res["subspace"], res["breakdown"], res["measure"], "mean", "greater"))
            #     fact.append(fact_type_greater_than_mean.rank_N(res["subspace"], res["breakdown"], res["measure"], "max", "greater"))
            #     fact.append(fact_type_greater_than_mean.rank_N(res["subspace"], res["breakdown"], res["measure"], "min", "greater"))
            #     fact.append(fact_type_less_than_mean.rank_N(res["subspace"], res["breakdown"], res["measure"], "sum", "less"))
            #     fact.append(fact_type_less_than_mean.rank_N(res["subspace"], res["breakdown"], res["measure"], "mean", "less"))
            #     fact.append(fact_type_less_than_mean.rank_N(res["subspace"], res["breakdown"], res["measure"], "max", "less"))
            #     fact.append(fact_type_less_than_mean.rank_N(res["subspace"], res["breakdown"], res["measure"], "min", "less"))
            #     fact.append(fact_type_greater_than_mean.association_N(res["subspace"], measure, "greater"))
            #     fact.append(fact_type_less_than_mean.association_N(res["subspace"], measure, "less"))
            #     fact.append(fact_type_greater_than_mean.extreme_N(res["subspace"], res["breakdown"], res["measure"], "sum", "greater", "maximum"))
            #     fact.append(fact_type_greater_than_mean.extreme_N(res["subspace"], res["breakdown"], res["measure"], "sum", "greater", "minimum"))
            #     fact.append(fact_type_greater_than_mean.extreme_N(res["subspace"], res["breakdown"], res["measure"], "mean", "greater", "maximum"))
            #     fact.append(fact_type_greater_than_mean.extreme_N(res["subspace"], res["breakdown"], res["measure"], "mean", "greater", "minimum"))
            #     fact.append(fact_type_greater_than_mean.extreme_N(res["subspace"], res["breakdown"], res["measure"], "max", "greater", "maximum"))
            #     fact.append(fact_type_greater_than_mean.extreme_N(res["subspace"], res["breakdown"], res["measure"], "max", "greater", "minimum"))
            #     fact.append(fact_type_greater_than_mean.extreme_N(res["subspace"], res["breakdown"], res["measure"], "min", "greater", "maximum"))
            #     fact.append(fact_type_greater_than_mean.extreme_N(res["subspace"], res["breakdown"], res["measure"], "min", "greater", "minimum"))
            #     fact.append(fact_type_less_than_mean.extreme_N(res["subspace"], res["breakdown"], res["measure"], "sum", "less", "maximum"))
            #     fact.append(fact_type_less_than_mean.extreme_N(res["subspace"], res["breakdown"], res["measure"], "sum", "less", "minimum"))
            #     fact.append(fact_type_less_than_mean.extreme_N(res["subspace"], res["breakdown"], res["measure"], "mean", "less", "maximum"))
            #     fact.append(fact_type_less_than_mean.extreme_N(res["subspace"], res["breakdown"], res["measure"], "mean", "less", "minimum"))
            #     fact.append(fact_type_less_than_mean.extreme_N(res["subspace"], res["breakdown"], res["measure"], "max", "less", "maximum"))
            #     fact.append(fact_type_less_than_mean.extreme_N(res["subspace"], res["breakdown"], res["measure"], "max", "less", "minimum"))
            #     fact.append(fact_type_less_than_mean.extreme_N(res["subspace"], res["breakdown"], res["measure"], "min", "less", "maximum"))
            #     fact.append(fact_type_less_than_mean.extreme_N(res["subspace"], res["breakdown"], res["measure"], "min", "less", "minimum"))
            #     fact.append(fact_type_greater_than_mean.outlier_N(res["subspace"], res["breakdown"], res["measure"], "sum", "greater"))
            #     fact.append(fact_type_greater_than_mean.outlier_N(res["subspace"], res["breakdown"], res["measure"], "mean", "greater"))
            #     fact.append(fact_type_greater_than_mean.outlier_N(res["subspace"], res["breakdown"], res["measure"], "max", "greater"))
            #     fact.append(fact_type_greater_than_mean.outlier_N(res["subspace"], res["breakdown"], res["measure"], "min", "greater"))
            #     fact.append(fact_type_less_than_mean.outlier_N(res["subspace"], res["breakdown"], res["measure"], "sum", "less"))
            #     fact.append(fact_type_less_than_mean.outlier_N(res["subspace"], res["breakdown"], res["measure"], "mean", "less"))
            #     fact.append(fact_type_less_than_mean.outlier_N(res["subspace"], res["breakdown"], res["measure"], "max", "less"))
            #     fact.append(fact_type_less_than_mean.outlier_N(res["subspace"], res["breakdown"], res["measure"], "min", "less"))
                
    # Calculate data fact in each subspace, when breakdown is "time"   
    for res in result_T: 
        df_for_category = df_for_cal
        df_for_time = df_for_cal
        df_greater_than_mean = df_for_cal
        df_less_than_mean = df_for_cal
        for col in list(columns_dic.items()):  # example: col= [0:"Cylinders"1:"number"]
            # When subspace is "category"
            if col[0] == res["subspace"] and col[1] == "C":
                unique_values = df_for_cal[res["subspace"]].unique()
                for value in unique_values:
                    df_for_category = df_for_cal[df_for_cal[res["subspace"]] == value]
                    fact_type_category = FactTypeGenerator(df_for_category)
                    fact.append(fact_type_category.difference_C(res["subspace"], res["breakdown"], res["measure"], "sum", value))
                    fact.append(fact_type_category.difference_C(res["subspace"], res["breakdown"], res["measure"], "mean", value))
                    fact.append(fact_type_category.difference_C(res["subspace"], res["breakdown"], res["measure"], "max", value))
                    fact.append(fact_type_category.difference_C(res["subspace"], res["breakdown"], res["measure"], "min", value))
                    fact.append(fact_type_category.proportion_C(res["subspace"], res["breakdown"], res["measure"], "sum", value))
                    fact.append(fact_type_category.proportion_C(res["subspace"], res["breakdown"], res["measure"], "mean", value))
                    fact.append(fact_type_category.proportion_C(res["subspace"], res["breakdown"], res["measure"], "max", value))
                    fact.append(fact_type_category.proportion_C(res["subspace"], res["breakdown"], res["measure"], "min", value))
                    fact.append(fact_type_category.overall_trend_C(res["subspace"], res["breakdown"], res["measure"], "sum", value))
                    fact.append(fact_type_category.overall_trend_C(res["subspace"], res["breakdown"], res["measure"], "mean", value))
                    fact.append(fact_type_category.overall_trend_C(res["subspace"], res["breakdown"], res["measure"], "max", value))
                    fact.append(fact_type_category.overall_trend_C(res["subspace"], res["breakdown"], res["measure"], "min", value))
                    fact.append(fact_type_category.segment_trend_C(res["subspace"], res["breakdown"], res["measure"], "sum", value))
                    fact.append(fact_type_category.segment_trend_C(res["subspace"], res["breakdown"], res["measure"], "mean", value))
                    fact.append(fact_type_category.segment_trend_C(res["subspace"], res["breakdown"], res["measure"], "max", value))
                    fact.append(fact_type_category.segment_trend_C(res["subspace"], res["breakdown"], res["measure"], "min", value))
                    fact.append(fact_type_category.rank_C(res["subspace"], res["breakdown"], res["measure"], "sum", value))
                    fact.append(fact_type_category.rank_C(res["subspace"], res["breakdown"], res["measure"], "mean", value))
                    fact.append(fact_type_category.rank_C(res["subspace"], res["breakdown"], res["measure"], "max", value))
                    fact.append(fact_type_category.rank_C(res["subspace"], res["breakdown"], res["measure"], "min", value))
                    fact.append(fact_type_category.association_C(res["subspace"], measure, value))
                    fact.append(fact_type_category.extreme_C(res["subspace"], res["breakdown"], res["measure"], "sum", value, "maximum"))
                    fact.append(fact_type_category.extreme_C(res["subspace"], res["breakdown"], res["measure"], "sum", value, "minimum"))
                    fact.append(fact_type_category.extreme_C(res["subspace"], res["breakdown"], res["measure"], "mean", value, "maximum"))  
                    fact.append(fact_type_category.extreme_C(res["subspace"], res["breakdown"], res["measure"], "mean", value, "minimum"))
                    fact.append(fact_type_category.extreme_C(res["subspace"], res["breakdown"], res["measure"], "max", value, "maximum"))
                    fact.append(fact_type_category.extreme_C(res["subspace"], res["breakdown"], res["measure"], "max", value, "minimum"))
                    fact.append(fact_type_category.extreme_C(res["subspace"], res["breakdown"], res["measure"], "min", value, "maximum"))
                    fact.append(fact_type_category.extreme_C(res["subspace"], res["breakdown"], res["measure"], "min", value, "minimum"))
                    fact.append(fact_type_category.outlier_C(res["subspace"], res["breakdown"], res["measure"], "sum", value))
                    fact.append(fact_type_category.outlier_C(res["subspace"], res["breakdown"], res["measure"], "mean", value))
                    fact.append(fact_type_category.outlier_C(res["subspace"], res["breakdown"], res["measure"], "max", value))
                    fact.append(fact_type_category.outlier_C(res["subspace"], res["breakdown"], res["measure"], "min", value))
                   
            # When subspace is "time"
            elif col[0] == res["subspace"] and col[1] == "T":  
                unique_values = df_for_cal[res["subspace"]].unique()
                for value in unique_values:
                    df_for_time = df_for_cal[df_for_cal[res["subspace"]] == value]
                    fact_type_time = FactTypeGenerator(df_for_time)
                    fact.append(fact_type_time.difference_C(res["subspace"], res["breakdown"], res["measure"], "sum", value))
                    fact.append(fact_type_time.difference_C(res["subspace"], res["breakdown"], res["measure"], "mean", value))
                    fact.append(fact_type_time.difference_C(res["subspace"], res["breakdown"], res["measure"], "max", value))
                    fact.append(fact_type_time.difference_C(res["subspace"], res["breakdown"], res["measure"], "min", value))
                    fact.append(fact_type_time.proportion_C(res["subspace"], res["breakdown"], res["measure"], "sum", value))
                    fact.append(fact_type_time.proportion_C(res["subspace"], res["breakdown"], res["measure"], "mean", value))
                    fact.append(fact_type_time.proportion_C(res["subspace"], res["breakdown"], res["measure"], "max", value))
                    fact.append(fact_type_time.proportion_C(res["subspace"], res["breakdown"], res["measure"], "min", value))
                    fact.append(fact_type_time.overall_trend_C(res["subspace"], res["breakdown"], res["measure"], "sum", value))
                    fact.append(fact_type_time.overall_trend_C(res["subspace"], res["breakdown"], res["measure"], "mean", value))
                    fact.append(fact_type_time.overall_trend_C(res["subspace"], res["breakdown"], res["measure"], "max", value))
                    fact.append(fact_type_time.overall_trend_C(res["subspace"], res["breakdown"], res["measure"], "min", value))
                    fact.append(fact_type_time.segment_trend_C(res["subspace"], res["breakdown"], res["measure"], "sum", value))
                    fact.append(fact_type_time.segment_trend_C(res["subspace"], res["breakdown"], res["measure"], "mean", value))
                    fact.append(fact_type_time.segment_trend_C(res["subspace"], res["breakdown"], res["measure"], "max", value))
                    fact.append(fact_type_time.segment_trend_C(res["subspace"], res["breakdown"], res["measure"], "min", value))
                    fact.append(fact_type_time.rank_C(res["subspace"], res["breakdown"], res["measure"], "sum", value))
                    fact.append(fact_type_time.rank_C(res["subspace"], res["breakdown"], res["measure"], "mean", value))
                    fact.append(fact_type_time.rank_C(res["subspace"], res["breakdown"], res["measure"], "max", value))
                    fact.append(fact_type_time.rank_C(res["subspace"], res["breakdown"], res["measure"], "min", value))
                    fact.append(fact_type_time.association_C(res["subspace"], measure, value))
                    fact.append(fact_type_time.extreme_C(res["subspace"], res["breakdown"], res["measure"], "sum", value, "maximum")) 
                    fact.append(fact_type_time.extreme_C(res["subspace"], res["breakdown"], res["measure"], "sum", value, "minimum"))
                    fact.append(fact_type_time.extreme_C(res["subspace"], res["breakdown"], res["measure"], "mean", value, "maximum"))
                    fact.append(fact_type_time.extreme_C(res["subspace"], res["breakdown"], res["measure"], "mean", value, "minimum"))
                    fact.append(fact_type_time.extreme_C(res["subspace"], res["breakdown"], res["measure"], "max", value, "maximum"))
                    fact.append(fact_type_time.extreme_C(res["subspace"], res["breakdown"], res["measure"], "max", value, "minimum"))
                    fact.append(fact_type_time.extreme_C(res["subspace"], res["breakdown"], res["measure"], "min", value, "maximum"))
                    fact.append(fact_type_time.extreme_C(res["subspace"], res["breakdown"], res["measure"], "min", value, "minimum"))
                    fact.append(fact_type_time.outlier_C(res["subspace"], res["breakdown"], res["measure"], "sum", value))
                    fact.append(fact_type_time.outlier_C(res["subspace"], res["breakdown"], res["measure"], "mean", value))
                    fact.append(fact_type_time.outlier_C(res["subspace"], res["breakdown"], res["measure"], "max", value))
                    fact.append(fact_type_time.outlier_C(res["subspace"], res["breakdown"], res["measure"], "min", value))

            # # When subspace is "number"
            # elif col[0] == res["subspace"] and col[1] == "N":
            #     df_greater_than_mean = df_for_cal[df_for_cal[res["subspace"]] >= df_for_cal[res["subspace"]].mean()]
            #     df_less_than_mean = df_for_cal[df_for_cal[res["subspace"]] < df_for_cal[res["subspace"]].mean()]
            #     fact_type_greater_than_mean = FactTypeGenerator(df_greater_than_mean)
            #     fact_type_less_than_mean = FactTypeGenerator(df_less_than_mean)
            #     fact.append(fact_type_greater_than_mean.difference_N(res["subspace"], res["breakdown"], res["measure"], "sum", "greater"))
            #     fact.append(fact_type_greater_than_mean.difference_N(res["subspace"], res["breakdown"], res["measure"], "mean", "greater"))
            #     fact.append(fact_type_greater_than_mean.difference_N(res["subspace"], res["breakdown"], res["measure"], "max", "greater"))
            #     fact.append(fact_type_greater_than_mean.difference_N(res["subspace"], res["breakdown"], res["measure"], "min", "greater"))
            #     fact.append(fact_type_less_than_mean.difference_N(res["subspace"], res["breakdown"], res["measure"], "sum", "less"))
            #     fact.append(fact_type_less_than_mean.difference_N(res["subspace"], res["breakdown"], res["measure"], "mean", "less"))
            #     fact.append(fact_type_less_than_mean.difference_N(res["subspace"], res["breakdown"], res["measure"], "max", "less"))
            #     fact.append(fact_type_less_than_mean.difference_N(res["subspace"], res["breakdown"], res["measure"], "min", "less"))
            #     fact.append(fact_type_greater_than_mean.proportion_N(res["subspace"], res["breakdown"], res["measure"], "sum", "greater"))
            #     fact.append(fact_type_greater_than_mean.proportion_N(res["subspace"], res["breakdown"], res["measure"], "mean", "greater"))
            #     fact.append(fact_type_greater_than_mean.proportion_N(res["subspace"], res["breakdown"], res["measure"], "max", "greater"))
            #     fact.append(fact_type_greater_than_mean.proportion_N(res["subspace"], res["breakdown"], res["measure"], "min", "greater"))
            #     fact.append(fact_type_less_than_mean.proportion_N(res["subspace"], res["breakdown"], res["measure"], "sum", "less"))
            #     fact.append(fact_type_less_than_mean.proportion_N(res["subspace"], res["breakdown"], res["measure"], "mean", "less"))
            #     fact.append(fact_type_less_than_mean.proportion_N(res["subspace"], res["breakdown"], res["measure"], "max", "less"))
            #     fact.append(fact_type_less_than_mean.proportion_N(res["subspace"], res["breakdown"], res["measure"], "min", "less"))
            #     fact.append(fact_type_greater_than_mean.overall_trend_N(res["subspace"], res["breakdown"], res["measure"], "sum", "greater"))
            #     fact.append(fact_type_greater_than_mean.overall_trend_N(res["subspace"], res["breakdown"], res["measure"], "mean", "greater"))
            #     fact.append(fact_type_greater_than_mean.overall_trend_N(res["subspace"], res["breakdown"], res["measure"], "max", "greater"))
            #     fact.append(fact_type_greater_than_mean.overall_trend_N(res["subspace"], res["breakdown"], res["measure"], "min", "greater"))
            #     fact.append(fact_type_less_than_mean.overall_trend_N(res["subspace"], res["breakdown"], res["measure"], "sum", "less"))
            #     fact.append(fact_type_less_than_mean.overall_trend_N(res["subspace"], res["breakdown"], res["measure"], "mean", "less"))
            #     fact.append(fact_type_less_than_mean.overall_trend_N(res["subspace"], res["breakdown"], res["measure"], "max", "less"))
            #     fact.append(fact_type_less_than_mean.overall_trend_N(res["subspace"], res["breakdown"], res["measure"], "min", "less"))
            #     fact.append(fact_type_greater_than_mean.segment_trend_N(res["subspace"], res["breakdown"], res["measure"], "sum", "greater"))
            #     fact.append(fact_type_greater_than_mean.segment_trend_N(res["subspace"], res["breakdown"], res["measure"], "mean", "greater"))
            #     fact.append(fact_type_greater_than_mean.segment_trend_N(res["subspace"], res["breakdown"], res["measure"], "max", "greater"))
            #     fact.append(fact_type_greater_than_mean.segment_trend_N(res["subspace"], res["breakdown"], res["measure"], "min", "greater"))
            #     fact.append(fact_type_less_than_mean.segment_trend_N(res["subspace"], res["breakdown"], res["measure"], "sum", "less"))
            #     fact.append(fact_type_less_than_mean.segment_trend_N(res["subspace"], res["breakdown"], res["measure"], "mean", "less"))
            #     fact.append(fact_type_less_than_mean.segment_trend_N(res["subspace"], res["breakdown"], res["measure"], "max", "less"))
            #     fact.append(fact_type_less_than_mean.segment_trend_N(res["subspace"], res["breakdown"], res["measure"], "min", "less"))
            #     fact.append(fact_type_greater_than_mean.rank_N(res["subspace"], res["breakdown"], res["measure"], "sum", "greater"))
            #     fact.append(fact_type_greater_than_mean.rank_N(res["subspace"], res["breakdown"], res["measure"], "mean", "greater"))
            #     fact.append(fact_type_greater_than_mean.rank_N(res["subspace"], res["breakdown"], res["measure"], "max", "greater"))
            #     fact.append(fact_type_greater_than_mean.rank_N(res["subspace"], res["breakdown"], res["measure"], "min", "greater"))
            #     fact.append(fact_type_less_than_mean.rank_N(res["subspace"], res["breakdown"], res["measure"], "sum", "less"))
            #     fact.append(fact_type_less_than_mean.rank_N(res["subspace"], res["breakdown"], res["measure"], "mean", "less"))
            #     fact.append(fact_type_less_than_mean.rank_N(res["subspace"], res["breakdown"], res["measure"], "max", "less"))
            #     fact.append(fact_type_less_than_mean.rank_N(res["subspace"], res["breakdown"], res["measure"], "min", "less"))
            #     fact.append(fact_type_greater_than_mean.association_N(res["subspace"], measure, "greater"))
            #     fact.append(fact_type_less_than_mean.association_N(res["subspace"], measure, "less"))
            #     fact.append(fact_type_greater_than_mean.extreme_N(res["subspace"], res["breakdown"], res["measure"], "sum", "greater", "maximum"))
            #     fact.append(fact_type_greater_than_mean.extreme_N(res["subspace"], res["breakdown"], res["measure"], "sum", "greater", "minimum"))
            #     fact.append(fact_type_greater_than_mean.extreme_N(res["subspace"], res["breakdown"], res["measure"], "mean", "greater", "maximum"))
            #     fact.append(fact_type_greater_than_mean.extreme_N(res["subspace"], res["breakdown"], res["measure"], "mean", "greater", "minimum"))
            #     fact.append(fact_type_greater_than_mean.extreme_N(res["subspace"], res["breakdown"], res["measure"], "max", "greater", "maximum"))
            #     fact.append(fact_type_greater_than_mean.extreme_N(res["subspace"], res["breakdown"], res["measure"], "max", "greater", "minimum"))
            #     fact.append(fact_type_greater_than_mean.extreme_N(res["subspace"], res["breakdown"], res["measure"], "min", "greater", "maximum"))
            #     fact.append(fact_type_greater_than_mean.extreme_N(res["subspace"], res["breakdown"], res["measure"], "min", "greater", "minimum"))
            #     fact.append(fact_type_less_than_mean.extreme_N(res["subspace"], res["breakdown"], res["measure"], "sum", "less", "maximum"))
            #     fact.append(fact_type_less_than_mean.extreme_N(res["subspace"], res["breakdown"], res["measure"], "sum", "less", "minimum"))
            #     fact.append(fact_type_less_than_mean.extreme_N(res["subspace"], res["breakdown"], res["measure"], "mean", "less", "maximum"))
            #     fact.append(fact_type_less_than_mean.extreme_N(res["subspace"], res["breakdown"], res["measure"], "mean", "less", "minimum"))
            #     fact.append(fact_type_less_than_mean.extreme_N(res["subspace"], res["breakdown"], res["measure"], "max", "less", "maximum"))
            #     fact.append(fact_type_less_than_mean.extreme_N(res["subspace"], res["breakdown"], res["measure"], "max", "less", "minimum"))
            #     fact.append(fact_type_less_than_mean.extreme_N(res["subspace"], res["breakdown"], res["measure"], "min", "less", "maximum"))
            #     fact.append(fact_type_less_than_mean.extreme_N(res["subspace"], res["breakdown"], res["measure"], "min", "less", "minimum"))
            #     fact.append(fact_type_greater_than_mean.outlier_N(res["subspace"], res["breakdown"], res["measure"], "sum", "greater"))
            #     fact.append(fact_type_greater_than_mean.outlier_N(res["subspace"], res["breakdown"], res["measure"], "mean", "greater"))
            #     fact.append(fact_type_greater_than_mean.outlier_N(res["subspace"], res["breakdown"], res["measure"], "max", "greater"))
            #     fact.append(fact_type_greater_than_mean.outlier_N(res["subspace"], res["breakdown"], res["measure"], "min", "greater"))
            #     fact.append(fact_type_less_than_mean.outlier_N(res["subspace"], res["breakdown"], res["measure"], "sum", "less"))
            #     fact.append(fact_type_less_than_mean.outlier_N(res["subspace"], res["breakdown"], res["measure"], "mean", "less"))
            #     fact.append(fact_type_less_than_mean.outlier_N(res["subspace"], res["breakdown"], res["measure"], "max", "less"))
            #     fact.append(fact_type_less_than_mean.outlier_N(res["subspace"], res["breakdown"], res["measure"], "min", "less"))  
    return fact
