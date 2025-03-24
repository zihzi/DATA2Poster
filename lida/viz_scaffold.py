from dataclasses import asdict
from typing import Dict


class ChartScaffold(object):
    """Return code scaffold for charts in multiple visualization libraries"""

    def __init__(
        self,
    ) -> None:

        pass

    def get_template(self, df_data_scope: Dict):
        # 1. Use 'fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))' to create a 1x2 grid of subplots.
        instructions = {
                "role": "system",
                "content": f"""Use {df_data_scope[0]} and grouped by {df_data_scope[1]} as x-axis,and use {df_data_scope[2]} as y-axis to create a Bar Chart.
                                Bar Chart ALWAYS comply with the following rules: 
                                1. NEVER add a legend.
                                2. Add a title to bar chart which can intepret the chart in 8 words.  
                                3. Make sure to use color chosen from ['#CDE8E5', '#77B0AA', '#135D66']. 
                                4. NEVER import module named 'vega_datasets' in the code.
                                Solve the task carefully by completing ONLY the <imports> AND <stub> section. DO NOT WRITE ANY CODE TO LOAD THE DATA. 
                                The data is already loaded and available in the variable data.
                                Always add a type that is BASED on semantic_type to each field such as :Q, :O, :N, :T, :G. Use :T if semantic_type is year or date. 
                                The plot method must return an altair object (chart)`. Think step by step. \n""",
        }
        
        template = \
                f"""
                    import altair as alt
                    <imports>
                    def plot(data: pd.DataFrame):
                

                        <stub> # only modify this section
                     
                        return chart
                    chart = plot(data) # data already contains the data to be plotted.  Always include this line. No additional code beyond this line.
                """
        return template, instructions
    
            