import pandas as pd
import json
import altair as alt
import altair_saver
df = pd.read_csv("data/cas_cancer_nc.csv")
df.drop(columns=["id"],inplace=True)
df.drop(columns=["ICD-10"],inplace=True)
df.to_csv("data/cas_cancer_nc.csv", index=False)


# def strip_datasets(vega_lite_json_path, output_path):
#     with open(vega_lite_json_path, 'r') as f:
#         chart = json.load(f)

#     # Save datasets separately if needed
#     datasets = chart.pop("datasets", None)  # Remove 'datasets'
#     with open("DATA2Poster_json/dataset_only.json", "w") as f:
#         json.dump(datasets, f, indent=2)


#     # Save only the chart spec without datasets
#     with open(output_path, 'w') as f:
#         json.dump(chart, f, indent=2)

#     print("Chart specification saved to:", output_path)
#     if datasets:
#         print("Note: datasets section removed.")

# # Example usage
# strip_datasets("DATA2Poster_json/vega_lite_json_3.json", "DATA2Poster_json/chart_spec_only.json")



# # Create a multiline json
# record_dict = json.loads(df.to_json(orient = "records"))
# record_json = json.dumps(record_dict)

# with open(output_path, 'w') as f:
#     f.write(record_json)

# # def load_json(json_file):
# #                     with open(json_file, "r", encoding="utf-8") as fh:
# #                         return json.load(fh)


# # data_list = load_json('train_set.json')
# # docs = [item for item in data_list] 
# # print(docs[0])
# data = pd.read_csv('data/Sleep_health_and_lifestyle.csv')
# chart = alt.Chart(data).mark_bar().encode(
#         x=alt.X('Occupation:N', title='Occupation', sort='-y'),
#         y=alt.Y('Avg_Heart Rate:Q', title='Average Heart Rate'),
#         color=alt.Color('Occupation:N', scale=alt.Scale(domain=data['Occupation'].unique(), range=["#FFD7D1", "#D8E2DC", "#FFE5D9", "#FFD7BA", "#FEC89A"])),
#         tooltip=['Occupation:N', 'Avg_Heart Rate:Q']
#     ).properties(
#         title='Average Heart Rate by Occupation',
#         width=600,
#         height=400
#     ).configure_axis(
#         labelFontSize=12,
#         titleFontSize=14
#     ).configure_title(
#         fontSize=16,
#         anchor='start'
#     )
#     # Save the chart as a PNG file
# chart_json = chart.to_json()
# with open('data/occupation_heart_rate.json', 'w') as f:
#     f.write(chart_json)
# # chart.save('data/occupation_heart_rate.png')