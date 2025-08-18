from typing import Dict, List, Tuple
import json
import pandas as pd
import altair as alt
import vl_convert as vlc
from vegafusion.runtime import VegaFusionRuntime



# chart = alt.Chart.from_dict(vlspec)
# df = chart.transformed_data()
# print(df)

def transformed_datasets(vl_spec: dict, dataset_names: List[str] | None = None) -> Tuple[Dict[str, pd.DataFrame], list]:
    """Evaluate transforms and return requested datasets as DataFrames.

    Parameters
    ----------
    vl_spec : dict
        Vega-Lite v5 spec.
    dataset_names : list[str] | None
        Names of datasets in the compiled Vega spec to extract. If None, all
        top-level datasets are extracted.

    Returns
    -------
    (dfs, warnings)
        dfs: mapping of dataset name -> pandas.DataFrame
        warnings: list of VegaFusion warnings (e.g., row limits)
    """
    # 1) Compile Vega-Lite -> Vega (VegaFusion operates on Vega specs)
    vega_spec = vlc.vegalite_to_vega(json.dumps(vl_spec))
    # vega_spec = json.loads(vega_json)

    # 2) Pick datasets. If none provided, grab all top-level data names
    if dataset_names is None:
        dataset_names = [d.get("name") for d in vega_spec.get("data", []) if isinstance(d, dict) and "name" in d]

    # 3) Evaluate transforms for these datasets
    rt = VegaFusionRuntime()
    tables, warnings = rt.pre_transform_datasets(
        vega_spec,
        datasets=dataset_names,
        dataset_format="pandas",  # return DataFrames directly
    )

    # 4) Assemble mapping name -> DataFrame
    frames: Dict[str, pd.DataFrame] = {name: df for name, df in zip(dataset_names, tables)}
    return frames, warnings


# ──────────────────────────────────────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Minimal demo spec using your pattern (replace with your real spec)
    df = pd.read_csv('data/Occupation_by_gender.csv')
    with open('data2poster_json/vlspec1_8.json', 'r') as f:
        spec = json.load(f)
    spec["data"].pop("url", None)
    spec["data"]["values"] = df.to_dict(orient="records")
    print(spec)


    print("\n— VegaFusion datasets —")
    dfs, warns = transformed_datasets(spec)  # or pass specific dataset names
    for name, frame in dfs.items():
        print(f"dataset: {name} → shape={frame.shape}")
        print(frame.head())
    if warns:
        print("warnings:", warns)
# def trans_data(df):


#     # Assuming "df" is the input DataFrame


#     # Grouping and aggregating


#     result_df = df.groupby("Pacific Island Countries", as_index=False)["Number of employed persons"].sum().rename(columns={"Number of employed persons": "total_employed"})


#     return result_df

# trans_df = trans_data(df)

# trans_df.to_csv('DATA2Poster_df/transformed_df.csv', index=False)

# new_df = pd.read_csv('DATA2Poster_df/transformed_df.csv')

# # # Drop the 'Model' column
# # df["Year"] = df["Year"].astype(int)

# # # Save the cleaned CSV
# # df.to_csv("data/movies_record.csv", index=False)
# from PIL import Image

# # 開啟原始圖片
# img = Image.open("DATA2Poster_img/base/base_img_1.png").convert("RGBA")

# # 取得像素資料
# datas = img.getdata()

# new_data = []
# for item in datas:
#     # 假設白色是 (255, 255, 255)
#     if item[0] > 250 and item[1] > 250 and item[2] > 250:
#         # 將白色變成透明
#         new_data.append((255, 255, 255, 0))
#     else:
#         new_data.append(item)

# # 替換像素並儲存新圖片
# img.putdata(new_data)
# img.save("output_no_white.png", "PNG")




