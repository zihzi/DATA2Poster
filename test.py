import pandas as pd

df = pd.read_csv('data/police_killings.csv')
df['avg_povertyrate'] = df['avg_povertyrate'].astype(float)
df.to_csv('data/police_killings.csv', index=False)
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




