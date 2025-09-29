from typing import Dict, List, Tuple
import json
import pandas as pd
import altair as alt
import vl_convert as vlc
from vegafusion.runtime import VegaFusionRuntime
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A3, landscape
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from json_sanitizer import parse_jsonish

def agent10_pdf_creator():
    filename = "data_name_summary.pdf"
    filedir = "pdf"
    filepath = os.path.join(filedir, filename)
    background_image = f"figure/poster_background_v3.png"
    # Create canvas
    c = canvas.Canvas(filepath, pagesize=landscape(A3))
    width, height = landscape(A3)
    c.drawImage(background_image, 0, 0, width=width, height=height)

    # Generate introduction
    text_introduction = """This poster analyzes employment patterns across Pacific Island countries by gender and sector. Section one shows employment distribution and gender gaps among these countries. Section two explores employment shares and gender differences across economic sectors. Section three highlights sectoral and gender employment trends in Solomon Islands and Tokelau specifically.
"""

    # Generate conclusion 
    text_conclusion = """Solomon Islands leads employment with nearly 100,000 workers, far surpassing others, reflecting its larger economy and population. Male employment dominates across most Pacific Islands, except in Palau where gender balance is closer, indicating varied gender dynamics regionally.Agriculture and Fishing overwhelmingly dominate employment, highlighting reliance on primary industries, while construction and energy sectors also hold significant shares. Gender disparities are stark, with males prevailing in labor-intensive sectors and females more present in retail and education, underscoring entrenched occupational gender roles.In Solomon Islands, male employment vastly exceeds female in Agriculture and Fishing, reflecting traditional roles, while sectors like Education show more gender balance. Tokelau exhibits strong female dominance in Education and Social Work, contrasting with male-led Construction, revealing persistent sectoral gender segregation across islands."""
    
    # Generate Title from conclusion
    title = "How Do Gender and Sector Shape Employment Patterns Across Pacific Island Nations?"
    
    # Title
    p_title = Paragraph(title, ParagraphStyle(name='title', fontSize=30, fontName='Helvetica-Bold',leading=22, textColor="#2c2a32"))
    p_title.wrapOn(c, width,20)
    p_title.drawOn(c, 10, height-50)
    
    # Introduction content
    p_in = Paragraph(text_introduction, ParagraphStyle(name="introduction", fontSize=14, fontName='Helvetica', leading=14, alignment=4, textColor="#2c2a32"))
    p_in.wrapOn(c, width-600, 60)
    p_in.drawOn(c, 20, height-170)
    
     
    # Add section 1 descriptions

    p_desc = Paragraph("Solomon Islands leads employment numbers, with males consistently outnumbering females across countries.", ParagraphStyle(name="insight", fontSize=16, fontName='Helvetica-Bold',leading=14, alignment=4, textColor="#2c2a32"))
    p_desc.wrapOn(c, 550, 25)
    p_desc.drawOn(c, 50, height-221)

    # Add section 2 descriptions 
    p_desc = Paragraph("Agriculture dominates employment, while sectors like Construction and Accommodation show significant gendered participation.", ParagraphStyle(name="insight", fontSize=16, fontName='Helvetica-Bold',leading=14, alignment=4, textColor="#2c2a32"))
    p_desc.wrapOn(c, 550, 25)
    p_desc.drawOn(c, 50, height-540)
    # Add section 3 descriptions 
    p_desc = Paragraph("Female employment dominates in Tokelau sectors, while males lead in Solomon Islands, showing varied gender dynamics.", ParagraphStyle(name="insight", fontSize=16, fontName='Helvetica-Bold',leading=14, alignment=4, textColor="#2c2a32"))
    p_desc.wrapOn(c, 500, 20)
    p_desc.drawOn(c, 685, height-92)
    # Add section 1 images
    c.drawImage(f"data2poster_chart/image1.png", 20, height-505, width=280, height=280)
    c.drawImage(f"data2poster_chart/image2.png", 330, height-505, width=290, height=280)

    # Add section 2 images
    c.drawImage(f"data2poster_chart/image3.png", 20, height-815, width=280, height=260)
    c.drawImage(f"data2poster_chart/image4.png", 330, height-815, width=290, height=260)

    # Add section 3 images
    c.drawImage(f"data2poster_chart/image5.png", 670, height-340, width=500, height=230)
    c.drawImage(f"data2poster_chart/image6.png", 670, height-600, width=500, height=230)

    # Conclusion content
    p_con = Paragraph(text_conclusion, ParagraphStyle(name="conclusion", fontSize=14, fontName='Helvetica', leading=14, alignment=4, textColor="#2c2a32"))
    p_con.wrapOn(c, 520, 140)
    p_con.drawOn(c, 653, height-817)

    c.save()
agent10_pdf_creator()



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




