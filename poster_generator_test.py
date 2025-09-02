import json
import os
import re
import random
from PIL import Image 
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A3, landscape
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from text_generator import introduction, conclusion, improve_title




def create_pdf(data_name,insight_list,section_insight_list,chart_id_list,chart_url_list, column_1, column_2, entity_1,entity_2,section_header_list,openai_key):

    filename = f"{data_name}_summary.pdf"
    filedir = "pdf"
    filepath = os.path.join(filedir, filename)
    background_image = f"figure/poster_background_v3.png"
    # Create canvas
    c = canvas.Canvas(filepath, pagesize=landscape(A3))
    width, height = landscape(A3)
    c.drawImage(background_image, 0, 0, width=width, height=height)

    # Generate introduction
    text_introduction = introduction(chart_url_list , section_header_list,openai_key)

    # Generate conclusion 
    text_conclusion = conclusion(insight_list ,text_introduction, column_1, column_2, entity_1, entity_2, openai_key)

    # Generate Title from conclusion
    title = improve_title(text_introduction, text_conclusion, openai_key)
    
    # # Title
    p_title = Paragraph(title, ParagraphStyle(name='title', fontSize=30, fontName='Helvetica-Bold',leading=22, textColor="#2c2a32"))
    p_title.wrapOn(c, width,20)
    p_title.drawOn(c, 10, height-50)
    
    # # Introduction content
    # p_introduction = Paragraph("Introduction", ParagraphStyle(name='introduction', fontSize=26, fontName='Helvetica-Bold', textColor=text_color))
    # p_introduction.wrapOn(c, width/2,200)
    # p_introduction.drawOn(c, 30, height-115)
    p_in = Paragraph(text_introduction, ParagraphStyle(name="introduction", fontSize=14, fontName='Helvetica', leading=14, alignment=4, textColor="#2c2a32"))
    p_in.wrapOn(c, width-600, 20)
    p_in.drawOn(c, 30, height-180)
    
     
    # Add section 1 descriptions

    p_desc = Paragraph(section_insight_list[0], ParagraphStyle(name="insight", fontSize=16, fontName='Helvetica-Bold',leading=14, alignment=4, textColor="#2c2a32"))
    p_desc.wrapOn(c, 550, 20)
    p_desc.drawOn(c, 50, height-225)

    # Add section 2 descriptions 
    p_desc = Paragraph(section_insight_list[1], ParagraphStyle(name="insight", fontSize=16, fontName='Helvetica-Bold',leading=14, alignment=4, textColor="#2c2a32"))
    p_desc.wrapOn(c, 550, 20)
    p_desc.drawOn(c, 50, height-545)
    # Add section 3 descriptions 
    p_desc = Paragraph(section_insight_list[2], ParagraphStyle(name="insight", fontSize=16, fontName='Helvetica-Bold',leading=14, alignment=4, textColor="#2c2a32"))
    p_desc.wrapOn(c, 500, 20)
    p_desc.drawOn(c, 685, height-90)
    # Add section 1 images
    c.drawImage(f"data2poster_chart/image{chart_id_list[0]}.png", 20, height-518, width=280, height=280)
    c.drawImage(f"data2poster_chart/image{chart_id_list[1]}.png", 330, height-518, width=290, height=280)

    # Add section 2 images
    c.drawImage(f"data2poster_chart/image{chart_id_list[2]}.png", 20, height-825, width=280, height=260)
    c.drawImage(f"data2poster_chart/image{chart_id_list[3]}.png", 330, height-825, width=290, height=260)

    # Add section 3 images
    c.drawImage(f"data2poster_chart/image{chart_id_list[4]}.png", 650, height-385, width=520, height=280)
    c.drawImage(f"data2poster_chart/image{chart_id_list[5]}.png", 650, height-675, width=520, height=280)

    # # Conclusion content
    # p_conclusion = Paragraph("Conclusion", ParagraphStyle(name='conclusion', fontSize=26, fontName='Helvetica-Bold', textColor=text_color))
    # p_conclusion.wrapOn(c, width/2, 200)
    # p_conclusion.drawOn(c, 30, height-725)
    p_con = Paragraph(text_conclusion, ParagraphStyle(name="conclusion", fontSize=14, fontName='Helvetica', leading=14, alignment=4, textColor="#2c2a32"))
    p_con.wrapOn(c, 525, 200)
    p_con.drawOn(c, 650, height-820)

    c.save()