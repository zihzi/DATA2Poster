import json
import os
import re
import random
from PIL import Image 
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A3, landscape
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from text_generator import introduction, conclusion



def create_pdf(dataset, q_for_nl4DV, title, insight_list,  openai_key):
    num = random.choice([1, 2, 3])
    if num == 1:
        text_color = "#feece9"
    elif num == 2:
        text_color = "#fff5e4"
    else:
        text_color = "#efecfd"
    filename = f"{dataset}_summary.pdf"
    filedir = "pdf"
    filepath = os.path.join(filedir, filename)
    background_image = f"figure/poster_background_{num}.png"
    # Create canvas
    c = canvas.Canvas(filepath, pagesize=landscape(A3))
    width, height = landscape(A3)
    c.drawImage(background_image, 0, 0, width=width, height=height)

    # Generate introduction
    text_introduction = introduction(title, q_for_nl4DV , openai_key)

    # Generate conclusion 
    text_conclusion = conclusion(title, insight_list, text_introduction, openai_key)
    
    # Title
    p_title = Paragraph(title, ParagraphStyle(name='title', fontSize=32, fontName='Helvetica-Bold',leading=34, textColor=text_color))
    p_title.wrapOn(c, width/2,200)
    p_title.drawOn(c, width/2-10, height-200)
    
    # Introduction content
    p_introduction = Paragraph("Introduction", ParagraphStyle(name='introduction', fontSize=26, fontName='Helvetica-Bold', textColor=text_color))
    p_introduction.wrapOn(c, width/2,200)
    p_introduction.drawOn(c, 30, height-115)
    p_in = Paragraph(text_introduction, ParagraphStyle(name="introdcution", fontSize=14, fontName='Helvetica', leading=18, alignment=4, textColor=text_color))
    p_in.wrapOn(c, width-755, 150)
    p_in.drawOn(c, 30, height-250)
    
     
    # Add images 1 and descriptions
    c.drawImage("data2poster_img/image_1.png", 55, height-580, width=280, height=210)
    p_desc = Paragraph(insight_list[0], ParagraphStyle(name="insight", fontSize=14, fontName='Helvetica',leading=18, alignment=4, textColor="#2c2a32"))
    p_desc.wrapOn(c, width/4, 100)
    p_desc.drawOn(c, 50, height-670)

    # Add images 2 and descriptions
    c.drawImage("data2poster_img/image_2.png", 455, height-580, width=280, height=210)
    p_desc = Paragraph(insight_list[1], ParagraphStyle(name="insight", fontSize=14, fontName='Helvetica',leading=18, alignment=4, textColor="#2c2a32"))
    p_desc.wrapOn(c, width-900, 30)
    p_desc.drawOn(c, 450, height-670)

    # Add images 3 and descriptions
    c.drawImage("data2poster_img/image_3.png", 845, height-580, width=280, height=210)
    p_desc = Paragraph(insight_list[2], ParagraphStyle(name="insight", fontSize=14, fontName='Helvetica',leading=18, alignment=4, textColor="#2c2a32"))
    p_desc.wrapOn(c, width-900, 30)
    p_desc.drawOn(c, 840, height-670)

    # Conclusion content
    p_conclusion = Paragraph("Conclusion", ParagraphStyle(name='conclusion', fontSize=26, fontName='Helvetica-Bold', textColor=text_color))
    p_conclusion.wrapOn(c, width/2, 200)
    p_conclusion.drawOn(c, 30, height-725)
    p_con = Paragraph(text_conclusion, ParagraphStyle(name="conclusion", fontSize=14, fontName='Helvetica', leading=18, alignment=4, textColor=text_color))
    p_con.wrapOn(c, width-70, 100)
    p_con.drawOn(c, 30, height-820)

    c.save()

    