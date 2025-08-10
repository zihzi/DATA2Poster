import json
import os
import re
import random
from PIL import Image 
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A3, landscape
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
# from text_generator import introduction, conclusion, improve_title




def create_pdf():
    
    filename = "_summary.pdf"
    filedir = "pdf"
    filepath = os.path.join(filedir, filename)
    background_image = f"figure/poster_background_v3.png"
    # Create canvas
    c = canvas.Canvas(filepath, pagesize=landscape(A3))
    width, height = landscape(A3)
    c.drawImage(background_image, 0, 0, width=width, height=height)

    
    # # Title
    p_title = Paragraph("The process can be extended to more depths by wrapping the second-level step in a recursive or while-loop with a depth counter.", ParagraphStyle(name='title', fontSize=22, fontName='Helvetica-Bold',leading=20, textColor="#2c2a32"))
    p_title.wrapOn(c, width,10)
    p_title.drawOn(c, 10, height-50)
    
    # # Introduction content
    # p_introduction = Paragraph("Introduction", ParagraphStyle(name='introduction', fontSize=26, fontName='Helvetica-Bold', textColor=text_color))
    # p_introduction.wrapOn(c, width/2,200)
    # p_introduction.drawOn(c, 30, height-115)
    p_in = Paragraph("The process can be extended to more depths by wrapping the second-level step in a recursive or while-loop with a depth counter.The process can be extended to more depths by wrapping the second-level step in a recursive or while-loop with a depth counter.The process can be extended to more depths by wrapping the second-level step in a recursive or while-loop with a depth counter.", ParagraphStyle(name="introdcution", fontSize=14, fontName='Helvetica', leading=18, alignment=4, textColor="#2c2a32"))
    p_in.wrapOn(c, width-600, 20)
    p_in.drawOn(c, 30, height-190)
    
     
    # Add images 1 and descriptions
    c.drawImage(f"data2poster_img/image1.png", 20 , height-520, width=280, height=280)
    p_desc = Paragraph("The process can be extended to more depths by wrapping the second-level step in a recursive or while-loop with a depth counter.", ParagraphStyle(name="insight", fontSize=14, fontName='Helvetica',leading=18, alignment=4, textColor="#2c2a32"))
    p_desc.wrapOn(c, 550, 20)
    p_desc.drawOn(c,50, height-240)

    # Add images 2 and descriptions
    c.drawImage(f"data2poster_img/image2.png", 330, height-520, width=290, height=280)
    p_desc = Paragraph("The process can be extended to more depths by wrapping the second-level step in a recursive or while-loop with a depth counter.", ParagraphStyle(name="insight", fontSize=14, fontName='Helvetica',leading=18, alignment=4, textColor="#2c2a32"))
    p_desc.wrapOn(c, 550, 20)
    p_desc.drawOn(c, 50, height-560)

    # Add images 3 and descriptions
    c.drawImage(f"data2poster_img/image3.png", 20, height-825, width=280, height=260)
    p_desc = Paragraph("The process can be extended to more depths by wrapping the second-level step in a recursive or while-loop with a depth counter.", ParagraphStyle(name="insight", fontSize=14, fontName='Helvetica',leading=18, alignment=4, textColor="#2c2a32"))
    p_desc.wrapOn(c, 500, 20)
    p_desc.drawOn(c, 685, height-100)
    c.drawImage(f"data2poster_img/image4.png", 330, height-825, width=290, height=260)
    c.drawImage(f"data2poster_img/image5.png", 650, height-385, width=520, height=280)
    c.drawImage(f"data2poster_img/image6.png", 650, height-675, width=520, height=280)

    # # Conclusion content
    # p_conclusion = Paragraph("Conclusion", ParagraphStyle(name='conclusion', fontSize=26, fontName='Helvetica-Bold', textColor=text_color))
    # p_conclusion.wrapOn(c, width/2, 200)
    # p_conclusion.drawOn(c, 30, height-725)
    p_con = Paragraph("The process can be extended to more depths by wrapping the second-level step in a recursive or while-loop with a depth counter.The process can be extended to more depths by wrapping the second-level step in a recursive or while-loop with a depth counter.The process can be extended to more depths by wrapping the second-level step in a recursive or while-loop with a depth counter.", ParagraphStyle(name="conclusion", fontSize=14, fontName='Helvetica', leading=18, alignment=4, textColor="#2c2a32"))
    p_con.wrapOn(c, 450, 120)
    p_con.drawOn(c, 680, height-830)

    c.save()

create_pdf()