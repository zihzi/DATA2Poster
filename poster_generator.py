import json
import re
from PIL import Image 
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Paragraph, Frame
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from text_generator import introduction, description, conclusion
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

def create_pdf(dataset, topic, title, description_list, summary,  openai_key):
    filename = f"{dataset}_summary.pdf"
    background_image = "background.jpg"
    # Create canvas
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    c.drawImage(background_image, 0, 0, width=width, height=height)

    # Generate introduction
    text_introduction = introduction(title, topic, openai_key)

    # Generate description for 3 charts
    description_list = []
    for i in range(1,4):
        with open(f"data2poster_json/vega_lite_json_{i}.json") as f:
            vega_lite_json = json.load(f)
        text_description = description(vega_lite_json, openai_key)
        description_list.append(text_description)    
    
    # Generate conclusion
    final_distribution = description_list
    text_conclusion = conclusion(final_distribution, summary, openai_key)

    # Title
    tilte_style = ParagraphStyle(name='title', fontSize=20, fontName='Helvetica-Bold',leading=20, textColor="#545454")
    p_title = Paragraph(title, tilte_style)
    p_title.wrapOn(c, width-180, 100)
    p_title.drawOn(c, width/8, height-150)
    
    # Introduction content
    p_in = Paragraph(text_introduction, ParagraphStyle(name="introdcution", fontSize=14, fontName='Helvetica',leading=14, textColor="#2c2a32"))
    p_in.wrapOn(c, width-100, 100)
    p_in.drawOn(c, 50, height-320)
    
    def add_image_to_box(image_path, x, y, box_width=200, box_height=150):
       # Load and resize image
       img = Image.open(image_path)
       aspect = img.width / img.height
       
       # Calculate dimensions to fit in box while maintaining aspect ratio
       if aspect > box_width/box_height:
           img_width = box_width - 20  # Padding
           img_height = img_width / aspect
       else:
           img_height = box_height
           img_width = img_height * aspect
           
       # Draw white background box
       c.setFillColor(colors.white)
       c.rect(x, y, box_width, box_height, fill=1)
       
       # Draw image
       c.drawImage(image_path, x+10, y+10, width=img_width, height=img_height)
       
       return box_height

   # Add graphics with images
    graphic_boxes = [
       {"title": "Graphic 1", "image": "image_1.png", "x": 50, "y": height-500},
       {"title": "Graphic 2", "image": "image_2.png", "x": 50, "y": height-700},
       {"title": "Graphic 3", "image": "image_3.png", "x": 300, "y": height-500}
    ]
    i=0
    for box in graphic_boxes:
        box_height = add_image_to_box(box["image"], box["x"], box["y"])
        
        # Add description
        p_desc = Paragraph(description_list[i], ParagraphStyle(name="description", fontSize=12, fontName='Helvetica',leading=12, textColor="#2c2a32"))
        p_desc.wrapOn(c, width/3, 100)
        p_desc.drawOn(c, box["x"], box["y"]-30)
        i+=1


    # Conclusion content
    p_con = Paragraph(text_conclusion, ParagraphStyle(name="conclusion", fontSize=14, fontName='Helvetica',leading=14, textColor="#2c2a32"))
    p_con.wrapOn(c, width-100, 100)
    p_con.drawOn(c, 50, height-820)

    c.save()

# use llm to generate pdf code////////////////////////////////////////////////////////////////////////////////////
    # # Generate introduction
    # text_introduction = introduction(title, topic, openai_key)
    # # # Generate description for 3 charts
    # # description_list = []
    # # for i in range(1,4):
    # #     with open(f"data2poster_json/vega_lite_json_{i}.json") as f:
    # #         vega_lite_json = json.load(f)
    # #     text_description = description(vega_lite_json, openai_key)
    # #     description_list.append(text_description)    
    # image_path = ["image_1.png", "image_2.png", "image_3.png"]
    # # Generate conclusion
    # final_distribution = description_list
    # text_conclusion = conclusion(final_distribution, summary, openai_key)

    # code_template = \
    #     f"""
    #     from reportlab.pdfgen import canvas
    #     from reportlab.lib.pagesizes import A4
    #     from reportlab.platypus import Paragraph
    #     from reportlab.lib.styles import ParagraphStyle

    #     pdf_file = f"{dataset}_summary.pdf"
    #     can = canvas.Canvas(pdf_file)
    #     # Set pdf background color
    #     A4_height = A4[1]
    #     A4_width = A4[0]    
    #     left_padding = 20
    #     width = A4_width
    #     height = A4_height
    #     can.setFillColor("#E6F0EF")
    #     can.rect(0, 0, width, height, fill=1)

    #     # Set Line and rectangle above the title
    #     can.setFillColor("#B6D3D0")
    #     can.rect(left_padding, 731, 600-2*left_padding, 92, fill=1, stroke=0)
    #     can.setStrokeColor("#135D66")
    #     can.setLineWidth(5)
    #     can.line(left_padding, 820, 600-left_padding, 820)

    #     # Set title
    #     tilte_style = ParagraphStyle(name='Title', fontSize=16, fontName='Helvetica-Bold', textColor="#135D66")
    #     p_title = Paragraph({title}, tilte_style)
    #     p_title.wrapOn(can, 560, 100)
    #     p_title.drawOn(can, 30, 770)
    
    #     # Set introdution
    #     can.setFillColor("#135D66")
    #     can.rect(left_padding+2, 720, 110, 20, fill=1)
    #     can.setLineWidth(2)
    #     can.line(left_padding, 730, 600-left_padding, 730) 
    #     can.setFont("Helvetica-Bold", 16)
    #     can.setFillColor("white")
    #     can.drawString(left_padding+8, 725, "Introduction")
    #     # introdution content
    #     text_introduction = {text_introduction}
    #     p_in = Paragraph(text_introduction)
    #     p_in.wrapOn(can, 520, 100)
    #     p_in.drawOn(can, 40, 650)
    #     # Set 3 descriptions and 3 chart images 

    #     <stub> # only modify this section   
        
    #     # Set conclusion
    #     can.setFillColor("#135D66")
    #     can.rect(left_padding, 130, 110, 20, fill=1)
    #     can.setLineWidth(2)
    #     can.line(left_padding, 140, 600-left_padding, 140) 
    #     can.setFont("Helvetica-Bold", 16)
    #     can.setFillColor("white")
    #     can.drawString(left_padding+12, 135, "Conclusion")
    #     # conclusion content
    #     text_conclusion = {text_conclusion}
    #     p_con = Paragraph(text_conclusion)
    #     p_con.wrapOn(can, 520, 60)
    #     p_con.drawOn(can, 40, 50)

    #     # Set Line at the bottom
    #     can.setStrokeColor("#135D66")
    #     can.setLineWidth(5)
    #     can.line(left_padding, 20, 600-left_padding, 20)
        
    #     can.showPage()
    #     can.save()
    #     """
    # prompt = PromptTemplate(
    #         template="""You are an expert in Python programming and PDF generation using the ReportLab library. 
    #                     Your task is to GENERATE CODE BY MODIFYING THE SPECIFIED PARTS OF THE TEMPLATE BELOW.\n\n {code_template} 
    #                     The template is about generate an A4-sized PDF report containing the provided charts and text content. 
    #                     The PDF include three sections:
    #                     1. Title Section: A centered title at the top of the page.
    #                     2. Charts and Corresponding Descriptions Section : Visual elements (e.g., bar charts, line charts) and descriptive paragraphs at the middle of the page.
    #                     3. Conclusion Section: A conclusion at the bottom of the page.
    #                     The Title Section and the Conclusion Section are fixed and already done, 
    #                     all you have to do is set up the Charts and Corresponding Descriptions Section based on the following input.
    #                     The chart size should be clearly visible and not too small.
    #                     Make sure descriptive paragraphs placed near the charts. Ensure the text is well-aligned and does not overlap with the charts.
    #                     Input:
    #                     Charts: {image_path}
    #                     Charts Description: {description_list}
    #                     Please do not add any extra prose or explanation to your response. The reponse MUST BE THE COMPLETE EXECUTABLE CODE TEMPLATE with your modification.\n\n.
    #                     The FINAL COMPLETED CODE TEMPLATE above is ...
    #                     """,
    #         input_variables=["code_template", "image_path", "description_list"]
    #     )       
    # llm = ChatOpenAI(model_name='gpt-4o-mini', api_key = openai_key)
    # gpt4_image_chain = prompt | llm 
    # response = gpt4_image_chain.invoke(input= {"code_template":code_template, "image_path":image_path, "description_list":description_list})
    # code = response.content
    # """Preprocess code to remove any preamble and explanation text"""

    # code = code.replace("<imports>", "")
    # code = code.replace("<stub>", "")
    # code = code.replace("<transforms>", "")

    # # remove all text after chart = plot(data)
    # if "```" in code:
    #     pattern = r"```(?:\w+\n)?([\s\S]+?)```"
    #     matches = re.findall(pattern, code)
    #     if matches:
    #         code = matches[0]

    # if "from" in code:
    #     # return only text after the first import statement
    #     index = code.find("from")
    #     if index != -1:
    #         code = code[index:]
    # code = code.replace("```", "")
    # return code


# old code ///////////////////////////////////////////////////////////////////////////////////////////////////
  # pdf_file = f"{dataset}_summary.pdf"
    # can = canvas.Canvas(pdf_file)
    # # Set pdf background color
    # A4_height = A4[1]
    # A4_width = A4[0]    
    # left_padding = 20
    # width = A4_width
    # height = A4_height
    # can.setFillColor("#E6F0EF")
    # can.rect(0, 0, width, height, fill=1)

    # # Set Line and rectangle above the title
    # can.setFillColor("#B6D3D0")
    # can.rect(left_padding, 731, 600-2*left_padding, 92, fill=1, stroke=0)
    # can.setStrokeColor("#135D66")
    # can.setLineWidth(5)
    # can.line(left_padding, 820, 600-left_padding, 820)

    # # Set title
    # tilte_style = ParagraphStyle(name='Title', fontSize=16, fontName='Helvetica-Bold', textColor="#135D66")
    # p_title = Paragraph(title, tilte_style)
    # p_title.wrapOn(can, 560, 100)
    # p_title.drawOn(can, 30, 770)
   
    # # Set introdution
    # can.setFillColor("#135D66")
    # can.rect(left_padding+2, 720, 110, 20, fill=1)
    # can.setLineWidth(2)
    # can.line(left_padding, 730, 600-left_padding, 730) 
    # can.setFont("Helvetica-Bold", 16)
    # can.setFillColor("white")
    # can.drawString(left_padding+8, 725, "Introduction")
    # # introdution content
    # text_introduction = introduction(title, topic, openai_key)
    # p_in = Paragraph(text_introduction)
    # p_in.wrapOn(can, 520, 100)
    # p_in.drawOn(can, 40, 650)
    # #  x,y for 4 images
    # x_start = [0,25, 420, 25, 420]
    # y_start = [0,420, 420, 170, 170]


    # # x,y for 4 descriptions
    # x_start_desc = [190, 230, 190, 230]
    # y_start_desc = [540, 430, 310, 180]

    # # x,y for 2 images
    # # x_start = [40, 40]
    # # y_start = [470, 230]

    # # x,y for 2 descriptions
    # # x_start_desc = [40, 40]
    # # y_start_desc = [400, 160]

    # # x,y for 1 images
    # # x_start = [40]
    # # y_start = [470]

    # # x,y for 1 descriptions
    # # x_start_desc = [40]
    # # y_start_desc = [400]
    
    # # Set description
    # # my_json_list = []
    # description_list = ""
    # for i in range(1,4):
    #     with open(f"data2poster_json/vega_lite_json_{i}.json") as f:
    #         vega_lite_json = json.load(f)
    #         # my_json_list.append(vega_lite_json)
    #     text_description = description(vega_lite_json, openai_key)
    #     description_list = description_list + f"{i}.)" + text_description + f"\n"
    # p = Paragraph(description_list)
    # p.wrapOn(can, 180, 100)
    # p.drawOn(can, 190, 300)

    # final_distribution = p




    # for i in range(1,4):
    #     image_path = f'image_{i}.png'
    #     image = Image.open(image_path)
    #     image = image.resize((400,300))
    #     image.save(f"data2poster_img/image_{i}.png")
    #     can.drawImage(f"data2poster_img/image_{i}.png", x_start[i], y_start[i],width = 150,height=220)
    #     # size for 2 images
    #     # can.drawImage(f"image_{i}.png", x_start[i], y_start[i],width=520,height=160)

    # # Set conclusion
    # can.setFillColor("#135D66")
    # can.rect(left_padding, 130, 110, 20, fill=1)
    # can.setLineWidth(2)
    # can.line(left_padding, 140, 600-left_padding, 140) 
    # can.setFont("Helvetica-Bold", 16)
    # can.setFillColor("white")
    # can.drawString(left_padding+12, 135, "Conclusion")
    # # conclusion content
    # text_conclusion = conclusion(final_distribution, summary, openai_key)
    # p_con = Paragraph(text_conclusion)
    # p_con.wrapOn(can, 520, 60)
    # p_con.drawOn(can, 40, 50)

    # # Set Line at the bottom
    # can.setStrokeColor("#135D66")
    # can.setLineWidth(5)
    # can.line(left_padding, 20, 600-left_padding, 20)
    
    # can.showPage()
    # can.save()