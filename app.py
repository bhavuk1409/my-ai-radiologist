from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

from PIL import Image, ImageOps, ImageFilter
import pytesseract
from fpdf import FPDF
import io

# Load env
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# If on Windows and tesseract not on PATH, set:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(page_title="RadiMate AI - Multi-Image Radiology Report Generator")
st.title("RadiMate AI — Multi-Image Radiology OCR & Report Generator")
st.write("Upload one or more radiology report images (JPG/PNG/TIFF/BMP). I'll analyse them together and produce a formal PDF report.")

# File uploader for multiple images
uploaded_files = st.file_uploader(
    "Upload radiology report images",
    type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
    accept_multiple_files=True
)

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a radiologist named RadiMate AI. You are very good at reading radiology reports (CT, Ultrasound, X-ray, etc.). "
         "For the provided report text, produce a formal analysis suitable for a medical PDF. At the start greet the user. "
         "If the user did not provide text, ask them to provide it. If text was provided, analyze the report and produce a formal report text. "
         "At the end of the report, include a short plain-language summary (very simple words). "
         "Finally append one line: 'This is an AI generated report'."),
        ("user", "report_text:{report_text}")
    ]
)

# LLM and chain
llm = ChatGroq(model="llama3-70b-8192")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# OCR helpers
def preprocess_image_for_ocr(pil_img: Image.Image) -> Image.Image:
    img = pil_img.convert("L")
    img = ImageOps.autocontrast(img)
    w, h = img.size
    if max(w, h) < 2000:
        factor = int(2000 / max(w, h)) + 1
        img = img.resize((w*factor, h*factor), Image.LANCZOS)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    return img

def extract_text_from_image(pil_img: Image.Image) -> str:
    img = preprocess_image_for_ocr(pil_img)
    text = pytesseract.image_to_string(img, lang='eng')
    return text.strip()

def create_pdf_from_text(title: str, body: str) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=16, style='B')
    pdf.cell(0, 10, title, ln=True, align='C')
    pdf.ln(4)
    pdf.set_font("Arial", size=11)
    lines = body.split("\n")
    for line in lines:
        for chunk in [line[i:i+100] for i in range(0, len(line), 100)]:
            pdf.multi_cell(0, 7, chunk)
    return pdf.output(dest='S').encode('latin-1')

# Main flow
if uploaded_files:
    with st.spinner("Processing images and generating combined report..."):
        all_texts = []
        for file in uploaded_files:
            try:
                img = Image.open(io.BytesIO(file.read()))
                text = extract_text_from_image(img)
                if text:
                    all_texts.append(text)
            except Exception as e:
                st.error(f"OCR failed for {file.name}: {e}")

        combined_text = "\n\n".join(all_texts)

        if combined_text.strip():
            try:
                llm_output = chain.invoke({'report_text': combined_text})
            except Exception as e:
                st.error(f"LLM analysis failed: {e}")
                llm_output = ""

            if llm_output:
                st.success("Combined report generated successfully.")
                st.subheader("AI Report")
                st.write(llm_output)

                title = "COMPREHENSIVE MEDICAL IMAGING AND TREATMENT PROGRESS REPORT"
                pdf_bytes = create_pdf_from_text(title, llm_output)

                st.download_button(
                    label="Download PDF Report",
                    data=pdf_bytes,
                    file_name="RadiMate_AI_report.pdf",
                    mime="application/pdf"
                )
            else:
                st.error("No report generated from LLM.")
        else:
            st.error("No text found in uploaded images.")
else:
    st.info("Please upload one or more radiology report images above.")
