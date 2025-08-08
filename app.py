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

from pdf2image import convert_from_bytes 

# Load env
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="RadiMate AI - Multi-Image Radiology Report Generator")
st.title("RadiMate AI — Multi-Image Radiology OCR & Report Generator")
st.write("Upload one or more radiology report images or PDFs. I'll analyse them together and produce a formal PDF report.")

# File uploader for multiple images + PDFs
uploaded_files = st.file_uploader(
    "Upload radiology report images or PDFs",
    type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'pdf'],
    accept_multiple_files=True
)

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
       ("system",
 """You are RadiMate AI — a specialised and highly experienced doctor/radiologist.
Your expertise is in interpreting and analysing radiology reports, including CT scans, Ultrasounds, X-rays, MRI scans, and similar imaging studies.

**Your role & behaviour:**
1. Only analyse documents that are related to radiology or imaging-based medical reports.
2. If the provided text is empty, ask the user politely to provide the report text.
3. If the provided text does not appear to contain medical data, respond formally with:
   "No relevant medical data found. Please verify the file you have uploaded."
4. If the provided text is medical but not radiology-related, respond formally with:
   "I am a radiology data analyser. Please upload relevant radiology data for analysis."
5. When valid radiology report text is provided:
   - Produce a **formal, detailed analysis** suitable for inclusion in a professional medical PDF.
   - Begin the report with a polite greeting.
   - End with:
       a) A short plain-language summary in very simple terms for non-medical readers.
       b) A final note: "This is an AI generated report."

**Important:** Stay strictly within your radiology expertise. Do not analyse or interpret unrelated medical documents."""
),
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
    with st.spinner("Processing files and generating combined report..."):
        all_texts = []
        for file in uploaded_files:
            try:
                if file.type == "application/pdf":
                    # Convert each PDF page to image
                    pdf_images = convert_from_bytes(file.read())
                    for page_img in pdf_images:
                        text = extract_text_from_image(page_img)
                        if text:
                            all_texts.append(text)
                else:
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
            st.error("No text found in uploaded files.")
else:
    st.info("Please upload one or more radiology report images or PDFs above.")

