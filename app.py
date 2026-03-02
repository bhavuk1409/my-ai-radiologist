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

# ----------------------------
# App Config
# ----------------------------
st.set_page_config(page_title="RadiMate AI - Multi-File Medical Report Analyzer", layout="centered")
st.title("RadiMate AI — Radiology + Blood Report OCR & PDF Generator")
st.write("Upload one or more radiology reports (CT/MRI/X-ray/USG) and/or blood/lab reports (CBC/LFT/KFT/etc.). "
         "I’ll extract text (OCR), analyse all documents together, and generate a formal PDF report.")

# ----------------------------
# Env / API key
# ----------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.warning("GROQ_API_KEY not found in environment. Add it to your .env file as GROQ_API_KEY=...")

# (Optional) If Tesseract is not in PATH on macOS, set it explicitly:
# pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

# ----------------------------
# File uploader (multi)
# ----------------------------
uploaded_files = st.file_uploader(
    "Upload report images or PDFs",
    type=["png", "jpg", "jpeg", "bmp", "tiff", "pdf"],
    accept_multiple_files=True
)

# ----------------------------
# Prompt template (UPDATED)
# ----------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """You are RadiMate AI — a specialised and highly experienced medical report analyser.

Your expertise includes:
• Radiology imaging reports (CT, MRI, Ultrasound, X-ray, PET, Mammography, etc.)
• Blood and laboratory investigation reports (CBC, LFT, KFT, Lipid Profile, Thyroid, Blood Sugar, Hormone tests, etc.)

----------------------------------------
ROLE & BEHAVIOUR RULES
----------------------------------------

1. You must ONLY analyse:
   - Radiology imaging reports
   - Blood or laboratory investigation reports

2. If the provided text is empty:
   Politely request the user to upload a valid medical report.

3. If the provided text contains no medical data:
   Respond formally:
   "No relevant medical data found. Please verify the file you have uploaded."

4. If the provided text is medical but NOT radiology or blood/lab related:
   Respond formally:
   "I am specialised in analysing radiology and blood investigation reports. Please upload relevant medical data for analysis."

5. Only proceed with analysis if the content clearly relates to:
   - CT, MRI, Ultrasound, X-ray, PET scans, Mammography
   - Blood reports or laboratory test results

----------------------------------------
WHEN VALID RADIOLOGY REPORT IS PROVIDED:
----------------------------------------

- Begin with a polite formal greeting.
- Provide a professional and structured medical analysis.
- Interpret findings clearly.
- Mention normal vs abnormal findings.
- Highlight severity if indicated.

----------------------------------------
WHEN VALID BLOOD REPORT IS PROVIDED:
----------------------------------------

- Begin with a polite formal greeting.
- Analyse each major parameter:
    • Mention test name
    • Patient value
    • Normal reference range (if available in text)
    • Whether it is Low / Normal / High
- Highlight clinically important abnormalities.
- Explain possible meaning (non-diagnostic).
- Avoid prescribing medicines or exact doses.

----------------------------------------
ALWAYS INCLUDE:

1. A section titled:
   **"Full Summary (Simple Words)"**
   - Explain findings in everyday language.
   - Make it understandable for non-medical readers.
   - Mention overall health condition.

2. A section titled:
   **"Advice"**
   - Provide general non-prescriptive suggestions.
   - Example: follow-up tests, lifestyle modification, consult specialist.
   - Do NOT provide medication dosage or prescriptions.

----------------------------------------
IMPORTANT RESTRICTIONS:
----------------------------------------

- Stay strictly within your medical analysis expertise.
- Do not fabricate values.
- Do not diagnose definitively.
- Do not provide emergency or surgical decisions.
- Maintain a formal and professional tone in analysis.
- Keep "Full Summary" very simple and clear.

End every report with:

"This is an AI generated report and should not replace professional medical consultation."
"""),
        ("user", "report_text:\n{report_text}")
    ]
)

# ----------------------------
# LLM chain
# ----------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=groq_api_key,   # explicit is safer than relying on env in some setups
    temperature=0.2
)
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# ----------------------------
# OCR helpers
# ----------------------------
def preprocess_image_for_ocr(pil_img: Image.Image) -> Image.Image:
    """Make OCR more reliable by cleaning + upscaling."""
    img = pil_img.convert("L")
    img = ImageOps.autocontrast(img)

    w, h = img.size
    if max(w, h) < 2000:
        factor = int(2000 / max(w, h)) + 1
        img = img.resize((w * factor, h * factor), Image.LANCZOS)

    img = img.filter(ImageFilter.MedianFilter(size=3))
    return img


def extract_text_from_image(pil_img: Image.Image) -> str:
    img = preprocess_image_for_ocr(pil_img)
    text = pytesseract.image_to_string(img, lang="eng")
    return text.strip()


def create_pdf_from_text(title: str, body: str) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", size=14, style="B")
    pdf.multi_cell(0, 8, title, align="C")
    pdf.ln(4)

    pdf.set_font("Arial", size=11)

    # Safer line wrapping for long lines
    for line in body.split("\n"):
        if not line.strip():
            pdf.ln(3)
            continue
        pdf.multi_cell(0, 6, line)

    return pdf.output(dest="S").encode("latin-1")


# ----------------------------
# Main flow
# ----------------------------
if uploaded_files:
    with st.spinner("Processing files (OCR) and generating combined report..."):
        all_texts = []
        per_file_debug = []

        for file in uploaded_files:
            try:
                file_bytes = file.read()
                if not file_bytes:
                    per_file_debug.append(f"{file.name}: empty file bytes.")
                    continue

                if file.type == "application/pdf":
                    # Convert each PDF page to image (OCR)
                    pages = convert_from_bytes(file_bytes)
                    for idx, page_img in enumerate(pages, start=1):
                        text = extract_text_from_image(page_img)
                        if text:
                            all_texts.append(f"\n--- {file.name} (page {idx}) ---\n{text}\n")
                        else:
                            per_file_debug.append(f"{file.name} (page {idx}): no text extracted.")
                else:
                    img = Image.open(io.BytesIO(file_bytes))
                    text = extract_text_from_image(img)
                    if text:
                        all_texts.append(f"\n--- {file.name} ---\n{text}\n")
                    else:
                        per_file_debug.append(f"{file.name}: no text extracted.")

            except Exception as e:
                st.error(f"OCR failed for {file.name}: {e}")

        combined_text = "\n".join(all_texts).strip()

        if not combined_text:
            st.error("No text found in uploaded files.")
            if per_file_debug:
                with st.expander("Debug details (OCR)"):
                    st.write("\n".join(per_file_debug))
        else:
            # (Optional) show extracted text
            with st.expander("Show extracted text (OCR)"):
                st.text_area("Combined OCR Text", combined_text, height=300)

            try:
                llm_output = chain.invoke({"report_text": combined_text})
            except Exception as e:
                st.error(f"LLM analysis failed: {e}")
                llm_output = ""

            if not llm_output.strip():
                st.error("No report generated from LLM.")
            else:
                st.success("Combined report generated successfully.")
                st.subheader("AI Report")
                st.write(llm_output)

                title = "COMPREHENSIVE RADIOLOGY / BLOOD REPORT ANALYSIS"
                pdf_bytes = create_pdf_from_text(title, llm_output)

                st.download_button(
                    label="Download PDF Report",
                    data=pdf_bytes,
                    file_name="RadiMate_AI_report.pdf",
                    mime="application/pdf"
                )
else:
    st.info("Please upload one or more radiology report images / PDFs or blood report images / PDFs above.")
