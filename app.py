import streamlit as st
import fitz  # PyMuPDF
import os
import base64
from mimetypes import guess_type
from io import BytesIO, StringIO
import pandas as pd
import pdfplumber
from pydantic import (
    BaseModel,
    BeforeValidator,
    PlainSerializer,
    InstanceOf,
    WithJsonSchema,
    model_validator,
)
import instructor
from openai import AzureOpenAI
from typing import Any, Annotated, List
from dotenv import load_dotenv
load_dotenv()

# 1. Helpers for image -> data URL
def local_image_to_data_url(image_bytes: bytes, fmt: str = "png") -> str:
    mime_type = f"image/{fmt}"
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"

# 2. Helper to parse markdown table into DataFrame
def md_to_df(data: Any) -> Any:
    if isinstance(data, str):
        df = (
            pd.read_csv(StringIO(data), sep="|", index_col=1)
              .dropna(axis=1, how="all")
              .iloc[1:]
        )
        return df.applymap(lambda v: v.strip() if isinstance(v, str) else v)
    return data

# 3. Pydantic model for tabular output
MarkdownDataFrame = Annotated[
    InstanceOf[pd.DataFrame],
    BeforeValidator(md_to_df),
    PlainSerializer(lambda df: df.to_markdown()),
    WithJsonSchema({
        "type": "string",
        "description": "Markdown representation of the table"
    }),
]

class Table(BaseModel):
    caption: str
    dataframe: MarkdownDataFrame

    @model_validator(mode="after")
    def check_df(cls, vals):
        if vals.dataframe.empty:
            raise ValueError("No table extracted!")
        return vals

azure_cfg = st.secrets["azure"]
api_key     = azure_cfg["api_key"]
api_version = azure_cfg["api_version"]
azure_endpoint    = azure_cfg["endpoint"]
# 4. Set up the LLM client via Instructor
llm = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=azure_endpoint
)
client = instructor.from_openai(llm, mode=instructor.Mode.TOOLS)

# 5. Extraction function using Instructor
def extract_with_instructor(data_url: str, model: str = "gpt-4o",prompt_text= "Extract the all tables present") -> Table:
    return client.chat.completions.create(
        model=model,
        max_tokens=4000,
        response_model=Table,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": "Extract the all tables present"},
                ],
            }
        ],
    )
# def clean_header(raw_header):
#     """
#     Take a list of header strings (some may be empty or duplicated)
#     and return a new list where:
#       1. empty strings become "col_#"
#       2. duplicates get suffixes: name, name_1, name_2, …
#     """
#     counts = {}
#     cleaned = []
#     for idx, h in enumerate(raw_header):
#         h0 = h.strip() or f"col_{idx}"
#         cnt = counts.get(h0, 0)
#         counts[h0] = cnt + 1
#         if cnt:
#             cleaned.append(f"{h0}_{cnt}")
#         else:
#             cleaned.append(h0)
#     return cleaned

def clean_header(raw_header):
    """
    Take a list of header values (some may be None or duplicates)
    and return a list where:
      1. non-str or empty entries become "col_<index>"
      2. duplicate names get suffixes: name, name_1, name_2, …
    """
    counts = {}
    cleaned = []
    for idx, h in enumerate(raw_header):
        # Safely coerce non-str to empty and strip whitespace
        h_str = h.strip() if isinstance(h, str) else ""
        # Fallback name if header cell is empty
        base = h_str or f"col_{idx}"
        # Count duplicates
        cnt = counts.get(base, 0)
        counts[base] = cnt + 1

        # Append suffix if needed
        if cnt:
            cleaned.append(f"{base}_{cnt}")
        else:
            cleaned.append(base)
    return cleaned


# 6. Streamlit UI
def main():
    st.title("📄 PDF Table Extractor")
    prompt_text = st.text_area(
        "Extraction prompt",
        value="Extract all tables present on this page as a markdown table.",
        help="Customize the instruction you send to the LLM."
    )

    method = st.selectbox(
        "Select extraction method",
        ["PDFPlumber (pure-Python)", "Instructor (LLM + vision)"]
    )

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if not uploaded_file:
        st.info("Please upload a PDF to begin.")
        return

    pdf_bytes = uploaded_file.read()

    if method == "PDFPlumber (pure-Python)":
        st.subheader("Results via PDFPlumber")
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                st.markdown(f"**Table {tables}:**")
                if not tables:
                    st.write(f"Page {i}: no tables found.")
                    continue

                st.write(f"Page {i}:")
                for tbl_idx, tbl in enumerate(tables, start=1):
                    raw_header = tbl[0]
                    header = clean_header(raw_header)  # <— clean duplicates
                    data_rows = tbl[1:]
                    df = pd.DataFrame(data_rows, columns=header)  # now safe!

                    st.markdown(f"**Table {tbl_idx}:**")
                    st.dataframe(df)

    else:
        st.subheader("Results via Instructor + GPT-4o Vision")
        # convert and process each page
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for i in range(len(doc)):
            page = doc[i]
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")
            data_url = local_image_to_data_url(img_bytes, fmt="png")

            with st.spinner(f"Processing page {i+1}..."):
                try:
                    table = extract_with_instructor(data_url=data_url,prompt_text=prompt_text)
                    st.markdown(f"**Page {i+1}**")
                    st.dataframe(table.dataframe)
                except Exception as e:
                    st.error(f"Page {i+1} failed: {e}")

if __name__ == "__main__":
    main()
