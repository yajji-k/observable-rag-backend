from app.ingestion.loader import (
    load_pdf
)


# Load PDF content
text = load_pdf(
    "data/TA_wrkbk.pdf"
)


# Validate extracted text
print(f"{type(text)=}")

# Print preview of extracted content
print(text[:2000])