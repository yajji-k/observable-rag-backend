import re


def clean_text(text: str):

    # Remove excessive newlines
    text = re.sub(r"\n+", "\n", text)

    # Remove excessive spaces
    text = re.sub(r"\s+", " ", text)

    # Remove standalone page numbers
    text = re.sub(r"\b\d+\b", "", text)

    # Fix common PDF ligature artifacts
    text = text.replace("ﬁ", "fi")
    text = text.replace("ﬂ", "fl")

    # Remove extra spaces again after cleanup
    text = re.sub(r"\s+", " ", text)

    return text.strip()