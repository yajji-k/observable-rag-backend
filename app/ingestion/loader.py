from pypdf import PdfReader


def load_pdf(file_path: str):

    # Initialize PDF reader
    reader = PdfReader(file_path)

    # Store extracted text
    text = ""

    # Extract text from each page
    for page in reader.pages:

        page_text = page.extract_text()

        # Skip empty pages safely
        if page_text:

            text += page_text

    return text