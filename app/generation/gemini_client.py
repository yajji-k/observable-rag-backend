from google import genai

from app.config import GEMINI_API_KEY


# Initialize Gemini client once during application startup
client = genai.Client(
    api_key=GEMINI_API_KEY
)


def generate_response(prompt: str):

    # Send prompt to Gemini model
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text