import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

def test_judge():
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    prompt = "Say hello"
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_judge()
