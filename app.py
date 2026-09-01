import io
import os
import re

import streamlit as st
from gtts import gTTS
from google import genai
from google.genai import types


MODEL_NAME = "gemini-2.5-flash"
FALLBACK_MODEL_NAME = "gemini-2.5-flash"


def get_gemini_api_key() -> str | None:
    """Read the key from Streamlit secrets first, then from the environment."""
    try:
        gemini_key = st.secrets.get("GEMINI_API_KEY")
        google_key = st.secrets.get("GOOGLE_API_KEY")
    except FileNotFoundError:
        gemini_key = None
        google_key = None

    return (
        gemini_key
        or google_key
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )


def parse_result(raw_text: str) -> tuple[str, str]:
    """Extract the model's rating and explanation from its requested format."""
    text = (raw_text or "").strip()
    # Models sometimes wrap labels in Markdown, such as **RATING:** SAFE.
    clean_text = re.sub(r"[*_`]", "", text)

    rating_match = re.search(
        r"\bRATING\s*:\s*(SAFE|HARMFUL|MODERATE|UNCERTAIN)\b",
        clean_text,
        flags=re.IGNORECASE,
    )
    rating = rating_match.group(1).upper() if rating_match else "UNCERTAIN"

    explanation_match = re.search(
        r"\bEXPLANATION\s*:\s*(.*)",
        clean_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    explanation = explanation_match.group(1).strip() if explanation_match else clean_text
    explanation = explanation.strip(" -:\n")

    if not explanation:
        explanation = (
            "Ingredients ki photo se reliable analysis nahi ho paaya. "
            "Label ko clearly dikhakar dobara photo lein."
        )

    return rating, explanation


def analyze_product(client: genai.Client, image_bytes: bytes) -> tuple[str, str]:
    prompt = """
Aap food, cosmetic, ya medicine product ko image se identify karke uske
ingredients aur health impact ka general screening karne wale assistant hain.
Photo ki poori front packaging ko pehle inspect karein. Front brand name ya
product name, ya ingredients label — inmein se kisi ek source se product
identify karein:
1. Packet ke front par clearly visible brand name ya product name.
2. Ingredients label par clearly visible product information.

Agar front brand name ya product name clearly visible ho — jaise Kurkure,
Ketchup, Oreos, Maggi, Baby Wipes, ya koi aur recognizable product — to
visible product/brand name se general knowledge ka use karke health impact ko
directly evaluate karein. Is case mein ingredients label ka kuch hissa blurry
hone par bhi UNCERTAIN na dein; SAFE, HARMFUL, ya MODERATE mein se clear rating
dein. Ingredients label readable ho to uski details ko priority dein;
unreadable details ka अनुमान na lagayen. Answer ko concise rakhein taaki
classification jaldi complete ho.

Bilkul is format mein jawab dein:
RATING: SAFE
EXPLANATION: Hindi mein 1-2 simple lines.

RATING ke liye sirf SAFE, HARMFUL, MODERATE, ya UNCERTAIN use karein:
- SAFE: visible ingredients mein koi obvious concern nahi mila, lekin ise
  medical guarantee na batayen.
- HARMFUL: visible label par koi clearly concerning ingredient, warning, ya
  allergy risk dikh raha hai.
- MODERATE: product generally harmful nahi hai, lekin sugar, salt, fat,
  additives, fragrance, ya regular use/consumption se related health concern
  ho sakta hai.
- UNCERTAIN: label/ingredients readable nahi hain, product identify nahi hua,
  aur front par brand/product name bhi clearly identifiable nahi hai.

Hamesha सीमाएँ बताएं: yeh general information hai, diagnosis ya medical
advice nahi. Product clearly identify ho jaane par ingredient details ke liye
general product knowledge use karein; dono sources se product identify na ho to
RATING: UNCERTAIN dein.
"""

    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
    contents = [prompt, image_part]

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(max_output_tokens=250),
        )
    except Exception:
        # Retry the same verified model once if the first request is transiently
        # unavailable.
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(max_output_tokens=250),
        )

    return parse_result(response.text)


def make_hindi_audio(text: str) -> bytes:
    audio_buffer = io.BytesIO()
    gTTS(text=text, lang="hi").write_to_fp(audio_buffer)
    return audio_buffer.getvalue()


st.set_page_config(page_title="Auto Product Health Checker", page_icon="🧴")
st.title("Auto Product Health Checker")
st.caption("Product label ka general, image-based screening — medical advice nahi.")

api_key = get_gemini_api_key()
if not api_key:
    st.error(
        "GEMINI_API_KEY ya GOOGLE_API_KEY set nahi hai. Local run ke liye "
        "environment variable set karein ya Streamlit secrets mein inmein se "
        "koi ek add karein."
    )
    st.stop()

client = genai.Client(api_key=get_gemini_api_key())
img_input = st.camera_input("Product ka ingredients label camera ke saamne rakhein")

if img_input is not None:
    image_bytes = img_input.getvalue()

    try:
        with st.spinner("Ingredients aur warnings analyze ho rahe hain..."):
            rating, explanation = analyze_product(client, image_bytes)
    except Exception as e:
        st.error(
            "Image analysis fail ho gaya. API key, internet connection, aur "
            "Gemini model access check karke dobara try karein."
        )
        st.write(e)
        st.stop()

    if rating == "HARMFUL":
        st.error(f"Rating: {rating}")
    elif rating == "SAFE":
        st.success(f"Rating: {rating}")
    else:
        st.warning(f"Rating: {rating}")

    st.write(explanation)
    st.info(
        "Important: photo-based result final medical or allergy advice nahi hai. "
        "Severe allergy, pregnancy, ya health condition mein product label aur "
        "health professional ki advice follow karein."
    )

    try:
        with st.spinner("Hindi voice taiyar ho rahi hai..."):
            audio_bytes = make_hindi_audio(f"Rating: {rating}. {explanation}")
        st.audio(audio_bytes, format="audio/mp3", autoplay=True)
    except Exception:
        st.caption("Voice output available nahi ho paaya; text result upar diya gaya hai.")