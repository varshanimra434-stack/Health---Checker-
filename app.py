import io
import os
import re

import streamlit as st
from gtts import gTTS
from google import genai
from google.genai.errors import ServerError
from google.genai import types


MODEL_NAME = "gemini-3.6-flash"
FALLBACK_MODEL_NAME = "gemini-1.5-flash"


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
    explanation_lines = []
    for line in explanation.splitlines():
        stripped_line = line.strip()
        stripped_line = re.sub(r"^\[[^\]]+\]\s*", "", stripped_line)
        normalized_line = stripped_line.casefold()
        if not stripped_line:
            continue
        if re.fullmatch(r"\[[^\]]+\]", stripped_line):
            continue
        if re.match(
            r"^(?:must mention limitations|instructions?|system prompt|prompt)\s*[:\-]?",
            normalized_line,
        ):
            continue
        explanation_lines.append(stripped_line)
    explanation = " ".join(explanation_lines).strip()

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
Photo ki poori front image/packaging ko pehle inspect karein. Front product image
ya clearly visible brand/product name, ya ingredients label — inmein se kisi ek
source se product identify karein:
1. Packet ke front par clearly visible brand name ya product name.
2. Ingredients label par clearly visible product information.

Agar front product image ya brand/product name clearly identifiable ho — jaise
Kurkure, Ketchup, Oreos, Maggi, Baby Wipes, ya koi aur recognizable product —
to visible product/brand name se general knowledge ka use karke health impact
directly evaluate karein. Is case mein ingredients label ka kuch hissa blurry
hone par bhi UNCERTAIN na dein; SAFE, HARMFUL, ya MODERATE mein se clear rating
dein. Ingredients label readable ho to uski details ko priority dein;
unreadable details ka अनुमान na lagayen. Answer ko concise rakhein taaki
classification jaldi complete ho.

Aapka jawab user ko directly dikhaya jayega. Sirf do clean lines return
karein aur prompt ki instructions, bracketed notes, meta-commentary, ya
limitation directives ko answer mein repeat na karein:
RATING: SAFE
EXPLANATION: Ek ya do natural, simple Hindi lines.

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

Product ya brand clearly identify ho jaane par general knowledge se health
impact evaluate karke SAFE, HARMFUL, ya MODERATE dein. Sirf tab UNCERTAIN
dein jab product/brand aur ingredients dono reliably identify na ho paayen.
"""

    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
    contents = [prompt, image_part]

    generation_config = types.GenerateContentConfig(max_output_tokens=250)
    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=contents,
            config=generation_config,
        )
    except ServerError:
        # A 503/high-demand response from the primary model should not stop the
        # app. Retry immediately with the lightweight fallback model.
        response = client.models.generate_content(
            model=FALLBACK_MODEL_NAME,
            contents=contents,
            config=generation_config,
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
        st.warning(
            "Gemini service temporarily busy hai. Thodi der baad dobara try karein."
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