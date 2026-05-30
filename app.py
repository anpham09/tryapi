import streamlit as st
from PIL import Image
from io import BytesIO
import base64
import requests
import json
import re


st.set_page_config(
    page_title="TruthLens",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 TruthLens")
st.write(
    "Upload an image. TruthLens will analyze whether it appears AI-generated or non-AI, "
    "then show an educational explanation based on the strongest visual clue."
)


MODULE_INFO = {
    "Body": """Module 1: Body

The human body is one of the hardest things for AI to get right. Start with the hands — AI loves sneaking in an extra finger or two. From there, look at the teeth, the ears, and the hair. The skin may look too smooth, with no pores or texture. Eyes can look glassy or fake, and makeup may look painted on.""",

    "Text": """Module 2: Text

AI cannot spell. Any words you see in an AI image — on a sign, a t-shirt, a menu, or a book cover — may look fine from far away but turn into gibberish up close. Letters can blend together, flip backward, or become symbols that only look like letters.""",

    "Reflections": """Module 3: Reflections

Shadows and reflections follow rules — AI does not always know them. Shadows may point in different directions, mirrors may show impossible reflections, and water reflections may not match the real scene.""",

    "Backgrounds": """Module 4: Backgrounds

The further from the center of an AI image, the weirder it gets. Objects near the edges may blend together, people may melt into backgrounds, and buildings or doors may look geometrically impossible.""",

    "Textures": """Module 5: Textures

AI textures can look good in a thumbnail but fall apart when zoomed in. Fabric patterns may not follow the body, seams may disappear, zippers may stop halfway, and everything may look too new with no scratches, stains, or wear.""",

    "Metadata / Technical": """Module 6: Metadata / Technical

Real photos often carry hidden information such as camera model, lens, time, or location. AI images often have missing or suspicious metadata. Some AI tools may also leave behind software markers or recognizable style signatures.""",

    "Objects": """Module 7: Objects

Small details are where AI gives up fastest. Look at the glasses — the arms that hook behind the ears often fade out or disappear. Jewelry may sink into the skin, necklaces may float, and watch faces may have melted numbers, wrong times, or blurry details."""
}


def get_api_key():
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return None


def image_to_base64(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return image, image_base64


def extract_json(text):
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    return None


def analyze_image_with_anthropic(image_base64, api_key):
    prompt = """
You are TruthLens, an educational AI image analysis assistant.

Analyze the uploaded image and decide whether it appears AI-generated or non-AI/real.

Important:
- You are not perfect. If uncertain, still choose the closest result but explain the uncertainty.
- Choose exactly one module from this list:
Body, Text, Reflections, Backgrounds, Textures, Metadata / Technical, Objects.
- The module should be the strongest visual clue category.
- Return JSON only. Do not include markdown.

Use this exact JSON structure:
{
  "result": "AI Generated" or "Non-AI / Real",
  "confidence": number from 0 to 100,
  "module": "Body" or "Text" or "Reflections" or "Backgrounds" or "Textures" or "Metadata / Technical" or "Objects",
  "reason": "2 to 4 short sentences explaining the visual evidence."
}
"""

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    data = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 700,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    }

    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=data,
        timeout=60
    )

    if response.status_code != 200:
        return None, f"API request failed: {response.status_code}\n\n{response.text}"

    response_json = response.json()
    text = response_json["content"][0]["text"]

    result_json = extract_json(text)

    if result_json is None:
        return None, f"Could not parse API response:\n\n{text}"

    return result_json, None


api_key = get_api_key()

if api_key is None:
    st.error(
        "API key not found. Create `.streamlit/secrets.toml` and add:\n\n"
        'ANTHROPIC_API_KEY = "your_api_key_here"'
    )
    st.stop()


uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image, image_base64 = image_to_base64(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_column_width=True
    )

    if st.button("Analyze Image"):
        with st.spinner("Analyzing image..."):
            result_data, error = analyze_image_with_anthropic(image_base64, api_key)

        if error:
            st.error(error)
            st.stop()

        result = result_data.get("result", "Unknown")
        confidence = result_data.get("confidence", "Unknown")
        module = result_data.get("module", "Unknown")
        reason = result_data.get("reason", "No explanation provided.")

        st.subheader("Detection Result")

        if "AI Generated" in result:
            st.error("AI Generated")
        else:
            st.success("Non-AI / Real")

        st.metric("Confidence", f"{confidence}%")
        st.write(f"**Strongest clue category:** {module}")
        st.write(f"**Reason:** {reason}")

        st.subheader("Educational Explanation")

        if module in MODULE_INFO:
            st.info(MODULE_INFO[module])
        else:
            st.warning("No matching educational module was found.")
            st.write("The API returned this module:", module)

        with st.expander("Raw API JSON"):
            st.json(result_data)

else:
    st.info("Upload an image to begin.")
