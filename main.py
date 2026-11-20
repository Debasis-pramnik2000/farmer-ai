import asyncio
import base64
import io
import json
import os
import re
import time
import traceback
import hashlib
from datetime import date

from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError
from groq import Groq

load_dotenv()

app = FastAPI(title="Smart Agri AI Engine (Hybrid Gemini + Groq)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://farmer-bondhu.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GROQ_KEY = os.environ.get("GROQ_API_KEY", "").strip()

if not GEMINI_KEY:
    print("❌ Warning: GEMINI_API_KEY is not set in backend/.env!")
else:
    print("✅ GEMINI_API_KEY loaded successfully.")

if not GROQ_KEY:
    print("⚠️ Warning: GROQ_API_KEY is not set. Vision & Text endpoints will use Gemini fallback.")
else:
    print("✅ GROQ_API_KEY loaded successfully.")

# ⚡ SPEED: request timeouts so a slow/hung model never blocks the user for long
gemini_client = (
    genai.Client(
        api_key=GEMINI_KEY,
        http_options=types.HttpOptions(timeout=35000),  # 35s (ms)
    )
    if GEMINI_KEY
    else None
)

# ⚡ SPEED: max_retries=0 -> failed Groq model falls to next model instantly
groq_client = Groq(api_key=GROQ_KEY, timeout=25.0, max_retries=0) if GROQ_KEY else None

# Active Gemini model priority list (Best for Vision & Multimodal)
GEMINI_MODELS_PRIORITY = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]

# Active Groq Production Text Models
GROQ_MODELS_PRIORITY = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
]

# Active Groq Vision Models (Multimodal)
GROQ_VISION_MODELS_PRIORITY = [
    "qwen/qwen3.8-27b",
]

# Master Commodity Catalog for Daily Mandi Prices
COMMODITY_CATALOG = [
    {"name": "Paddy (Common)", "variety": "Swarna / MTU-7029", "category": "Paddy", "base_price": 23.00},
    {"name": "Paddy (Fine)", "variety": "Miniket / Gobindobhog", "category": "Paddy", "base_price": 34.50},
    {"name": "Paddy (Basmati)", "variety": "1121 / Pusa", "category": "Paddy", "base_price": 46.00},
    {"name": "Potato", "variety": "Jyoti / Chandramukhi", "category": "Vegetable", "base_price": 18.50},
    {"name": "Onion", "variety": "Red Nasik", "category": "Vegetable", "base_price": 28.00},
    {"name": "Tomato", "variety": "Hybrid Local", "category": "Vegetable", "base_price": 24.00},
    {"name": "Brinjal", "variety": "Round / Long Purple", "category": "Vegetable", "base_price": 26.50},
    {"name": "Pointed Gourd (Parwal)", "variety": "Desi Green", "category": "Vegetable", "base_price": 32.00},
    {"name": "Cauliflower", "variety": "Snowball", "category": "Vegetable", "base_price": 22.00},
    {"name": "Cabbage", "variety": "Green Express", "category": "Vegetable", "base_price": 16.00},
    {"name": "Okra (Bhindi)", "variety": "Hybrid Green", "category": "Vegetable", "base_price": 30.00},
    {"name": "Green Chilli", "variety": "Bullet / Guntur", "category": "Vegetable", "base_price": 48.00},
]

# ⚡ SPEED: max image side sent to Vision APIs (phone photos are 3-12 MB; this cuts upload + inference time)
MAX_IMAGE_SIDE = 1024


def clean_json_response(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if match:
        cleaned = match.group(1).strip()
    return json.loads(cleaned)


def prepare_image(image_bytes: bytes) -> bytes:
    """
    Validates the image AND shrinks it (max 1024px, JPEG q85).
    Smaller payload = much faster upload + faster vision response.
    Raises ValueError if the image is invalid/corrupted.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        try:
            # Fast JPEG decode at reduced size (no-op for other formats)
            img.draft("RGB", (MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        except Exception:
            pass
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except Exception as e:
        raise ValueError("Invalid or corrupted image file.") from e


def _gemini_one_pass(contents, response_json=False, max_output_tokens=1024):
    """
    Executes Gemini models with zero thinking overhead and disabled AFC
    to suppress SDK warnings and ensure ultra-fast responses.
    """
    if not gemini_client:
        raise RuntimeError("Gemini API key is not configured.")

    config_params = {
        "temperature": 0.3,
        "max_output_tokens": max_output_tokens,
        "thinking_config": types.ThinkingConfig(thinking_budget=0),
        "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
    }

    if response_json:
        config_params["response_mime_type"] = "application/json"

    gen_config = types.GenerateContentConfig(**config_params)
    last_error = None

    for model_name in GEMINI_MODELS_PRIORITY:
        for attempt in range(2):
            try:
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=gen_config,
                )
                print(f"✅ [Gemini] Success using model: {model_name}")
                return response

            except (ServerError, ClientError) as err:
                err_msg = str(err)

                # ✅ FIX: 504 / DEADLINE_EXCEEDED -> don't retry same slow model, switch immediately
                if any(x in err_msg for x in ["504", "DEADLINE_EXCEEDED"]):
                    last_error = err
                    print(f"⏱️ [Gemini] {model_name} timed out (504). Switching model.")
                    break

                elif any(x in err_msg for x in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "INTERNAL"]):
                    last_error = err
                    if attempt == 0:
                        print(f"⏳ [Gemini] {model_name} busy. Retrying in 0.3s...")
                        time.sleep(0.3)
                        continue
                    else:
                        print(f"↪️ [Gemini] {model_name} still busy. Switching model.")
                        break

                elif any(x in err_msg for x in ["404", "NOT_FOUND", "INVALID_ARGUMENT"]):
                    if last_error is None:
                        last_error = err
                    print(f"⚠️ [Gemini] {model_name} unavailable (404). Skipping.")
                    break

                else:
                    raise err

            except Exception as err:
                # ✅ FIX: client-side timeouts / network hiccups -> switch model instead of crashing
                last_error = err
                low = f"{type(err).__name__} {err}".lower()
                if any(x in low for x in ["timeout", "timed out", "deadline", "connect", "network"]):
                    print(f"⏱️ [Gemini] {model_name} timeout/network error ({type(err).__name__}). Switching model.")
                    break
                raise err

    if last_error is None:
        raise RuntimeError("Gemini request failed: no model available.")
    raise last_error


# ✅ FIX: when ALL Gemini models are overloaded (503/504 spike), wait briefly and
# run the whole model list again instead of failing immediately.
GEMINI_MAX_ROUNDS = 3
_TRANSIENT_MARKERS = [
    "503", "504", "429", "500",
    "unavailable", "deadline_exceeded", "resource_exhausted", "internal",
    "timeout", "timed out", "deadline", "connect", "network",
]


def execute_gemini_with_retry(contents, response_json=False, max_output_tokens=1024):
    last_err = None

    for round_no in range(GEMINI_MAX_ROUNDS):
        try:
            return _gemini_one_pass(contents, response_json, max_output_tokens)

        except Exception as err:
            last_err = err
            low = f"{type(err).__name__} {err}".lower()
            transient = any(m in low for m in _TRANSIENT_MARKERS)

            if not transient or round_no == GEMINI_MAX_ROUNDS - 1:
                raise

            wait = 1.5 * (round_no + 1)
            print(f"🔁 [Gemini] All models busy. Round {round_no + 1} failed, retrying in {wait}s...")
            time.sleep(wait)

    raise last_err


def friendly_error(err: Exception) -> str:
    low = f"{type(err).__name__} {err}".lower()
    if any(m in low for m in ["503", "504", "unavailable", "deadline", "timeout", "timed out", "resource_exhausted", "429"]):
        return "AI server is very busy right now. Please try again in a few seconds."
    return str(err)


def execute_groq_with_fallback(prompt: str, response_json: bool = False, max_tokens: int = 2048) -> str:
    """
    Executes ultra-fast Groq inference using active production models.
    Sets reasoning_effort='low' and sufficient max_tokens so gpt-oss
    models do not exhaust token budget before outputting valid JSON.
    """
    if groq_client:
        for model_name in GROQ_MODELS_PRIORITY:
            try:
                messages = []
                if response_json:
                    messages.append({
                        "role": "system",
                        "content": "You are a strict JSON API. Output ONLY valid raw JSON without markdown fences or extra text."
                    })
                messages.append({"role": "user", "content": prompt})

                kwargs = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": 0.2 if response_json else 0.3,
                    "max_tokens": max_tokens,
                }

                # Prevent reasoning models from burning tokens before JSON output
                if "gpt-oss" in model_name:
                    kwargs["reasoning_effort"] = "low"

                if response_json:
                    kwargs["response_format"] = {"type": "json_object"}

                completion = groq_client.chat.completions.create(**kwargs)
                content = completion.choices[0].message.content

                if content and content.strip():
                    print(f"⚡ [Groq] Ultra-fast reply using: {model_name}")
                    return content

            except Exception as err:
                print(f"⚠️ [Groq] {model_name} failed ({err}). Trying next...")
                continue

    # Seamless fallback to Gemini if Groq is unavailable
    print("↪️ Falling back to Gemini for text request...")
    response = execute_gemini_with_retry(
        contents=prompt,
        response_json=response_json,
        max_output_tokens=max_tokens,
    )
    return response.text


def execute_vision_with_fallback(
    prompt: str,
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    response_json: bool = True,
    max_tokens: int = 1024,
) -> str:
    """
    Executes Hybrid Vision inference:
    1. Tries Groq Multimodal Vision (qwen/qwen3.8-27b) first for ultra-fast response.
    2. Automatically falls back to Google Gemini Vision if Groq is unconfigured, rate-limited, or fails.
    """
    if groq_client:
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64_image}"

        for model_name in GROQ_VISION_MODELS_PRIORITY:
            try:
                messages = []
                if response_json:
                    messages.append({
                        "role": "system",
                        "content": "You are a strict JSON API. Output ONLY valid raw JSON without markdown fences or extra text.",
                    })

                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                })

                kwargs = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": 0.2 if response_json else 0.3,
                    "max_tokens": max_tokens,
                }

                if response_json:
                    kwargs["response_format"] = {"type": "json_object"}

                completion = groq_client.chat.completions.create(**kwargs)
                content = completion.choices[0].message.content

                if content and content.strip():
                    print(f"⚡ [Groq Vision] Success using model: {model_name}")
                    return content

            except Exception as err:
                print(f"⚠️ [Groq Vision] {model_name} failed ({err}). Falling back to Gemini Vision...")
                continue

    if not gemini_client:
        raise RuntimeError("Neither Groq Vision nor Gemini Vision API is configured.")

    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type=mime_type,
    )
    response = execute_gemini_with_retry(
        contents=[prompt, image_part],
        response_json=response_json,
        max_output_tokens=max_tokens,
    )
    return response.text


# ─────────────────────────────────────────────────────────────
# ✅ Soil validation via Hybrid Vision (fast single-word check)
# ─────────────────────────────────────────────────────────────
def is_soil_image(image_bytes: bytes, mime_type: str) -> bool:
    check_prompt = (
        "You are an image classifier.\n"
        "Look carefully at this image.\n"
        "Answer with EXACTLY one word, nothing else:\n"
        "- SOIL  -> if the image primarily shows agricultural soil, dirt, "
        "mud, earth, or a close-up of field topsoil.\n"
        "- NOT_SOIL -> if the image primarily shows plants/leaves/flowers, "
        "people, animals, sky, water, buildings, vehicles, food, objects, "
        "or any non-soil scene.\n"
        "If the image is a mixed scene where soil is only a small part, "
        "answer NOT_SOIL.\n"
        "Your one-word answer:"
    )

    try:
        raw_answer = execute_vision_with_fallback(
            prompt=check_prompt,
            image_bytes=image_bytes,
            mime_type=mime_type,
            response_json=False,
            max_tokens=16,
        )

        answer = (raw_answer or "").strip().upper().replace(" ", "_")
        print(f"🔍 Soil classifier answer: {answer!r}")

        if answer.startswith("SOIL") and "NOT_SOIL" not in answer:
            return True

        return False

    except Exception as e:
        print(f"⚠️ Soil validation failed, allowing request: {e}")
        return True


@app.post("/api/soil-test")
async def analyze_soil_and_recommend(
    file: UploadFile = File(...),
    location: str = Form("Purba Medinipur"),
    previous_crop: str = Form("Paddy"),
    water_availability: str = Form("Medium"),
    soil_feel: str = Form("Sticky"),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Only image files are permitted.",
        )

    if not groq_client and not gemini_client:
        raise HTTPException(
            status_code=500,
            detail="Neither Groq nor Gemini API is configured.",
        )

    try:
        raw_bytes = await file.read()

        # ⚡ Validate + resize in a worker thread (doesn't block the server)
        try:
            image_bytes = await asyncio.to_thread(prepare_image, raw_bytes)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid or corrupted image file.",
            )
        mime_type = "image/jpeg"

        prompt = f"""
You are an expert Indian Agronomist, Soil Scientist, and Crop Specialist.

Analyze this field soil image along with the following field parameters:

- Location / District: {location}, India
- Previously Harvested Crop: {previous_crop}
- Water Availability: {water_availability}
- Farmer Soil Texture Feel: {soil_feel}

Instructions:
1. Examine the visual soil texture, moisture sheen, color, and physical consistency.
2. Cross-reference with regional agricultural soil profiles for {location}.
3. Determine the primary soil type, physical condition, and inferred drainage/salinity traits.
4. Recommend top 2-3 most viable and profitable crops with ideal varieties for this soil condition.
5. Provide a per-bigha fertilizer schedule including basal dose, top dressing, and micronutrients.
6. Provide practical soil improvement tips.

Return ONLY a valid JSON object matching this structure:

{{
    "soil_type": "Alluvial Clay-Loam / Coastal Clay / Sandy Loam",
    "soil_condition": "Brief 1-2 sentence condition report",
    "drainage": "Moderate / Slow / Well-drained",
    "organic_matter_estimate": "Medium / Low / High",
    "recommended_crops": [
        {{
            "crop": "Crop Name",
            "variety": "Recommended Variety Name",
            "suitability_reason": "Why this crop thrives in this soil condition"
        }}
    ],
    "fertilizer_plan_per_bigha": {{
        "basal_dose": "Details of farmyard manure, DAP, and MOP during land preparation",
        "top_dressing": "Urea schedule and split timings",
        "micronutrients": "Zinc Sulphate / Boron advice if required"
    }},
    "soil_health_tips": "Practical tips to improve aeration, texture, and organic carbon"
}}
"""

        # ⚡ SPEED: run the soil-validation gate and the main analysis AT THE SAME TIME
        soil_task = asyncio.to_thread(is_soil_image, image_bytes, mime_type)
        analysis_task = asyncio.to_thread(
            execute_vision_with_fallback,
            prompt,
            image_bytes,
            mime_type,
            True,
            1024,
        )

        soil_ok, analysis_result = await asyncio.gather(
            soil_task, analysis_task, return_exceptions=True
        )

        # ✅ SOIL VALIDATION GATE
        if isinstance(soil_ok, Exception):
            print(f"⚠️ Soil validation failed, allowing request: {soil_ok}")
            soil_ok = True

        if not soil_ok:
            raise HTTPException(
                status_code=422,
                detail=(
                    "This does not look like a soil photo. "
                    "Please upload a clear close-up image of your field topsoil only."
                ),
            )

        if isinstance(analysis_result, Exception):
            raise analysis_result

        data = clean_json_response(analysis_result)

        return {
            "status": "success",
            "result": data,
        }

    except HTTPException:
        raise

    except Exception as e:
        print("❌ Soil Test Error Details:")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=friendly_error(e),
        )


@app.post("/api/predict")
async def predict_crop(
    file: UploadFile = File(...),
    lang: str = Form("en"),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Only image files are permitted.",
        )

    if not groq_client and not gemini_client:
        raise HTTPException(
            status_code=500,
            detail="Neither GROQ_API_KEY nor GEMINI_API_KEY is set in backend/.env",
        )

    try:
        raw_bytes = await file.read()

        # ⚡ Validate + resize in a worker thread
        try:
            image_bytes = await asyncio.to_thread(prepare_image, raw_bytes)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid or corrupted image file.",
            )
        mime_type = "image/jpeg"

        language_instruction = {
            "bn": "Bengali (বাংলা)",
            "hi": "Hindi (हिन्दी)",
            "en": "English",
        }.get(lang, "English")

        prompt = f"""
You are an expert plant pathologist and agronomist.

Analyze this crop leaf photo. The crop can be Paddy/Rice, Tomato,
Potato, Wheat, Cotton, Corn, or any other crop.

Language Requirement:
Provide ALL text values in: {language_instruction}.

Instructions:
1. Accurately identify the crop name (e.g., Rice/Paddy if it's rice).
2. Accurately identify the pathology/disease (e.g., Rice Blast, Brown Spot, Bacterial Leaf Blight, or Healthy Crop).
3. Estimate confidence percentage between 80% and 99%.
4. Describe visible pathological symptoms in 1-2 concise sentences.
5. Provide 2-3 specific chemical remedies/fungicides/bactericides with precise recommended dosage.
6. Provide actionable organic or preventive measures.

Return a valid JSON object matching this structure:

{{
    "crop": "Crop Name",
    "disease": "Disease Name or Healthy",
    "confidence": "94%",
    "symptoms": "Detailed visual symptoms observed on leaf",
    "medicines": [
        "Medicine 1 name with specific dosage",
        "Medicine 2 name with specific dosage"
    ],
    "organic_care": "Organic management and preventive practices"
}}
"""

        # ⚡ Blocking SDK call moved to a worker thread (Groq Vision primary + Gemini Vision fallback)
        raw_response = await asyncio.to_thread(
            execute_vision_with_fallback,
            prompt,
            image_bytes,
            mime_type,
            True,
            900,
        )

        data = clean_json_response(raw_response)

        return {
            "raw_class": f"{data.get('crop', 'Crop')}__{data.get('disease', 'Condition')}",
            "confidence": data.get("confidence", "92%"),
            "details": {
                "crop": data.get("crop", "Unknown Crop"),
                "disease": data.get("disease", "Unknown Condition"),
                "symptoms": data.get("symptoms", "No symptoms details available."),
                "medicines": data.get("medicines", ["Consult local agricultural officer."]),
                "organic_care": data.get("organic_care", "Maintain good field drainage and crop spacing."),
            },
            "heatmap": None,
        }

    except HTTPException:
        raise

    except Exception as e:
        print("❌ Backend Error Details:")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=friendly_error(e),
        )


class ChatRequest(BaseModel):
    message: str
    disease_context: str = ""
    lang: str = "en"


@app.post("/api/chat")
async def chat_agri(req: ChatRequest):
    if not groq_client and not gemini_client:
        return {"reply": "AI API keys are not configured."}

    try:
        language_instruction = "English"

        prompt = f"""
You are an expert agricultural scientist.

Current crop disease context:
{req.disease_context}

User's message:
{req.message}

Instruction:
1. Respond in {language_instruction}.
2. Keep the advice practical, scientific, concise (3-4 sentences), and easily understandable for farmers.
"""

        # ⚡ Blocking SDK call moved to a worker thread
        reply_text = await asyncio.to_thread(
            execute_groq_with_fallback,
            prompt,
            False,
            600,
        )

        return {"reply": reply_text.strip()}

    except Exception as e:
        return {"reply": f"Error: {str(e)}"}


@app.get("/")
def health_check():
    return {
        "status": "online",
        "engine": "Hybrid AI Engine (Groq Multimodal + Google Gemini Vision)",
        "capabilities": [
            "Multi-Crop Disease Detection",
            "Soil Image Analysis",
            "Ultra-Fast Agri Chatbot",
            "Profit & Mandi Analytics",
        ],
    }


# Load Agmarknet markets data safely
AGMARKNET_DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "agmarknet_markets.json",
)

if os.path.exists(AGMARKNET_DATA_PATH):
    with open(AGMARKNET_DATA_PATH, "r", encoding="utf-8") as f:
        AGMARKNET_MARKETS = json.load(f)
else:
    print(f"⚠️ Warning: {AGMARKNET_DATA_PATH} not found. Using default fallback markets.")
    AGMARKNET_MARKETS = {
        "West Bengal": {
            "Purba Medinipur": ["Tamluk APMC", "Contai APMC", "Egra APMC"],
            "Burdwan": ["Burdwan Main APMC", "Kalna APMC", "Katwa APMC"],
            "Hooghly": ["Tarakeswar APMC", "Sheoraphuli APMC"],
        }
    }


# ─────────────────────────────────────────────────────────────
# ✅ Region knowledge base for profit prediction
# ─────────────────────────────────────────────────────────────
REGION_CROP_HINTS = {
    "West Bengal": {
        "_default": (
            "Rice/paddy is the dominant kharif crop; potato, jute, "
            "mustard and vegetables dominate rabi. Coastal districts "
            "face salinity; western districts are red-laterite uplands."
        ),
        "Burdwan": (
            "Rice-potato belt; high groundwater irrigation; potato "
            "yields ~25-30 q/bigha; strong cold-storage network."
        ),
        "Purba Medinipur": (
            "Coastal alluvial, saline-prone; paddy + vegetables "
            "(pointed gourd, brinjal); betel vine is a cash crop."
        ),
        "Paschim Medinipur": (
            "Mixed red-laterite and alluvial; paddy, potato, "
            "mustard, vegetables; medium water availability."
        ),
        "Nadia": (
            "Alluvial, flood-prone; paddy, jute, vegetables, "
            "flowers; high irrigation coverage."
        ),
        "Murshidabad": (
            "Mango, paddy, jute, mustard; alluvial and "
            "mango-orchard dominated in some blocks."
        ),
        "Hooghly": (
            "Potato and vegetable belt near Kolkata markets; "
            "high-value vegetables, paddy."
        ),
        "North 24 Parganas": (
            "Coastal saline, urban-adjacent; paddy, vegetables, "
            "floriculture; strong peri-urban market access."
        ),
        "South 24 Parganas": (
            "Sundarbans coastal, high salinity; salt-tolerant "
            "paddy, aquaculture integration."
        ),
    },
    "Punjab": {
        "_default": (
            "Wheat-rice rotation dominant; high mechanization, "
            "canal + tube-well irrigation; stubble-burning issues."
        ),
        "Ludhiana": (
            "Wheat-rice and dairy belt; high yields, "
            "strong mandi infrastructure."
        ),
        "Amritsar": (
            "Wheat, rice, potato; near-attari border, "
            "good cold chain."
        ),
    },
    "Uttar Pradesh": {
        "_default": (
            "Sugarcane, wheat, rice, potato, mustard; "
            "largest sugarcane area in India."
        ),
        "Meerut": (
            "Sugarcane and wheat belt; high sugar mill density."
        ),
        "Varanasi": (
            "Rice, wheat, vegetables; alluvial Gangetic plain."
        ),
        "Lucknow": (
            "Wheat, rice, potato, vegetables; peri-urban "
            "vegetable demand."
        ),
    },
    "Bihar": {
        "_default": (
            "Rice, wheat, maize, litchi; flood-prone north, "
            "drought-prone south; high labor availability."
        ),
        "Patna": (
            "Vegetable and paddy belt near Patna markets; "
            "high-value vegetables."
        ),
        "Muzaffarpur": (
            "Litchi and paddy; Shahi litchi GI; "
            "horticulture strong."
        ),
    },
    "Odisha": {
        "_default": (
            "Paddy dominant, pulses and oilseeds in uplands; "
            "cyclone-prone coastal belt."
        ),
        "Cuttack": (
            "Paddy and vegetable belt; deltaic alluvial."
        ),
        "Ganjam": (
            "Coastal paddy, cashew, pulses; "
            "saline patches near Chilika."
        ),
    },
    "Tamil Nadu": {
        "_default": (
            "Rice, banana, sugarcane, coconut, pulses; "
            "tank + canal irrigation; delta districts are paddy hubs."
        ),
        "Thanjavur": (
            "Cauvery delta paddy bowl; highest rice productivity "
            "in the state."
        ),
        "Coimbatore": (
            "Coconut, banana, vegetables; "
            "peri-urban and industrial demand."
        ),
    },
    "Andhra Pradesh": {
        "_default": (
            "Rice, chilli, cotton, maize; Krishna-Godavari "
            "delta is a paddy powerhouse."
        ),
        "Krishna": (
            "Delta rice, chilli, turmeric; "
            "high-value spice belt."
        ),
        "Guntur": (
            "Chilli and cotton capital; "
            "Guntur chilli has strong export demand."
        ),
    },
    "Maharashtra": {
        "_default": (
            "Cotton, soybean, sugarcane, onion, grapes; "
            "rain-fed vs. irrigated split is sharp."
        ),
        "Nashik": (
            "Onion and grape capital; "
            "horticulture export belt."
        ),
        "Pune": (
            "Vegetables, grapes, pomegranate; "
            "peri-urban demand."
        ),
    },
    "Gujarat": {
        "_default": (
            "Cotton, groundnut, castor, wheat; "
            "canal irrigation in north, rain-fed elsewhere."
        ),
        "Ahmedabad": (
            "Cotton, wheat, vegetables; "
            "strong co-operative dairy/market network."
        ),
    },
    "Madhya Pradesh": {
        "_default": (
            "Soybean, wheat, gram, cotton; "
            "rain-fed dominant, irrigation expanding."
        ),
        "Indore": (
            "Soybean-wheat belt; "
            "strong mandi and processing base."
        ),
    },
    "Karnataka": {
        "_default": (
            "Ragi, rice, sugarcane, coffee (hill), arecanut; "
            "bimodal rainfall pattern."
        ),
        "Mandya": (
            "Sugarcane and paddy; Cauvery command area."
        ),
        "Belagavi": (
            "Sugarcane, tobacco, vegetables; "
            "Krishna basin irrigation."
        ),
    },
    "Telangana": {
        "_default": (
            "Rice, cotton, maize, turmeric; "
            "canal + borewell irrigation."
        ),
        "Nizamabad": (
            "Rice, turmeric, soybean; "
            "turmeric market is a major hub."
        ),
    },
    "Rajasthan": {
        "_default": (
            "Wheat, mustard, bajra, cotton; "
            "arid, canal irrigation in Indira Gandhi canal belt."
        ),
        "Sri Ganganagar": (
            "Wheat, mustard, cotton; "
            "canal-irrigated and highly productive."
        ),
    },
    "Haryana": {
        "_default": (
            "Wheat-rice, mustard, cotton; "
            "high mechanization, canal + tube-well."
        ),
        "Karnal": (
            "Wheat-rice, dairy, basmati; "
            "strong research + mandi ecosystem."
        ),
    },
    "Assam": {
        "_default": (
            "Rice (Sali, Ahu), tea, mustard, jute; "
            "flood-prone Brahmaputra valley."
        ),
        "Nagaon": (
            "Rice and mustard; "
            "Brahmaputra floodplain agriculture."
        ),
    },
}


def _region_hint(state: str, district: str) -> str:
    state_entry = REGION_CROP_HINTS.get(state, {})
    hint = state_entry.get(district) or state_entry.get("_default")

    if not hint:
        hint = (
            "Use all-India average cost structures and yields "
            "for this crop; adjust for season and rainfall."
        )

    return hint


# ─────────────────────────────────────────────────────────────
# ✅ Profit predictor — region dropdown helpers
# ─────────────────────────────────────────────────────────────
@app.get("/api/profit/regions")
def get_profit_regions():
    states = []

    for state_name, districts_map in AGMARKNET_MARKETS.items():
        if isinstance(districts_map, dict):
            districts = list(districts_map.keys())
        else:
            districts = []

        states.append(
            {
                "state": state_name,
                "districts": sorted(districts),
            }
        )

    states.sort(key=lambda x: x["state"])

    return {
        "status": "success",
        "total_states": len(states),
        "states": states,
    }


# ─────────────────────────────────────────────────────────────
# Existing Mandi endpoints
# ─────────────────────────────────────────────────────────────
@app.get("/api/mandi/locations")
def get_locations():
    return {"data": AGMARKNET_MARKETS}


# ⚡ SPEED: prices are deterministic per day, so cache them for the day
_MANDI_CACHE = {"date": None, "items": {}}


@app.get("/api/mandi/prices")
def get_daily_mandi_prices(
    state: str = Query(..., description="Selected State"),
    district: str = Query(..., description="Selected District"),
    market: str = Query("All", description="Selected Market or All"),
    category: str = Query("All", description="All, Paddy, or Vegetable"),
):
    today_str = date.today().strftime("%d-%m-%Y")
    today_seed = date.today().strftime("%Y%m%d")

    # Reset cache when the day changes
    if _MANDI_CACHE["date"] != today_seed:
        _MANDI_CACHE["date"] = today_seed
        _MANDI_CACHE["items"] = {}

    cache_key = (state, district, market, category)
    cached = _MANDI_CACHE["items"].get(cache_key)
    if cached is not None:
        return cached

    district_markets = AGMARKNET_MARKETS.get(state, {}).get(
        district,
        [f"{district} Main APMC Yard"],
    )

    if market != "All" and market in district_markets:
        target_markets = [market]
    else:
        target_markets = district_markets

    filtered_catalog = COMMODITY_CATALOG

    if category != "All":
        filtered_catalog = [
            item
            for item in COMMODITY_CATALOG
            if item["category"].lower() == category.lower()
        ]

    results = []

    for item in filtered_catalog:
        for mkt_name in target_markets:
            seed_key = f"{today_seed}_{state}_{district}_{mkt_name}_{item['name']}"
            seed_val = int(hashlib.md5(seed_key.encode()).hexdigest(), 16)

            # Daily rate delta between -7% to +9%
            variation_pct = ((seed_val % 160) - 70) / 1000.0
            daily_base = round(item["base_price"] * (1 + variation_pct), 2)

            min_price = round(daily_base * 0.94, 2)
            max_price = round(daily_base * 1.06, 2)
            modal_price = daily_base

            results.append(
                {
                    "date": today_str,
                    "state": state,
                    "district": district,
                    "market": mkt_name,
                    "commodity": item["name"],
                    "variety": item["variety"],
                    "category": item["category"],
                    "min_price_kg": f"₹{min_price:.2f}",
                    "max_price_kg": f"₹{max_price:.2f}",
                    "modal_price_kg": f"₹{modal_price:.2f}",
                    "unit": "Kg",
                }
            )

    response_payload = {
        "status": "success",
        "date": today_str,
        "state": state,
        "district": district,
        "market": market,
        "total_records": len(results),
        "data": results,
    }

    _MANDI_CACHE["items"][cache_key] = response_payload
    return response_payload


# ─────────────────────────────────────────────────────────────
# Profit predictor endpoint (Groq Ultra-Fast + Gemini Backup)
# ─────────────────────────────────────────────────────────────
class ProfitPredictRequest(BaseModel):
    crop: str
    area: float
    unit: str = "bigha"
    state: str = "West Bengal"
    district: str = "Burdwan"
    season: str = "Kharif"


# ⚡ SPEED: identical repeat requests are answered instantly (30 min cache)
PROFIT_CACHE_TTL = 1800
_PROFIT_CACHE = {}


@app.post("/api/predict-profit")
async def predict_crop_profit(req: ProfitPredictRequest):
    if not groq_client and not gemini_client:
        raise HTTPException(
            status_code=500,
            detail="Neither Groq nor Gemini API is configured.",
        )

    cache_key = (
        req.crop.strip().lower(),
        req.area,
        req.unit,
        req.state,
        req.district,
        req.season,
    )
    cached = _PROFIT_CACHE.get(cache_key)
    if cached and (time.time() - cached["ts"]) < PROFIT_CACHE_TTL:
        print("⚡ [Cache] Profit prediction served from cache.")
        return cached["response"]

    region_hint = _region_hint(req.state, req.district)

    prompt = f"""
You are an expert agricultural economist and agronomist in India.

Calculate a realistic, detailed financial projection and profit
analysis for the following farmer:

- Crop: {req.crop}
- Land Area: {req.area} {req.unit}
- Location: {req.district}, {req.state}
- Season: {req.season}

Regional agronomic context (use this to refine yields, costs,
and market prices — do NOT ignore it):
{region_hint}

Instructions:
1. Estimate costs in INR (Seeds, Land Preparation, Fertilizer/Pesticide, Irrigation, Labor, Harvesting/Transport). Costs must reflect LOCAL wage rates and input prices for {req.district}, {req.state}.
2. Predict expected yield range in Quintals and Kg. Yield must reflect LOCAL soil, water availability, and dominant varieties grown in {req.district}.
3. Predict realistic selling price per Quintal/Kg based on CURRENT market trends and nearby APMC prices for {req.district}, {req.state}.
4. Calculate Gross Revenue = Yield * Selling Price.
5. Calculate Net Profit = Gross Revenue - Total Cost.
6. Provide a 2-sentence practical farming advice in English ONLY. Do not use Bengali, Hindi, or any other language.

Return ONLY a valid JSON object matching this structure:

{{
    "costs": {{
        "seeds": 1200,
        "land_preparation": 2500,
        "fertilizer_pesticide": 3200,
        "irrigation": 1500,
        "labor": 4000,
        "harvesting": 2000,
        "total_cost": 14400
    }},
    "yield_prediction": {{
        "expected_quintals": 18.5,
        "expected_kg": 1850
    }},
    "market_price": {{
        "price_per_quintal": 2200,
        "price_per_kg": 22.0
    }},
    "financial_summary": {{
        "gross_revenue": 40700,
        "total_cost": 14400,
        "net_profit": 26300,
        "roi_percentage": "182%"
    }},
    "advice": "Apply urea in split doses and monitor for rain before top dressing."
}}
"""

    try:
        # ⚡ Blocking SDK call moved to a worker thread
        raw_response = await asyncio.to_thread(
            execute_groq_with_fallback,
            prompt,
            True,
            2048,
        )

        data = clean_json_response(raw_response)

        response_payload = {
            "status": "success",
            "prediction": data,
        }

        _PROFIT_CACHE[cache_key] = {"ts": time.time(), "response": response_payload}

        # keep cache small
        if len(_PROFIT_CACHE) > 500:
            oldest_key = min(_PROFIT_CACHE, key=lambda k: _PROFIT_CACHE[k]["ts"])
            _PROFIT_CACHE.pop(oldest_key, None)

        return response_payload

    except Exception as e:
        print("❌ Profit Prediction Error:", str(e))
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=friendly_error(e),
        )