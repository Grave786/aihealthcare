import json
import logging
import os
from pathlib import Path
from urllib import error, request

import re
import random
import time
import joblib
import pandas as pd
from django.conf import settings


MODEL_DIR = Path(settings.BASE_DIR)

DIABETES_FORM_FIELDS = [
    ("pregnancies", "Pregnancies"),
    ("glucose", "Glucose"),
    ("blood_pressure", "BloodPressure"),
    ("skin_thickness", "SkinThickness"),
    ("insulin", "Insulin"),
    ("bmi", "BMI"),
    ("diabetes_pedigree_function", "DiabetesPedigreeFunction"),
    ("diabetes_age", "Age"),
]

HEART_FORM_FIELDS = [
    ("heart_age", "age"),
    ("sex", "sex"),
    ("cp", "cp"),
    ("trestbps", "trestbps"),
    ("chol", "chol"),
    ("fbs", "fbs"),
    ("restecg", "restecg"),
    ("thalach", "thalach"),
    ("exang", "exang"),
    ("oldpeak", "oldpeak"),
    ("slope", "slope"),
    ("ca", "ca"),
    ("thal", "thal"),
]

DIABETES_COLUMNS = [column for _, column in DIABETES_FORM_FIELDS]
HEART_COLUMNS = [column for _, column in HEART_FORM_FIELDS]

_diabetes_model = None
_diabetes_scaler = None
_heart_model = None
_heart_scaler = None

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_TIMEOUT_SECONDS = 30
GEMINI_MAX_ATTEMPTS = 3
GEMINI_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}

logger = logging.getLogger(__name__)


def _load_pickle(filename):
    path = MODEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {filename}. Place it in the project root: {MODEL_DIR}"
        )
    return joblib.load(path)


def load_models():
    global _diabetes_model, _diabetes_scaler, _heart_model, _heart_scaler

    if _diabetes_model is None or _diabetes_scaler is None:
        _diabetes_model = _load_pickle("diabetes_model.pkl")
        _diabetes_scaler = _load_pickle("diabetes_scaler.pkl")

    if _heart_model is None or _heart_scaler is None:
        _heart_model = _load_pickle("heart_model.pkl")
        _heart_scaler = _load_pickle("heart_scaler.pkl")

    return _diabetes_model, _diabetes_scaler, _heart_model, _heart_scaler


def load_diabetes_model():
    global _diabetes_model, _diabetes_scaler

    if _diabetes_model is None or _diabetes_scaler is None:
        _diabetes_model = _load_pickle("diabetes_model.pkl")
        _diabetes_scaler = _load_pickle("diabetes_scaler.pkl")

    return _diabetes_model, _diabetes_scaler


def load_heart_model():
    global _heart_model, _heart_scaler

    if _heart_model is None or _heart_scaler is None:
        _heart_model = _load_pickle("heart_model.pkl")
        _heart_scaler = _load_pickle("heart_scaler.pkl")

    return _heart_model, _heart_scaler


def _build_feature_frame(input_data, field_map):
    if isinstance(input_data, dict):
        values = [float(input_data[form_field]) for form_field, _ in field_map]
    else:
        values = [float(value) for value in input_data]

    columns = [training_column for _, training_column in field_map]
    return pd.DataFrame([values], columns=columns)


def predict_diabetes(input_data):
    diabetes_model, diabetes_scaler = load_diabetes_model()
    input_df = _build_feature_frame(input_data, DIABETES_FORM_FIELDS)
    scaled_input = diabetes_scaler.transform(input_df)
    probability = diabetes_model.predict_proba(scaled_input)[0][1]

    return {"probability": round(probability * 100, 2)}


def predict_heart(input_data):
    heart_model, heart_scaler = load_heart_model()
    input_df = _build_feature_frame(input_data, HEART_FORM_FIELDS)
    scaled_input = heart_scaler.transform(input_df)
    probability = heart_model.predict_proba(scaled_input)[0][1]

    return {"probability": round(probability * 100, 2)}


def risk_from_probability(probability):
    if probability < 30:
        return "Low Risk"
    if probability <= 70:
        return "Moderate Risk"
    return "High Risk"


def predict_disease(cleaned_data):
    prediction_type = cleaned_data.get("prediction_type", "both")
    diabetes_prob = None
    heart_prob = None

    if prediction_type in ("diabetes", "both"):
        diabetes_prob = predict_diabetes(cleaned_data)["probability"]

    if prediction_type in ("heart", "both"):
        heart_prob = predict_heart(cleaned_data)["probability"]

    if prediction_type == "diabetes":
        disease = "Diabetes"
        probability = diabetes_prob
    elif prediction_type == "heart":
        disease = "Heart Disease"
        probability = heart_prob
    elif diabetes_prob > heart_prob:
        disease = "Diabetes"
        probability = diabetes_prob
    else:
        disease = "Heart Disease"
        probability = heart_prob

    risk = risk_from_probability(probability)

    return {
        "disease": disease,
        "probability": probability,
        "risk": risk,
        "diabetes_probability": diabetes_prob,
        "heart_probability": heart_prob,
    }


def fallback_recommendation(disease, probability, risk):
    if disease == "Diabetes":
        disease_precautions = [
            "Keep a regular record of glucose-related symptoms.",
            "Limit sweet drinks, sweets, and refined carbohydrates.",
            "Discuss blood sugar screening with a qualified doctor.",
        ]
        breakfast = ["Oats with nuts.", "Unsweetened yogurt with fruit."]
        lunch = [
            "Dal, vegetables, and a small whole-grain portion.",
            "Salad with lean protein.",
        ]
        tests = [
            "Ask about fasting glucose testing.",
            "Ask about HbA1c testing.",
            "Review BMI and waist measurements.",
        ]
    else:
        disease_precautions = [
            "Track blood pressure and chest-related symptoms.",
            "Reduce fried, salty, and high-fat foods.",
            "Discuss heart risk screening with a qualified doctor.",
        ]
        breakfast = ["Whole-grain toast with eggs.", "Fruit with unsweetened yogurt."]
        lunch = [
            "Vegetables with dal or lean protein.",
            "Brown rice or roti in a modest portion.",
        ]
        tests = [
            "Ask about blood pressure checks.",
            "Ask about cholesterol testing.",
            "Review ECG need with a doctor.",
        ]

    general_precautions = [
        "This result is only a risk estimate, not a diagnosis.",
        f"Review the {risk.lower()} result with a qualified doctor.",
    ]
    random.shuffle(disease_precautions)

    recommendation_data = {
        "summary": f"{disease} risk is estimated at {probability}% ({risk}).",
        "precautions": general_precautions + disease_precautions[:1],
        "diet_plan": {
            "breakfast": breakfast,
            "lunch": lunch,
            "dinner": ["Vegetable soup with protein.", "Keep dinner light and balanced."],
            "snacks": ["Nuts or sprouts.", "Fresh fruit without added sugar."],
        },
        "exercise": [
            "Walk regularly if comfortable.",
            "Add gentle stretching most days.",
            "Avoid overexertion during symptoms.",
        ],
        "lifestyle": [
            "Sleep on a steady schedule.",
            "Avoid tobacco and limit alcohol.",
            "Manage stress with breathing or relaxation.",
        ],
        "tests": tests,
        "when_to_consult_doctor": [
            "Seek urgent care for chest pain.",
            "Seek urgent care for breathing difficulty.",
            "Consult soon if symptoms worsen.",
        ],
    }
    return _format_recommendation(recommendation_data)


def _clean_gemini_model_name(model_name):
    model_name = (model_name or DEFAULT_GEMINI_MODEL).strip()
    return model_name.removeprefix("models/")


def _safe_input_summary(input_data):
    if not input_data:
        return "No additional input values provided."

    hidden_keys = {"csrfmiddlewaretoken"}
    lines = []
    for key, value in input_data.items():
        if key in hidden_keys or value in (None, ""):
            continue
        label = key.replace("_", " ").title()
        lines.append(f"- {label}: {value}")

    return "\n".join(lines) if lines else "No additional input values provided."


# def _build_recommendation_prompt(disease, probability, risk, input_data=None):
#     input_summary = _safe_input_summary(input_data)
#     return f"""
# Patient risk summary:
# - Main risk area: {disease}
# - Risk chance: {probability}%
# - Risk level: {risk}

# Patient input values:
# {input_summary}

# Create safe, beginner-friendly healthcare guidance for a web app result page.

# Rules:
# - Do not diagnose the patient.
# - Do not prescribe medicines, dosages, supplements, or treatments.
# - Do not claim the result is certain.
# - Recommend consulting a qualified doctor where appropriate.
# - Mention emergency care for severe symptoms such as chest pain, breathing difficulty, fainting, confusion, or rapidly worsening symptoms.
# - Keep the language simple for non-medical users.
# - Keep each bullet short and practical.

# Return only valid JSON using this exact schema:
# {{
#   "summary": "one short sentence explaining the result safely",
#   "precautions": ["bullet", "bullet", "bullet"],
#   "diet_plan": {{
#     "breakfast": ["bullet", "bullet"],
#     "lunch": ["bullet", "bullet"],
#     "dinner": ["bullet", "bullet"],
#     "snacks": ["bullet", "bullet"]
#   }},
#   "exercise": ["bullet", "bullet", "bullet"],
#   "lifestyle": ["bullet", "bullet", "bullet"],
#   "tests": ["bullet", "bullet", "bullet"],
#   "when_to_consult_doctor": ["bullet", "bullet", "bullet"]
# }}
# """


def _build_recommendation_prompt(disease, probability, risk, input_data=None):
    input_summary = _safe_input_summary(input_data)

    variation_seed = random.randint(1, 100000)
    timestamp = int(time.time())

    return f"""
You are generating UNIQUE personalized healthcare advice.

IMPORTANT:
- Every response MUST be DIFFERENT.
- Do NOT repeat same sentences.
- Use different wording, food suggestions, and lifestyle tips each time.

Variation seed: {variation_seed}
Timestamp: {timestamp}

Patient Details:
- Disease: {disease}
- Risk: {risk}
- Probability: {probability}%

Patient Inputs:
{input_summary}

PERSONALIZATION RULES:
- If Diabetes, focus on sugar, carbs, and BMI.
- If Heart Disease, focus on cholesterol, BP, and exercise.
- High risk means stricter advice.
- Low risk means preventive advice.

STRICT RULES:
- No medicines
- No diagnosis
- Use simple language
- Make response DIFFERENT every time
- Keep each list to exactly 3 short bullets.
- Keep every bullet under 16 words.
- Return complete valid JSON only. Do not add markdown fences.

Return JSON:
{{
  "summary": "...",
  "precautions": ["..."],
  "diet_plan": {{
    "breakfast": ["..."],
    "lunch": ["..."],
    "dinner": ["..."],
    "snacks": ["..."]
  }},
  "exercise": ["..."],
  "lifestyle": ["..."],
  "tests": ["..."],
  "when_to_consult_doctor": ["..."]
}}
"""

# def _gemini_payload(prompt):
#     return {
#         "system_instruction": {
#             "parts": [
#                 {
#                     "text": (
#                         "You are a careful healthcare education assistant. "
#                         "You provide safe informational wellness guidance only, "
#                         "never diagnosis or medicine prescriptions."
#                     )
#                 }
#             ]
#         },
#         "contents": [
#             {
#                 "role": "user",
#                 "parts": [{"text": prompt}],
#             }
#         ],
#         "generationConfig": {
#             "temperature": 0.2,
#             "topP": 0.8,
#             "maxOutputTokens": 1200,
#             "responseMimeType": "application/json",
#         },
#     }

def _gemini_payload(prompt):
    return {
        "system_instruction": {
            "parts": [
                {
                    "text": (
                        "You are a healthcare assistant. "
                        "Always generate varied, personalized responses."
                    )
                }
            ]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "topP": 0.9,
            "maxOutputTokens": 2048,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "summary": {"type": "STRING"},
                    "precautions": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    },
                    "diet_plan": {
                        "type": "OBJECT",
                        "properties": {
                            "breakfast": {
                                "type": "ARRAY",
                                "items": {"type": "STRING"},
                            },
                            "lunch": {
                                "type": "ARRAY",
                                "items": {"type": "STRING"},
                            },
                            "dinner": {
                                "type": "ARRAY",
                                "items": {"type": "STRING"},
                            },
                            "snacks": {
                                "type": "ARRAY",
                                "items": {"type": "STRING"},
                            },
                        },
                        "required": ["breakfast", "lunch", "dinner", "snacks"],
                    },
                    "exercise": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    },
                    "lifestyle": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    },
                    "tests": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    },
                    "when_to_consult_doctor": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    },
                },
                "required": [
                    "summary",
                    "precautions",
                    "diet_plan",
                    "exercise",
                    "lifestyle",
                    "tests",
                    "when_to_consult_doctor",
                ],
            },
        },
    }

def _extract_gemini_text(response_data):
    candidates = response_data.get("candidates", [])
    if not candidates:
        raise ValueError("Gemini returned no candidates.")

    candidate = candidates[0]
    finish_reason = candidate.get("finishReason")
    if finish_reason == "MAX_TOKENS":
        raise ValueError(
            "Gemini response was cut off. Increase maxOutputTokens or shorten the prompt."
        )

    parts = candidate.get("content", {}).get("parts", [])
    text_parts = [part.get("text", "") for part in parts if part.get("text")]
    text = "\n".join(text_parts).strip()
    if not text:
        raise ValueError("Gemini returned an empty recommendation.")

    return text



def _parse_json_text(text):
    try:
        # Remove markdown
        cleaned = text.strip()
        if "```" in cleaned:
            cleaned = re.sub(r"```json|```", "", cleaned).strip()

        # Extract JSON block ONLY
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)

        return json.loads(cleaned)

    except Exception as e:
        logger.warning("Gemini JSON parsing failed: %s", e)
        logger.debug("Raw Gemini output: %s", text[:2000])
        raise


def _format_list(items):
    if not items:
        return "- Discuss personalized guidance with a qualified doctor."
    return "\n".join(f"- {item}" for item in items)


def _format_recommendation(data):
    diet_plan = data.get("diet_plan", {})
    return "\n\n".join(
        [
            "Summary:\n" + data.get("summary", "Review this result with a qualified doctor."),
            "Precautions:\n" + _format_list(data.get("precautions")),
            (
                "Diet Plan:\n"
                "Breakfast:\n"
                + _format_list(diet_plan.get("breakfast"))
                + "\nLunch:\n"
                + _format_list(diet_plan.get("lunch"))
                + "\nDinner:\n"
                + _format_list(diet_plan.get("dinner"))
                + "\nSnacks:\n"
                + _format_list(diet_plan.get("snacks"))
            ),
            "Exercise:\n" + _format_list(data.get("exercise")),
            "Lifestyle:\n" + _format_list(data.get("lifestyle")),
            "Tests:\n" + _format_list(data.get("tests")),
            "When to consult doctor:\n"
            + _format_list(data.get("when_to_consult_doctor")),
        ]
    )


def call_gemini_recommendation(prompt):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured.")

    model = _clean_gemini_model_name(os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL))
    url = GEMINI_API_URL.format(model=model)
    payload = _gemini_payload(prompt)
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    for attempt in range(1, GEMINI_MAX_ATTEMPTS + 1):
        gemini_request = request.Request(
            url,
            data=data,
            headers=headers,
            method="POST",
        )

        try:
            with request.urlopen(
                gemini_request, timeout=GEMINI_TIMEOUT_SECONDS
            ) as response:
                response_data = json.loads(response.read().decode("utf-8"))
            break
        except error.HTTPError as exc:
            should_retry = exc.code in GEMINI_RETRY_STATUS_CODES
            if not should_retry or attempt == GEMINI_MAX_ATTEMPTS:
                raise

            wait_seconds = 2 ** (attempt - 1)
            logger.warning(
                "Gemini temporary HTTP error %s; retrying in %ss (%s/%s)",
                exc.code,
                wait_seconds,
                attempt,
                GEMINI_MAX_ATTEMPTS,
            )
            time.sleep(wait_seconds)
        except (TimeoutError, error.URLError) as exc:
            if attempt == GEMINI_MAX_ATTEMPTS:
                raise

            wait_seconds = 2 ** (attempt - 1)
            logger.warning(
                "Gemini request failed temporarily: %s; retrying in %ss (%s/%s)",
                exc,
                wait_seconds,
                attempt,
                GEMINI_MAX_ATTEMPTS,
            )
            time.sleep(wait_seconds)

    text = _extract_gemini_text(response_data)
    return _parse_json_text(text)


# def get_recommendation(disease, probability, risk, input_data=None):
#     prompt = _build_recommendation_prompt(disease, probability, risk, input_data)
#     try:
#         recommendation_data = call_gemini_recommendation(prompt)
#         return _format_recommendation(recommendation_data)
#     except (
#         ValueError,
#         json.JSONDecodeError,
#         TimeoutError,
#         error.URLError,
#         error.HTTPError,
#     ):
#         return fallback_recommendation(disease, probability, risk)

def get_recommendation(disease, probability, risk, input_data=None):
    prompt = _build_recommendation_prompt(disease, probability, risk, input_data)

    try:
        recommendation_data = call_gemini_recommendation(prompt)
        logger.info("Gemini recommendation generated successfully.")
        return _format_recommendation(recommendation_data)

    except (
        ValueError,
        json.JSONDecodeError,
        TimeoutError,
        error.URLError,
        error.HTTPError,
    ) as exc:
        logger.warning("Using local recommendation fallback: %s", exc)
        return fallback_recommendation(disease, probability, risk)
