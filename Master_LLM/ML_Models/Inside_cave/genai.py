import os
import json
from typing import Any, Dict
import google.generativeai as genai
from dotenv import load_dotenv
from Master_LLM.ML_Models.Inside_cave.model.inside_cave import process_video_file

# ---- Load .env ----
load_dotenv()

# ---- Configure Gemini ----
API_KEY = os.getenv("GOOGLE_API_KEY1")
if not API_KEY:
    raise ValueError("Set GOOGLE_API_KEY in your .env file")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash-lite")

# ---------------- Helpers ----------------
def safe_json_loads(text: str):
    """Clean and parse JSON safely."""
    try:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        return json.loads(cleaned)
    except Exception:
        return None

# ---------------- Confidence Scaling ----------------
def scale_confidence(conf: float) -> int:
    """
    Scale confidence according to piecewise rules:
    - <=40% → same (or minimum 40)
    - 40-55% → 40-90% scaled linearly
    - 55-60% → 90-95% scaled linearly
    - >60% → 98% capped
    Input conf is float between 0 and 1.
    """
    actual = conf * 100

    if actual <= 40:
        scaled = actual
    elif 40 < actual <= 55:
        scaled = 40 + (actual - 40) * (90 - 40) / (55 - 40)
    elif 55 < actual <= 60:
        scaled = 90 + (actual - 55) * (95 - 90) / (60 - 55)
    else:  # >60
        scaled = 98

    return int(round(scaled))

# ---------------- Gemini Summarization ----------------
def summarize_with_gemini(predictions: Dict[str, Any]) -> Dict[str, Any]:
    highest_conf = 0
    flat_preds = []

    for frame, preds in predictions.items():
        for p in preds:
            conf = p.get("confidence", 0)
            flat_preds.append(f"{frame}: {p['class']} (conf={conf:.2f})")
            if conf > highest_conf:
                highest_conf = conf

    preds_text = "\n".join(flat_preds) or "No predictions available"

    system_prompt = """You are analyzing cave-inside video predictions from an ML model.
Please provide a short human-readable summary (2-3 sentences),
highlighting the most important detection (highest confidence).
Also classify:
- riskLevel: Low, Medium, or High
- confidence: integer 0-100
- rockSize: Small, Medium, Large
- trajectory: Stable, Moderate, Unstable
- recommendations: 1-3 short actionable items

Return ONLY valid JSON in this format:
{
  "riskLevel": "...",
  "confidence": ...,
  "rockSize": "...",
  "trajectory": "...",
  "recommendations": ["...", "..."]
}"""

    messages = [
        {"role": "model", "parts": [{"text": system_prompt}]},
        {"role": "user", "parts": [{"text": f"Predictions:\n{preds_text}"}]}
    ]

    try:
        chat = model.start_chat(history=messages)
        resp = chat.send_message("Summarize predictions as instructed.")
        text = resp.text.strip()
        summary = safe_json_loads(text)

        if not summary:
            raise ValueError("Gemini returned invalid JSON")

        # Apply scaling logic to highest confidence
        summary["confidence"] = scale_confidence(highest_conf)
        return summary

    except Exception as e:
        return {
            "riskLevel": "Unknown",
            "confidence": scale_confidence(highest_conf),
            "rockSize": "Unknown",
            "trajectory": "Unknown",
            "recommendations": [f"❌ Gemini failed: {e}"]
        }

# ---------------- Conclusion Analysis ----------------
def conclusion_analysis(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adjusts riskLevel, confidence, trajectory, rockSize, and recommendations
    based on all available factors.
    """
    conf = analysis.get("confidence", 0)
    rock_size = analysis.get("rockSize", "Unknown")
    trajectory = analysis.get("trajectory", "Unknown")

    # ----- Adjust riskLevel based on confidence -----
    if conf <= 40:
        risk = "Low"
    elif conf <= 60:
        risk = "Medium"
    elif conf <= 75:
        risk = "High"
    else:
        risk = "Very High"

    # ----- Adjust trajectory if missing -----
    if trajectory == "Unknown":
        if conf > 70:
            trajectory = "Unstable"
        elif conf > 50:
            trajectory = "Moderate"
        else:
            trajectory = "Stable"

    # ----- Adjust rockSize if missing -----
    if rock_size == "Unknown":
        if conf > 75:
            rock_size = "Large"
        elif conf > 50:
            rock_size = "Medium"
        else:
            rock_size = "Small"

    # ----- Adjust recommendations based on riskLevel -----
    recommendations = []
    if risk in ["Low", "Medium"]:
        recommendations.append("Continue monitoring")
        if rock_size in ["Medium", "Large"]:
            recommendations.append("Schedule inspection")
    elif risk in ["High", "Very High"]:
        recommendations.append("Immediate inspection")
        recommendations.append("Evacuate personnel if necessary")
        if trajectory == "Unstable":
            recommendations.append("Reinforce support structures")

    # Remove duplicates
    recommendations = list(dict.fromkeys(recommendations))

    # ----- Return adjusted analysis -----
    return {
        "riskLevel": risk,
        "confidence": conf,
        "trajectory": trajectory,
        "rockSize": rock_size,
        "recommendations": recommendations
    }

# ---------------- Core Function ----------------
def process_video_and_summarize(video_file: Any) -> Dict[str, Any]:
    try:
        all_predictions = process_video_file(video_file)
    except Exception as e:
        return {"success": False, "error": f"❌ Video processing failed: {e}"}

    summary = summarize_with_gemini(all_predictions)

    analysis = {
        "riskLevel": summary.get("riskLevel", "Unknown"),
        "confidence": summary.get("confidence", 0),
        "trajectory": summary.get("trajectory", "Unknown"),
        "rockSize": summary.get("rockSize", "Unknown"),
        "recommendations": summary.get(
            "recommendations",
            ["Continue monitoring", "Schedule inspection"]
        )
    }

    final_analysis = conclusion_analysis(analysis)

    return {"success": True, "analysis": final_analysis}

# ---------------- Example Usage ----------------
if __name__ == "__main__":
    test_video = r"C:\Users\KAIZEN\Downloads\vedio_testing\inner_cave\generated-video.mp4"
    with open(test_video, "rb") as f:
        result = process_video_and_summarize(f)

    print("\n📌 Final API Response:")
    print(json.dumps(result, indent=2))
