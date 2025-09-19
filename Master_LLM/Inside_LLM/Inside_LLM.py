import os
import json
import re
import asyncio
import google.generativeai as genai
from Inside_LLM.prompts import system_prompt
from Inside_LLM.tools import available_tools

# --- Configure Gemini ---
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("Environment variable GOOGLE_API_KEY is not set")
genai.configure(api_key=API_KEY)

# Gemini model
model = genai.GenerativeModel(model_name="gemini-2.5-flash-lite")

# --- Utility: Safe JSON parse ---
def safe_json_loads(text: str):
    try:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        return json.loads(cleaned)
    except Exception:
        return None

# --- Extract coordinates ---
def extract_coordinates(text: str):
    match = re.findall(r"(-?\d{1,3}\.\d+)", text)
    if len(match) >= 2:
        try:
            lat, lon = float(match[0]), float(match[1])
            return lat, lon
        except:
            return None
    return None

# --- Safety classification (hard rules) ---
def classify_safety(mine_name, soil, weather, forecast):
    sm = soil.get("moisture", 0)
    humidity = weather.get("humidity", 0)
    rain_prob = weather.get("precipitation_probability", 0)
    rain_amount = weather.get("precipitation", 0)

    # Example rules (tune as per site safety guidelines)
    if sm and sm > 0.30:
        return "unsafe_to_work", f"Soil moisture too high ({sm:.3f})"
    if humidity and rain_prob and humidity > 85 and rain_prob > 70:
        return "unsafe_to_work", f"High humidity {humidity}% + rain probability {rain_prob}%"
    if rain_amount and rain_amount > 2.0:
        return "unsafe_to_work", f"Heavy rainfall detected ({rain_amount} mm)"

    return "safe_to_work", "All parameters within safe thresholds"

# --- Main Bot ---
async def run_bot(query: str) -> dict:
    chat = model.start_chat(history=[])

    # --- Step 1: web_search ---
    prompt1 = (
        f"{system_prompt}\n\n"
        f"User Query:\n{query}\n\n"
        f"Return coordinates of this mine using the web_search tool in JSON format like:\n"
        '{"step": "action", "tool": "web_search", "tool_input": "<search query>"}'
    )
    response1 = chat.send_message(prompt1)
    parsed1 = safe_json_loads(response1.text)
    print(parsed1)
    if not parsed1:
        return {"status": "error", "error": "Invalid JSON from Gemini", "raw": response1.text}

    tool_input = parsed1.get("tool_input")
    web_result = available_tools["web_search"]["function"](tool_input)
    coords = extract_coordinates(web_result)
    print(coords)
    if not coords:
        return {"status": "error", "error": "Could not extract coordinates", "raw": web_result}
    lat, lon = coords

    # --- Step 2: weather_and_soil ---
    soil_weather_output = available_tools["weather_and_soil"]["function"](lat, lon, tool_input)
    if not soil_weather_output:
        return {"status": "error", "error": "weather_and_soil returned None"}

    soil = soil_weather_output.get("soil", {})
    weather = soil_weather_output.get("weather", {}).get("current", {})
    ndvi = soil_weather_output.get("ndvi", {})

    # --- Determine simple weather condition ---
    precipitation = weather.get("precipitation", 0)
    if precipitation == 0:
        condition_str = "sunny/clear"
    elif precipitation < 2:
        condition_str = "cloudy"
    else:
        condition_str = "rainy"

    # --- Build concise condition string ---
    condition = (
        f"Humidity: {weather.get('humidity', 'N/A')}%, "
        f"Air Temperature: {weather.get('temperature', 'N/A')}°C, "
        f"Soil Moisture: {soil.get('moisture', 'N/A')}, "
        f"Soil Temp 0cm: {soil.get('t0_celsius', 'N/A')}°C, "
        f"Soil Temp 10cm: {soil.get('t10_celsius', 'N/A')}°C, "
        f"NDVI: {ndvi.get('ndvi', 'N/A')}, "
        f"Weather Condition: {condition_str}"
    )

    # --- Step 3: Apply rules ---
    conclusion, rule_analysis = classify_safety(tool_input, soil, weather, soil_weather_output.get("weather", {}).get("forecast", []))

    # --- Step 4: Ask Gemini for explanation (optional) ---
    prompt2 = (
        f"{system_prompt}\n\n"
        f"Mine: {tool_input}\n\n"
        f"Weather & Soil Data:\n{condition}\n\n"
        f"My rule-based system classified this mine as: {conclusion} ({rule_analysis}).\n"
        "Write a short numeric analysis summary in JSON with key 'analysis'."
    )
    response2 = chat.send_message(prompt2)
    parsed2 = safe_json_loads(response2.text)
    gemini_analysis = parsed2.get("analysis") if parsed2 else rule_analysis

    return {
        "status": "done",
        "mine": tool_input,
        "conclusion": conclusion,
        "condition": condition,
        "analysis": gemini_analysis,
        "coords": {"lat": lat, "lon": lon}
    }

# --- Example Usage ---
if __name__ == "__main__":
    async def test_bot():
        queries = [
            "Malanjkhand Copper Mine Madhya Pradesh",
            #"Tummalapalle Uranium Mine Andhra Pradesh",
            # "Khetri Copper Mine Rajasthan",
            # "Jharia Coal Mine Jharkhand",
            # "Singrauli Coalfield Madhya Pradesh",
            # "Bailadila Iron Ore Mines Chhattisgarh",
            # "Rampura Agucha Mines",
        ]
        for q in queries:
            result = await run_bot(q)
            print("\n[FINAL RESULT]", json.dumps(result, indent=2))

    asyncio.run(test_bot())













































#         queries = [
#                 "Gevra OC coal mine Chhattisgarh",
#                 "Malanjkhand Copper Mine Madhya Pradesh",
#                 "Tummalapalle Uranium Mine Andhra Pradesh",
#                 "Khetri Copper Mine Rajasthan",
#                 "Jharia Coal Mine Jharkhand",
#                 "Singrauli Coalfield Madhya Pradesh",
#                 "Bailadila Iron Ore Mines Chhattisgarh",
#                 "Rampura Agucha Mines",
#             ]