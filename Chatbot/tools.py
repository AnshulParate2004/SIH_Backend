import requests
import subprocess
from zoneinfo import ZoneInfo
import time
from datetime import datetime, timezone
from dotenv import load_dotenv
import os
import requests
import json
import re

load_dotenv()

AGRO_API_KEY = os.getenv("AGRO_API_KEY")
AGRO_BASE_URL = "https://api.agromonitoring.com/agro/1.0"

def extract_coordinates(text: str):
    """
    Extract latitude and longitude from a text string.
    Supports formats like:
    - 22.0066° N, 80.7040° E
    - lat:22.0066, lon:80.7040
    - 22.0066, 80.7040
    """
    if not text:
        return None

    # Pattern for "22.0066° N, 80.7040° E"
    match = re.search(r"([-+]?\d{1,2}\.\d+)[°]?\s*[NnSs]?,?\s*([-+]?\d{1,3}\.\d+)[°]?\s*[EeWw]?", text)
    if match:
        return float(match.group(1)), float(match.group(2))

    # Pattern for "lat:22.0066, lon:80.7040"
    match = re.search(r"lat[: ]\s*([-+]?\d{1,2}\.\d+).*lon[: ]\s*([-+]?\d{1,3}\.\d+)", text, re.I)
    if match:
        return float(match.group(1)), float(match.group(2))

    # Pattern for "22.0066, 80.7040"
    match = re.search(r"([-+]?\d{1,2}\.\d+)\s*,\s*([-+]?\d{1,3}\.\d+)", text)
    if match:
        return float(match.group(1)), float(match.group(2))

    return None

# --- Helper functions ---
def kelvin_to_celsius(kelvin):
    return round(kelvin - 273.15, 2)

def get_existing_polygons():
    url = f"{AGRO_BASE_URL}/polygons?appid={AGRO_API_KEY}"
    r = requests.get(url)
    r.raise_for_status()
    return r.json()

def create_polygon(lat, lon, name="my_polygon"):
    delta = 0.01  # ~1 km offset
    coords = [
        [lon - delta, lat - delta],
        [lon - delta, lat + delta],
        [lon + delta, lat + delta],
        [lon + delta, lat - delta],
        [lon - delta, lat - delta],
    ]
    poly_data = {
        "name": name,
        "geo_json": {
            "type": "Feature",
            "properties": {},
            "geometry": {"type": "Polygon", "coordinates": [coords]},
        },
    }
    url = f"{AGRO_BASE_URL}/polygons?appid={AGRO_API_KEY}"
    r = requests.post(url, json=poly_data)
    r.raise_for_status()
    return r.json()

def get_or_create_polygon(lat, lon, name):
    polygons = get_existing_polygons()
    for poly in polygons:
        if poly["name"] == name:
            return poly
    return create_polygon(lat, lon, name)

def fetch_soil_data(polygon_id):
    url = f"{AGRO_BASE_URL}/soil?polyid={polygon_id}&appid={AGRO_API_KEY}"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()
    return {
        "t0_celsius": kelvin_to_celsius(data.get("t0", 0)),
        "t10_celsius": kelvin_to_celsius(data.get("t10", 0)),
        "moisture": data.get("moisture"),
        "timestamp": datetime.fromtimestamp(data.get("ts", 0), tz=timezone.utc).isoformat(),
    }

def fetch_weather_openmeteo(lat, lon, hours=24):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ["temperature_2m", "relative_humidity_2m", "precipitation", "precipitation_probability"],
        "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "precipitation_probability"],
        "forecast_hours": hours,
        "timezone": "auto",
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()
    result = {
        "current": {
            "temperature": data["current"].get("temperature_2m"),
            "humidity": data["current"].get("relative_humidity_2m"),
            "precipitation": data["current"].get("precipitation"),
            "precipitation_probability": data["current"].get("precipitation_probability"),
        },
        "forecast": []
    }
    for i, t in enumerate(data["hourly"]["time"][:hours]):
        result["forecast"].append({
            "time": t,
            "temperature": data["hourly"]["temperature_2m"][i],
            "humidity": data["hourly"]["relative_humidity_2m"][i],
            "precipitation": data["hourly"]["precipitation"][i],
            "precipitation_probability": data["hourly"]["precipitation_probability"][i],
        })
    return result

def fetch_ndvi(polygon_id):
    end = int(time.time())
    start = end - 30 * 24 * 3600  # last 30 days
    url = f"{AGRO_BASE_URL}/ndvi/history?polyid={polygon_id}&start={start}&end={end}&appid={AGRO_API_KEY}"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()
    if not data:
        return None
    latest = max(data, key=lambda x: x["dt"])
    return {
        "date": datetime.fromtimestamp(latest["dt"], tz=timezone.utc).isoformat(),
        "ndvi": round(latest["data"]["mean"], 3),
    }

def weatherandsoil_search(lat, lon, name):
    result = {}

    # Polygon
    polygon = get_or_create_polygon(lat, lon, name)
    polygon_id = polygon["id"]
    result["polygon_id"] = polygon_id
    result["polygon_area_ha"] = round(polygon["area"]/10000, 2)

    # Soil
    try:
        result["soil"] = fetch_soil_data(polygon_id)
    except Exception:
        result["soil"] = None

    # Weather
    try:
        result["weather"] = fetch_weather_openmeteo(lat, lon, hours=24)
    except Exception:
        result["weather"] = None

    # Vegetation (NDVI)
    try:
        result["ndvi"] = fetch_ndvi(polygon_id)
    except Exception:
        result["ndvi"] = None

    return result


def get_weather(city: str):
    city = (city or "").strip()
    if not city:
        return "Please provide a city."
    url = f"https://wttr.in/{city}?format=%C+%t"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return f"The weather in {city} is {r.text}."
        return f"Weather lookup failed with status {r.status_code}."
    except requests.RequestException as e:
        return f"Weather lookup error: {e}"


def get_time(place_or_tz: str) -> str:
    s = (place_or_tz or "").strip().lower()
    if not s:
        return "Please provide a place or timezone."
    if any(k in s for k in ["india", "ist", "kolkata", "delhi", "mumbai", "nagpur"]):
        tz = "Asia/Kolkata"
    elif "/" in s:
        tz = place_or_tz.strip()
    else:
        tz = "UTC"
    try:
        now = datetime.now(ZoneInfo(tz))
        return f"Current time in {tz} is {now.strftime('%Y-%m-%d %H:%M:%S')}."
    except Exception:
        now = datetime.utcnow()
        return f"Unknown timezone. UTC time is {now.strftime('%Y-%m-%d %H:%M:%S')}."


# Load API key from .env file
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is not set in the environment")

def search_tavily(query):
    """Search the web using Tavily API."""
    url = "https://api.tavily.com/search"
    headers = {
        "Authorization": f"Bearer {TAVILY_API_KEY}",
        "Content-Type": "application/json"
    }
    json_data = {
        "query": query,
        "search_depth": "advanced",
        "include_answer": True,
    }
    try:
        res = requests.post(url, headers=headers, json=json_data)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        print(f"Error in search_tavily: {e}")
        return {}

# --- New Combined Tool ---
def search_weather_and_soil(place_name: str):
    """Search for a place via Tavily, then fetch weather + soil + NDVI for it."""
    search_results = search_tavily(f"coordinates of {place_name}")
    if not search_results or "results" not in search_results:
        return {"error": "Could not retrieve location data."}

    lat, lon = None, None

    # Try structured coordinates first
    for result in search_results.get("results", []):
        if "coordinates" in result:
            coords = result["coordinates"]
            lat, lon = coords.get("lat"), coords.get("lon")
            break

    # If not found, extract from answer
    if (lat is None or lon is None) and "answer" in search_results:
        coords = extract_coordinates(search_results["answer"])
        if coords:
            lat, lon = coords

    # If still not found, scan all result contents
    if lat is None or lon is None:
        for result in search_results.get("results", []):
            coords = extract_coordinates(result.get("content", ""))
            if coords:
                lat, lon = coords
                break

    if lat is None or lon is None:
        return {"error": "No coordinates found for the place."}

    return weatherandsoil_search(lat, lon, place_name)

# --- Available tools ---
available_tools = {
    "get_weather": {
        "fn": get_weather,
        "description": "Return current weather for a city.",
        "input_format": {"city": "string"}
    },
    "get_time": {
        "fn": get_time,
        "description": "Get current time by city hint or timezone.",
        "input_format": {"place_or_tz": "string"}
    },
    # "web_search": {
    #     "fn": search_tavily,
    #     "description": "Performs a web search using Tavily API. Use only for generic queries or location info.",
    #     "input_format": {"query": "string"}
    # },
    "search_weather_and_soil": {
        "fn": search_weather_and_soil,
        "description": "Searches a place name, retrieves coordinates, and fetches soil, weather, and NDVI data.",
        "input_format": {"place_name": "string"}
    },
}