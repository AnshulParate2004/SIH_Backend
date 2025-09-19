# import requests
# import time
# from datetime import datetime, timezone

# # --- Configuration ---
# AGRO_API_KEY = "e4a991d1363456ecbe9b5b0895ae2083"
# AGRO_BASE_URL = "https://api.agromonitoring.com/agro/1.0"


# # --- Helper functions ---
# def kelvin_to_celsius(kelvin):
#     return round(kelvin - 273.15, 2)


# def get_existing_polygons():
#     url = f"{AGRO_BASE_URL}/polygons?appid={AGRO_API_KEY}"
#     r = requests.get(url)
#     r.raise_for_status()
#     return r.json()


# def create_polygon(lat, lon, name="my_polygon"):
#     delta = 0.01  # ~1 km offset
#     coords = [
#         [lon - delta, lat - delta],
#         [lon - delta, lat + delta],
#         [lon + delta, lat + delta],
#         [lon + delta, lat - delta],
#         [lon - delta, lat - delta],
#     ]

#     poly_data = {
#         "name": name,
#         "geo_json": {
#             "type": "Feature",
#             "properties": {},
#             "geometry": {"type": "Polygon", "coordinates": [coords]},
#         },
#     }

#     url = f"{AGRO_BASE_URL}/polygons?appid={AGRO_API_KEY}"
#     r = requests.post(url, json=poly_data)
#     r.raise_for_status()
#     return r.json()


# def get_or_create_polygon(lat, lon, name):
#     polygons = get_existing_polygons()
#     for poly in polygons:
#         if poly["name"] == name:
#             print(f"   🔄 Reusing existing polygon: {poly['id']}")
#             return poly
#     print("   ➕ Creating new polygon...")
#     return create_polygon(lat, lon, name)


# def fetch_soil_data(polygon_id):
#     url = f"{AGRO_BASE_URL}/soil?polyid={polygon_id}&appid={AGRO_API_KEY}"
#     r = requests.get(url)
#     r.raise_for_status()
#     data = r.json()

#     return {
#         "t0_celsius": kelvin_to_celsius(data.get("t0", 0)),
#         "t10_celsius": kelvin_to_celsius(data.get("t10", 0)),
#         "moisture": data.get("moisture"),
#         "timestamp": datetime.fromtimestamp(
#             data.get("ts", 0), tz=timezone.utc
#         ).isoformat(),
#     }


# def fetch_weather_openmeteo(lat, lon, hours=24):
#     """
#     Get current weather + rainfall forecast using Open-Meteo API.
#     """
#     url = "https://api.open-meteo.com/v1/forecast"
#     params = {
#         "latitude": lat,
#         "longitude": lon,
#         "current": ["temperature_2m", "relative_humidity_2m",
#                     "precipitation", "precipitation_probability"],
#         "hourly": ["temperature_2m", "relative_humidity_2m",
#                    "precipitation", "precipitation_probability"],
#         "forecast_hours": hours,
#         "timezone": "auto",
#     }

#     r = requests.get(url, params=params)
#     r.raise_for_status()
#     data = r.json()

#     result = {
#         "current": {
#             "temperature": data["current"].get("temperature_2m"),
#             "humidity": data["current"].get("relative_humidity_2m"),
#             "precipitation": data["current"].get("precipitation"),
#             "precipitation_probability": data["current"].get("precipitation_probability"),
#         },
#         "forecast": []
#     }

#     for i, t in enumerate(data["hourly"]["time"][:hours]):
#         result["forecast"].append({
#             "time": t,
#             "temperature": data["hourly"]["temperature_2m"][i],
#             "humidity": data["hourly"]["relative_humidity_2m"][i],
#             "precipitation": data["hourly"]["precipitation"][i],
#             "precipitation_probability": data["hourly"]["precipitation_probability"][i],
#         })

#     return result


# def fetch_ndvi(polygon_id):
#     """Fetch latest NDVI for polygon (last 30 days)."""
#     end = int(time.time())
#     start = end - 30 * 24 * 3600  # last 30 days
#     url = f"{AGRO_BASE_URL}/ndvi/history?polyid={polygon_id}&start={start}&end={end}&appid={AGRO_API_KEY}"

#     r = requests.get(url)
#     r.raise_for_status()
#     data = r.json()

#     if not data:
#         return None

#     latest = max(data, key=lambda x: x["dt"])
#     return {
#         "date": datetime.fromtimestamp(latest["dt"], tz=timezone.utc).isoformat(),
#         "ndvi": round(latest["data"]["mean"], 3),
#     }


# def weatherandsoil_search(lat, lon, name):
#     print(f"\n📍 Location: {name} at ({lat}, {lon})")

#     result = {}

#     # Polygon
#     polygon = get_or_create_polygon(lat, lon, name)
#     polygon_id = polygon["id"]
#     print(f"   Polygon ID: {polygon_id} | Area: {round(polygon['area']/10000, 2)} ha")
#     result["polygon_id"] = polygon_id
#     result["polygon_area_ha"] = round(polygon["area"]/10000, 2)

#     # Soil
#     print("   🌱 Soil Data...")
#     soil = None
#     try:
#         soil = fetch_soil_data(polygon_id)
#         print(f"   ✅ {soil}")
#         result["soil"] = soil
#     except Exception as e:
#         print(f"   ❌ Soil error: {e}")
#         result["soil"] = None

#     # Weather
#     print("   🌧 Weather & Rainfall (Open-Meteo)...")
#     weather = None
#     try:
#         weather = fetch_weather_openmeteo(lat, lon, hours=24)
#         print(f"   ✅ Current: {weather['current']}")
#         print("      ⏳ Next 24h forecast:")
#         for f in weather["forecast"]:
#             print(f"         {f['time']}: {f['precipitation']} mm rain ({f['precipitation_probability']}% chance)")
#         result["weather"] = weather
#     except Exception as e:
#         print(f"   ❌ Weather error: {e}")
#         result["weather"] = None

#     # Vegetation (NDVI)
#     print("   🍃 Vegetation (NDVI)...")
#     ndvi = None
#     try:
#         ndvi = fetch_ndvi(polygon_id)
#         if ndvi:
#             print(f"   ✅ NDVI {ndvi['ndvi']} (as of {ndvi['date']})")
#         else:
#             print("   ❌ No NDVI data available")
#         result["ndvi"] = ndvi
#     except Exception as e:
#         print(f"   ❌ NDVI error: {e}")
#         result["ndvi"] = None

#     return result






















































import requests
import time
from datetime import datetime, timezone

# --- Configuration ---
AGRO_API_KEY = "e4a991d1363456ecbe9b5b0895ae2083"
AGRO_BASE_URL = "https://api.agromonitoring.com/agro/1.0"

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




















































######### Test
# if __name__ == "__main__":
#     # Run for Kollur Mine
#     weatherandsoil_search(16.717, 80.033, "Kollur Mine")

#     # Run for another mine (example: Kolar Gold Fields)
#     weatherandsoil_search(13.3, 77.583, "Kolar Gold Fields")
