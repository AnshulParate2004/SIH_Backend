from .websearch import search_tavily
from .weatherandsoil_search import weatherandsoil_search

def web_search_tool(query: str) -> str:
    try:
        result = search_tavily(query)
        return result.get("answer", "No relevant results found.")
    except Exception as e:
        return f"Error in web_search_tool: {e}"

def weather_and_soil_tool(lat: float, lon: float, name: str) -> str:
    try:
        return weatherandsoil_search(lat, lon, name)
    except Exception as e:
        return f"Error in weather_and_soil_tool: {e}"

available_tools = {
    "web_search": {
        "function": web_search_tool,
        "description": "Performs a web search and returns summarized results.",
        "input_format": {"query": "string"}
    },
    "weather_and_soil": {
        "function": weather_and_soil_tool,
        "description": (
            "Fetches soil conditions, weather forecast, and NDVI vegetation index "
            "for a given location (latitude, longitude, and name)."
        ),
        "input_format": {"lat": "float", "lon": "float", "name": "string"}
    },
}
