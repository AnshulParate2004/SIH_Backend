from dotenv import load_dotenv
import os
import requests
import json

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