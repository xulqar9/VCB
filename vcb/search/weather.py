"""Structured weather lookups through the already-verified private search route."""

from __future__ import annotations

import asyncio
import json
import math
from datetime import date

from .backends import RequestSpec
from .net import FetchError, RouteUnavailable, _json
from .sanitize import clean_query, clean_text

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_CURRENT_FIELDS = (
    "temperature_2m,relative_humidity_2m,apparent_temperature,"
    "precipitation,weather_code,wind_speed_10m"
)
_DAILY_FIELDS = (
    "weather_code,temperature_2m_max,temperature_2m_min,"
    "precipitation_probability_max"
)
_WEATHER_CODES = {
    0: "clear", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "dense drizzle",
    56: "light freezing drizzle", 57: "dense freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "light freezing rain", 67: "heavy freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light rain showers", 81: "rain showers", 82: "heavy rain showers",
    85: "light snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with light hail",
    99: "thunderstorm with heavy hail",
}


class WeatherDataError(ValueError):
    """The provider returned an incomplete or invalid weather response."""


def _number(data: dict, key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool):
        raise WeatherDataError("invalid numeric field")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise WeatherDataError("invalid numeric field") from None
    if not math.isfinite(number):
        raise WeatherDataError("invalid numeric field")
    return number


def _weather(code: object) -> str:
    if isinstance(code, bool):
        raise WeatherDataError("invalid weather code")
    try:
        number = float(code)
    except (TypeError, ValueError):
        raise WeatherDataError("invalid weather code") from None
    if not math.isfinite(number) or not number.is_integer():
        raise WeatherDataError("invalid weather code")
    value = int(number)
    return _WEATHER_CODES.get(value, f"weather code {value}")


def _celsius_fahrenheit(celsius: float) -> str:
    fahrenheit = celsius * 9 / 5 + 32
    return f"{celsius:.0f} C ({fahrenheit:.0f} F)"


def _format_weather(place: str, forecast: dict) -> str:
    current = forecast.get("current")
    daily = forecast.get("daily")
    if not isinstance(current, dict) or not isinstance(daily, dict):
        raise WeatherDataError("missing weather fields")

    condition = _weather(current.get("weather_code"))
    temp = _number(current, "temperature_2m")
    feels = _number(current, "apparent_temperature")
    humidity = _number(current, "relative_humidity_2m")
    wind = _number(current, "wind_speed_10m")
    precipitation = _number(current, "precipitation")
    lines = [
        f"Current weather for {place}: {condition}, {_celsius_fahrenheit(temp)}, "
        f"feels like {_celsius_fahrenheit(feels)}, humidity {humidity:.0f}%, "
        f"wind {wind:.0f} km/h, precipitation {precipitation:g} mm."
    ]

    dates = daily.get("time")
    codes = daily.get("weather_code")
    highs = daily.get("temperature_2m_max")
    lows = daily.get("temperature_2m_min")
    rain = daily.get("precipitation_probability_max")
    if not all(isinstance(values, list) for values in (dates, codes, highs, lows, rain)):
        raise WeatherDataError("missing daily forecast")
    lengths = {len(values) for values in (dates, codes, highs, lows, rain)}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) < 3:
        raise WeatherDataError("incomplete daily forecast")
    count = min(next(iter(lengths)), 3)

    for i in range(count):
        try:
            day = date.fromisoformat(str(dates[i])).strftime("%A")
        except ValueError:
            raise WeatherDataError("invalid forecast date") from None
        high, low = _number({"value": highs[i]}, "value"), _number({"value": lows[i]}, "value")
        chance = _number({"value": rain[i]}, "value")
        lines.append(
            f"{day}: {_weather(codes[i])}, high {_celsius_fahrenheit(high)}, "
            f"low {_celsius_fahrenheit(low)}, precipitation chance {chance:.0f}%."
        )
    lines.append("Source: Open-Meteo weather forecast; GeoNames location data.")
    return "\n".join(lines)


def _lookup_sync(service, location: str, timeout: float) -> str:
    route = service.route
    geocoded = _json(route.fetch(RequestSpec(
        _GEOCODE_URL,
        {"name": location, "count": "1", "language": "en", "format": "json"},
    ), timeout))
    places = geocoded.get("results")
    if not isinstance(places, list):
        raise WeatherDataError("invalid location results")
    if not places:
        return "I couldn't find that place. Try a city and country."
    place = places[0]
    if not isinstance(place, dict):
        raise WeatherDataError("invalid location result")
    latitude = _number(place, "latitude")
    longitude = _number(place, "longitude")
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise WeatherDataError("invalid coordinates")
    name = clean_text(place.get("name"), 80)
    if not name:
        raise WeatherDataError("missing place name")
    region = clean_text(place.get("admin1"), 60)
    country = clean_text(place.get("country"), 60)
    place_name = ", ".join(part for part in (name, region, country) if part)

    forecast = _json(route.fetch(RequestSpec(
        _FORECAST_URL,
        {
            "latitude": str(latitude),
            "longitude": str(longitude),
            "current": _CURRENT_FIELDS,
            "daily": _DAILY_FIELDS,
            "timezone": "auto",
            "forecast_days": "3",
        },
    ), timeout))
    return _format_weather(place_name, forecast)


async def lookup_weather(service, location: str) -> str:
    """Return current conditions and up to three forecast days, fail-closed."""
    city = clean_query(location, limit=100)
    if not city:
        return "Tell me a city and, if useful, its country so I can check the weather."
    if getattr(getattr(service, "route", None), "state", None) != "verified":
        return ("Weather is unavailable because the private online route isn't ready; "
                "nothing was sent.")

    timeout = service.cfg.effective_timeout_s
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_lookup_sync, service, city, timeout),
            timeout=timeout * 2 + 0.5,
        )
    except asyncio.TimeoutError:
        return "The weather lookup timed out. Please try again."
    except RouteUnavailable:
        return ("Weather is unavailable because the private online route isn't ready; "
                "nothing was sent.")
    except FetchError:
        return "The weather lookup failed through the private route. Please try again later."
    except WeatherDataError:
        return "The weather provider returned incomplete data. Please try again later."
