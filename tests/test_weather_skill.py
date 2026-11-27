import asyncio
import json
from types import SimpleNamespace

from vcb.search.net import Response
from vcb.search.skill import WeatherSkill


class _Route:
    def __init__(self, responses, state="verified"):
        self.responses = iter(responses)
        self.state = state
        self.calls = []

    def fetch(self, spec, timeout):
        self.calls.append((spec, timeout))
        return next(self.responses)


def _response(data):
    return Response(200, "application/json", json.dumps(data).encode("utf-8"))


def _service(responses, state="verified"):
    route = _Route(responses, state)
    return SimpleNamespace(
        route=route,
        cfg=SimpleNamespace(effective_timeout_s=4.0),
    )


def test_weather_uses_private_route_for_geocoding_and_forecast():
    service = _service([
        _response({"results": [{
            "name": "Paris", "admin1": "Ile-de-France", "country": "France",
            "latitude": 48.8566, "longitude": 2.3522,
        }]}),
        _response({
            "current": {
                "temperature_2m": 18, "apparent_temperature": 17,
                "relative_humidity_2m": 60, "precipitation": 0,
                "weather_code": 2, "wind_speed_10m": 12,
            },
            "daily": {
                "time": ["2026-11-27", "2026-11-28", "2026-11-29"],
                "weather_code": [2, 61, 0],
                "temperature_2m_max": [19, 16, 20],
                "temperature_2m_min": [11, 9, 12],
                "precipitation_probability_max": [10, 70, 0],
            },
        }),
    ])

    result = asyncio.run(WeatherSkill(service).run({"location": "Paris"}, None))

    assert "Current weather for Paris, Ile-de-France, France: partly cloudy" in result
    assert "18 C (64 F)" in result
    assert "Saturday: light rain" in result
    assert "precipitation chance 70%" in result
    assert "Source: Open-Meteo" in result
    geocode, forecast = service.route.calls
    assert geocode[0].url == "https://geocoding-api.open-meteo.com/v1/search"
    assert geocode[0].params["name"] == "Paris"
    assert forecast[0].url == "https://api.open-meteo.com/v1/forecast"
    assert forecast[0].params["forecast_days"] == "3"
    assert geocode[1] == forecast[1] == 4.0
    assert WeatherSkill.sensitive and WeatherSkill.untrusted


def test_weather_does_not_send_without_location_or_verified_route():
    missing_location = _service([])
    result = asyncio.run(WeatherSkill(missing_location).run({}, None))
    assert "Tell me a city" in result
    assert not missing_location.route.calls

    unavailable = _service([], state="starting")
    result = asyncio.run(WeatherSkill(unavailable).run({"location": "Paris"}, None))
    assert "private online route isn't ready" in result
    assert not unavailable.route.calls


def test_weather_handles_place_not_found_and_bad_provider_data():
    no_place = _service([_response({"results": []})])
    result = asyncio.run(WeatherSkill(no_place).run({"location": "Atlantis"}, None))
    assert "couldn't find that place" in result
    assert len(no_place.route.calls) == 1

    invalid_geocode = _service([_response({"error": True})])
    result = asyncio.run(WeatherSkill(invalid_geocode).run({"location": "Paris"}, None))
    assert "provider returned incomplete data" in result

    incomplete = _service([
        _response({"results": [{
            "name": "Paris", "latitude": 48.8566, "longitude": 2.3522,
        }]}),
        _response({"current": {}, "daily": {}}),
    ])
    result = asyncio.run(WeatherSkill(incomplete).run({"location": "Paris"}, None))
    assert "provider returned incomplete data" in result


