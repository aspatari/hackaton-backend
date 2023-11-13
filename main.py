import csv
from datetime import datetime
from operator import getitem

import pandas as pd
import requests
from haversine import haversine
from pyproj import Transformer


def extract_events():
    print("----Getting events---")

    url = "https://data.iharta.md/geoserver/accidente/ows"
    params = {
        "service": "WFS",
        "version": "1.0.0",
        "typeName": "accidente:33b0e9a1-32b0-4646-aac0-00dacad884f8",
        "request": "GetFeature",
        "propertyName": "id,geom,datetime,acc_cause,acc_type,meteo_condition,road_condition,nr_dead,nr_vehicles,nr_injured,nr_dead,nr_vehicles",
        "outputFormat": "application/json",
    }
    response = requests.get(url, params=params, verify=False)
    response.raise_for_status()  # Raise
    data = response.json()
    print(
        f"Find {data['totalFeatures']} accidents",
    )
    return data


def convert_moldavian_datetime(datetime_str):
    # Moldavian months mapping
    moldavian_months = {
        "Ianuarie": 1,
        "Februarie": 2,
        "Martie": 3,
        "Aprilie": 4,
        "Mai": 5,
        "Iunie": 6,
        "Iulie": 7,
        "August": 8,
        "Septembrie": 9,
        "Octombrie": 10,
        "Noiembrie": 11,
        "Decembrie": 12,
    }

    # Split the input string
    date_parts = datetime_str.split(", ")

    # Extract day, month, and year from the first part
    day, month, year = map(str.strip, date_parts[1].split(" "))

    # Replace Moldavian month with numeric month
    month = moldavian_months[month]

    # Extract time from the second part
    time_str = date_parts[2]

    # Combine the date and time parts into a format compatible with strptime
    formatted_datetime_str = f"{day} {month:02d} {year} {time_str}"

    # Parse the formatted string to get the datetime object
    datetime_obj = datetime.strptime(formatted_datetime_str, "%d %m %H:%M %Y")

    return datetime_obj


def determine_day_period(hour):
    if 0 <= hour < 3:
        return "Late Night"
    elif 3 <= hour < 6:
        return "Early Morning"
    elif 6 <= hour < 9:
        return "Morning"
    elif 9 <= hour < 12:
        return "Late Morning"
    elif 12 <= hour < 15:
        return "Afternoon"
    elif 15 <= hour < 18:
        return "Early Afternoon"
    elif 18 <= hour < 21:
        return "Evening"
    else:
        return "Late Evening"


def determine_season(date):
    month = date.month

    if 3 <= month <= 5:
        return "Spring"
    elif 6 <= month <= 8:
        return "Summer"
    elif 9 <= month <= 11:
        return "Autumn"
    else:
        return "Winter"


def calculate_accident_gravity(
    nr_dead,
    nr_injured,
    nr_vehicles,
    weight_dead=20,
    weight_injured=5,
    weight_vehicles=2,
):
    """
    Calculate the gravity of an accident.

    Parameters:
    nr_dead (int): Number of fatalities in the accident.
    nr_injured (int): Number of injuries in the accident.
    nr_vehicles (int): Number of vehicles involved in the accident.
    weight_dead (int): Weight assigned to each fatality. Default is 10.
    weight_injured (int): Weight assigned to each injury. Default is 5.
    weight_vehicles (int): Weight assigned to each vehicle involved. Default is 2.

    Returns:
    int: Severity score of the accident.
    """
    severity_score = (
        (weight_dead * nr_dead)
        + (weight_injured * nr_injured)
        + (weight_vehicles * nr_vehicles)
    )
    return severity_score


transformer = Transformer.from_crs("epsg:4026", "epsg:4326", always_xy=True)


def transform_events(data):
    accidents = data["features"]

    result = []
    for a in accidents:
        dt = convert_moldavian_datetime(a["properties"]["datetime"])
        d = {}
        d["id"] = int(a["properties"]["id"])
        d["datetime"] = dt
        d["weekday"] = dt.weekday()
        d["hour"] = dt.hour
        d["minutes"] = dt.minute
        d["day"] = dt.day
        d["month"] = dt.month
        d["year"] = dt.year
        d["day_period"] = determine_day_period(dt.hour)
        d["seasson"] = determine_season(dt)
        d["acc_cause"] = a["properties"]["acc_cause"]
        d["acc_type"] = a["properties"]["acc_type"]
        d["meteo_condition"] = a["properties"]["meteo_condition"]
        d["road_condition"] = a["properties"]["road_condition"]
        d["nr_injured"] = a["properties"]["nr_injured"]
        d["nr_dead"] = a["properties"]["nr_dead"]
        d["nr_vehicles"] = a["properties"]["nr_vehicles"]
        lon, lat = transformer.transform(
            a["geometry"]["coordinates"][0], a["geometry"]["coordinates"][1]
        )
        d["lat"] = lat
        d["lon"] = lon
        d["gravity"] = calculate_accident_gravity(
            a["properties"]["nr_dead"],
            a["properties"]["nr_injured"],
            a["properties"]["nr_vehicles"],
        )
        result.append(d)
    result.sort(key=lambda x: x["id"])
    return result


def load_to_csv(data):
    with open("result.csv", mode="w", encoding="utf-8") as file:
        columns = [
            "id",
            "datetime",
            "weekday",
            "hour",
            "minutes",
            "day",
            "month",
            "year",
            "day_period",
            "seasson",
            "acc_cause",
            "meteo_condition",
            "road_condition",
            "nr_injured",
            "nr_dead",
            "nr_vehicles",
            "lat",
            "lon",
            "gravity",
        ]
        writer = csv.writer(file)
        writer.writerow(columns)
        for r in data:
            writer.writerow([getitem(r, c) for c in columns])


def filter_accidents_by_radius(accidents_df, current_lat, current_lon, radius):
    """
    Filter accidents within a given radius from a specific point.

    Parameters:
    accidents_df (DataFrame): DataFrame containing accident data with latitude and longitude.
    current_lat (float): Latitude of the current location.
    current_lon (float): Longitude of the current location.
    radius (float): Radius in kilometers.

    Returns:
    DataFrame: Filtered DataFrame containing accidents within the specified radius.
    """

    # Define a function to calculate the distance between two points
    def calculate_distance(lat1, lon1, lat2, lon2):
        return haversine((lat1, lon1), (lat2, lon2))

    # Apply the distance calculation for each accident in the DataFrame
    distances = accidents_df.apply(
        lambda row: calculate_distance(
            current_lat, current_lon, row["lat"], row["lon"]
        ),
        axis=1,
    )

    # Filter accidents where the distance is less than or equal to the specified radius
    filtered_accidents = accidents_df[distances <= radius]

    return filtered_accidents


def get_weather_risk(lon, lat, high_risk=10, mediu_risk=5, low_risk=2):
    url = "https://api.weather.yandex.ru/v2/forecast"

    params = {"lat": lat, "lon": lon, "extra": True, "lang": "en_US"}
    headers = {"X-Yandex-API-Key": "24a79210-197d-4623-9178-c77ede68c748"}

    response = requests.request("GET", url, headers=headers, params=params)

    data = response.json()
    weather = data["fact"]["condition"]
    risk_map = {
        "clear": low_risk,
        "partly-cloudy": low_risk,
        "cloudy": low_risk,
        "overcast": low_risk,
        "drizzle": mediu_risk,
        "light-rain": mediu_risk,
        "rain": high_risk,
        "moderate-rain": high_risk,
        "heavy-rain": high_risk,
        "continuous-heavy-rain": high_risk,
        "showers": high_risk,
        "wet-snow": high_risk,
        "light-snow": mediu_risk,
        "snow": high_risk,
        "snow-showers": mediu_risk,
        "hail": high_risk,
        "thunderstorm": high_risk,
        "thunderstorm-with-rain": high_risk,
        "thunderstorm-with-hail": high_risk,
    }

    return weather, risk_map[weather]


def calculate_probabilities_weather(df, column_name):
    value_counts = df[column_name].value_counts(normalize=True)

    probabilities = {}
    for value, percentage in value_counts.items():
        probabilities[value] = percentage

    return probabilities


def calculate_season_probabilities(df):
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Extract month and map to seasons
    df["month"] = df["datetime"].dt.month
    season_mapping = {
        1: "winter",
        2: "winter",
        3: "spring",
        4: "spring",
        5: "spring",
        6: "summer",
        7: "summer",
        8: "summer",
        9: "autumn",
        10: "autumn",
        11: "autumn",
        12: "winter",
    }
    df["season"] = df["month"].map(season_mapping)

    # Count occurrences of each season
    season_counts = df["season"].value_counts(normalize=True)

    season_probabilities = {}
    for season, percentage in season_counts.items():
        season_probabilities[season] = percentage

    return season_probabilities


def calculate_time_of_day_probabilities(df):
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Extract hour and map to time categories
    df["hour"] = df["datetime"].dt.hour
    time_mapping = {
        0: "late night",
        1: "late night",
        2: "late night",
        3: "early morning",
        4: "early morning",
        5: "early morning",
        6: "morning",
        7: "morning",
        8: "morning",
        9: "late morning",
        10: "late morning",
        11: "late morning",
        12: "afternoon",
        13: "afternoon",
        14: "afternoon",
        15: "late afternoon",
        16: "late afternoon",
        17: "late afternoon",
        18: "evening",
        19: "evening",
        20: "evening",
        21: "night",
        22: "night",
        23: "night",
    }
    df["time_category"] = df["hour"].map(time_mapping)

    # Count occurrences of each time category
    time_category_counts = df["time_category"].value_counts(normalize=True)

    time_of_day_probabilities = {}
    for time_category, percentage in time_category_counts.items():
        time_of_day_probabilities[time_category] = percentage

    return time_of_day_probabilities


def calculate_day_of_week_probabilities(df):
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Extract day of the week
    df["day_of_week"] = df["datetime"].dt.day_name()

    # Count occurrences of each day of the week
    day_counts = df["day_of_week"].value_counts(normalize=True)

    day_of_week_probabilities = {}
    for day, percentage in day_counts.items():
        day_of_week_probabilities[day] = percentage

    return day_of_week_probabilities


weather_mapping = {
    "clear": "Timp senin",
    "partly-cloudy": "Înnourat",
    "cloudy": "Înnourat",
    "overcast": "Înnourat",
    "drizzle": "Înnourat",
    "light-rain": "Ploaie",
    "rain": "Ploaie",
    "moderate-rain": "Ploaie",
    "heavy-rain": "Ploaie",
    "continuous-heavy-rain": "Ploaie",
    "showers": "Ploaie",
    "wet-snow": "Lapoviță",
    "light-snow": "Ninsoare",
    "snow": "Ninsoare",
    "snow-showers": "Ninsoare",
    "hail": "Ninsoare",
    "thunderstorm": "Ploaie",
    "thunderstorm-with-rain": "Ploaie ",
    "thunderstorm-with-hail": "Ploaie",
    "fog": "Ceață",
}


class AccidentProbabilityModel:
    def __init__(self):
        # Conditional probabilities based on historical data
        self.p_weather = None
        self.p_season = None
        self.p_time_of_day = None
        self.p_day_of_week = None
        self.p_zone = None

    def set_dynamic_probabilities(
        self, weather, season, time_of_day, day_of_week, zone
    ):
        self.p_weather = weather
        self.p_season = season
        self.p_time_of_day = time_of_day
        self.p_day_of_week = day_of_week
        self.p_zone = zone

    def calculate_accident_probability(
        self, weather, season, time_of_day, day_of_week, proximity_zone
    ):
        # Check if dynamic probabilities are set
        if (
            self.p_weather is None
            or self.p_season is None
            or self.p_time_of_day is None
            or self.p_day_of_week is None
            or self.p_zone is None
        ):
            raise ValueError(
                "Dynamic probabilities are not set. Use set_dynamic_probabilities method."
            )

        # Calculate conditional probabilities
        p_weather_given_accident = self.p_weather.get(weather, 0)
        p_season_given_accident = self.p_season.get(season, 0)
        p_time_of_day_given_accident = self.p_time_of_day.get(time_of_day, 0)
        p_day_of_week_given_accident = self.p_day_of_week.get(day_of_week, 0)
        p_zone_given_accident = self.p_zone.get(proximity_zone, 0)

        weight_weather = 4.0
        weight_season = 3.5
        weight_time_of_day = 3.8
        weight_day_of_week = 3.2
        weight_zone = 4.5

        # Calculate overall probability of an accident
        p_accident_given_conditions = (
            weight_weather * p_weather_given_accident
            + weight_season * p_season_given_accident
            + weight_time_of_day * p_time_of_day_given_accident
            + weight_day_of_week * p_day_of_week_given_accident
            + weight_zone * p_zone_given_accident
        )

        normalization_factor = sum(
            [
                weight_weather,
                weight_season,
                weight_time_of_day,
                weight_day_of_week,
                weight_zone,
            ]
        )
        normalized_probability = p_accident_given_conditions * 2 / normalization_factor
        print(p_accident_given_conditions)
        print(normalized_probability)
        return normalized_probability


def calculate_current_values():
    current_datetime = datetime.now()

    # Calculate the current season
    current_month = current_datetime.month
    season_mapping = {
        1: "winter",
        2: "winter",
        3: "spring",
        4: "spring",
        5: "spring",
        6: "summer",
        7: "summer",
        8: "summer",
        9: "autumn",
        10: "autumn",
        11: "autumn",
        12: "winter",
    }
    current_season = season_mapping.get(current_month, "unknown")

    # Calculate the current time of day
    current_hour = current_datetime.hour
    time_mapping = {
        (0, 2): "late night",
        (3, 5): "early morning",
        (6, 8): "morning",
        (9, 11): "late morning",
        (12, 14): "afternoon",
        (15, 17): "late afternoon",
        (18, 20): "evening",
        (21, 23): "night",
    }
    current_time_of_day = next(
        (
            category
            for (start, end), category in time_mapping.items()
            if start <= current_hour <= end
        ),
        "unknown",
    )

    # Calculate the current day of the week
    current_day_of_week = current_datetime.strftime("%A")

    return current_season, current_time_of_day, current_day_of_week


def calculate_risk_levels(accidents, k=0.5):
    # Normalizing the counts and weights
    max_count = max(accident["count"] for accident in accidents.values())
    max_weight = max(accident["weight"] for accident in accidents.values())

    # Calculating the risk level for each radius
    risk_levels = {}
    for radius, data in accidents.items():
        normalized_count = data["count"] / max_count
        normalized_weight = data["weight"] / max_weight
        risk_level = k * normalized_count + (1 - k) * normalized_weight
        risk_levels[radius] = risk_level

    return risk_levels


def calculate_risk(lon, lat):
    events = transform_events(extract_events())
    df = pd.DataFrame(events)
    weather_probabilities = calculate_probabilities_weather(df, "meteo_condition")
    season_probabilities = calculate_season_probabilities(df)
    time_of_day_probabilities = calculate_time_of_day_probabilities(df)
    day_of_week_probabilities = calculate_day_of_week_probabilities(df)
    weather, weather_risk = get_weather_risk(lon, lat)
    filtered_accidents_300m = filter_accidents_by_radius(df, lat, lon, 0.3)
    filtered_accidents_500m = filter_accidents_by_radius(df, lat, lon, 0.5)
    filtered_accidents_1000m = filter_accidents_by_radius(df, lat, lon, 1)

    w1000, c1000 = (
        int(filtered_accidents_1000m["gravity"].sum()) or 1,
        len(filtered_accidents_1000m) or 1,
    )
    w500, c500 = (
        int(filtered_accidents_500m["gravity"].sum()),
        len(filtered_accidents_500m) or 1,
    )
    w300, c300 = (
        int(filtered_accidents_300m["gravity"].sum()),
        len(filtered_accidents_300m) or 1,
    )
    model = AccidentProbabilityModel()

    zone_probabilities = calculate_risk_levels(
        {
            "1000m": {"weight": int(w1000), "count": int(c1000)},
            "500m": {"weight": int(w500), "count": int(c500)},
            "300m ": {"weight": int(w300), "count": int(c300)},
        }
    )
    model.set_dynamic_probabilities(
        weather=weather_probabilities,
        season=season_probabilities,
        time_of_day=time_of_day_probabilities,
        day_of_week=day_of_week_probabilities,
        zone=zone_probabilities,
    )
    (
        current_season,
        current_time_of_day,
        current_day_of_week,
    ) = calculate_current_values()
    # Replace with the actual proximity zone data
    current_weather = weather_mapping[weather]  # Replace with the actual weather data
    # Calculate the probability of an accident given the current conditions

    from devtools import debug

    debug(
        {
            "accidents": {
                "1000m": {"weight": w1000, "count": c1000},
                "500m": {"weight": w500, "count": c500},
                "300m ": {"weight": w300, "count": c300},
            },
        }
    )
    return {
        "weather_risk": weather_risk,
        "accidents": {
            "1000m": {"weight": int(w1000), "count": int(c1000)},
            "500m": {"weight": int(w500), "count": int(c500)},
            "300m ": {"weight": int(w300), "count": int(c300)},
        },
        "risk": {
            "300m": model.calculate_accident_probability(
                current_weather,
                current_season,
                current_time_of_day,
                current_day_of_week,
                "300m",
            )
            * weather_risk,
            "500m": model.calculate_accident_probability(
                current_weather,
                current_season,
                current_time_of_day,
                current_day_of_week,
                "500m",
            )
            * weather_risk,
            "1000m": model.calculate_accident_probability(
                current_weather,
                current_season,
                current_time_of_day,
                current_day_of_week,
                "1000m",
            )
            * weather_risk,
        },
    }


if __name__ == "__main__":
    lat = 47.060097235530236
    lon = 28.837394986211194
    r = calculate_risk(lon, lat)

    print(r)
