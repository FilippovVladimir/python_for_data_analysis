import math
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import matplotlib.pyplot as plt


# 1) Данные (высокая точность, 6 знаков)
cities_hi = [
    # Техас
    {"city": "Austin", "state": "Texas", "coords": (30.267220, -97.743060), "tz": "America/Chicago", "cap": True,  "density": 3006.36},
    {"city": "Houston", "state": "Texas", "coords": (29.762780, -95.383060), "tz": "America/Chicago", "cap": False, "density": 3613.38},
    {"city": "Dallas", "state": "Texas", "coords": (32.779170, -96.806940), "tz": "America/Chicago", "cap": False, "density": 3840.90},
    {"city": "San Antonio", "state": "Texas", "coords": (29.425000, -98.493890), "tz": "America/Chicago", "cap": False, "density": 3001.91},
    {"city": "El Paso", "state": "Texas", "coords": (31.759170, -106.488610), "tz": "America/Denver", "cap": False, "density": 2626.69},

    # Флорида
    {"city": "Tallahassee", "state": "Florida", "coords": (30.438260, -84.280730), "tz": "America/New_York", "cap": True,  "density": 1926.00},
    {"city": "Miami", "state": "Florida", "coords": (25.780000, -80.210000), "tz": "America/New_York", "cap": False, "density": 12284.47},
    {"city": "Orlando", "state": "Florida", "coords": (28.540000, -81.380000), "tz": "America/New_York", "cap": False, "density": 2774.65},
    {"city": "Tampa", "state": "Florida", "coords": (27.950000, -82.460000), "tz": "America/New_York", "cap": False, "density": 3376.40},
    {"city": "Pensacola", "state": "Florida", "coords": (30.421390, -87.217220), "tz": "America/Chicago", "cap": False, "density": 2359.48},

    # Калифорния
    {"city": "Sacramento", "state": "California", "coords": (38.581600, -121.494400), "tz": "America/Los_Angeles", "cap": True,  "density": 5323.40},
    {"city": "Los Angeles", "state": "California", "coords": (34.052200, -118.243700), "tz": "America/Los_Angeles", "cap": False, "density": 8304.20},
    {"city": "San Diego", "state": "California", "coords": (32.715700, -117.161100), "tz": "America/Los_Angeles", "cap": False, "density": 4234.24},
    {"city": "San Francisco", "state": "California", "coords": (37.774900, -122.419400), "tz": "America/Los_Angeles", "cap": False, "density": 18634.65},
    {"city": "Fresno", "state": "California", "coords": (36.737800, -119.787100), "tz": "America/Los_Angeles", "cap": False, "density": 4795.54},
]


# создаем новый список городов с округленными координатами
def make_low_precision(cities, ndigits=2):
    cities_lo = []
    for c in cities:
        lat, lon = c["coords"]
        c2 = c.copy()
        c2["coords"] = (round(lat, ndigits), round(lon, ndigits))
        cities_lo.append(c2)
    return cities_lo


cities_lo = make_low_precision(cities_hi, 2)


# расстояние (формула гаверсинуса), ввод - градусы
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


CAPITAL_USA = (38.895110, -77.036370)  # Вашингтон


def add_distances(cities, usa_cap_coords):
    # добавка расстояния до столицы США
    for c in cities:
        lat, lon = c["coords"]
        c["dist_country_km"] = haversine_km(lat, lon, usa_cap_coords[0], usa_cap_coords[1])


def add_state_distances(cities):
    # собтраем координаты столиц штатов
    state_cap = {}
    for c in cities:
        if c["cap"]:
            state_cap[c["state"]] = c["coords"]

    # считаем расстояния до столицы своего штата
    for c in cities:
        cap_lat, cap_lon = state_cap[c["state"]]
        lat, lon = c["coords"]
        c["dist_state_km"] = haversine_km(lat, lon, cap_lat, cap_lon)


def key_city(c):
    return (c["city"], c["state"])


def compare_errors(cities_hi_list, cities_lo_list):
    # делаем словарь для быстрого доступа по ключу (город, штат)
    lo_map = {key_city(c): c for c in cities_lo_list}

    rows = []
    for c in cities_hi_list:
        k = key_city(c)
        c_lo = lo_map[k]

        err_country = abs(c_lo["dist_country_km"] - c["dist_country_km"]) / c["dist_country_km"] * 100
        err_state = abs(c_lo["dist_state_km"] - c["dist_state_km"]) / c["dist_state_km"] * 100 if c["dist_state_km"] != 0 else 0.0

        rows.append({
            "city": c["city"],
            "state": c["state"],
            "err_country_%": err_country,
            "err_state_%": err_state,
        })

    # сортируем по ошибке
    rows.sort(key=lambda x: x["err_country_%"], reverse=True)
    return rows


# Часовые пояса
def timezones_per_state(cities):
    d = {}
    for c in cities:
        st = c["state"]
        d.setdefault(st, set()).add(c["tz"])
    return {st: len(tzs) for st, tzs in d.items()}


def utc_offset_hours(dt_utc, tz_name):
    local = dt_utc.astimezone(ZoneInfo(tz_name))
    return local.utcoffset().total_seconds() / 3600


def longitude_offset_correlation(cities):
    # фиксируем дату, чтобы DST меньше мешал
    dt = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)

    lons = []
    offsets = []
    for c in cities:
        lat, lon = c["coords"]
        lons.append(lon)
        offsets.append(utc_offset_hours(dt, c["tz"]))

    # корреляция Пирсона
    return float(np.corrcoef(lons, offsets)[0, 1])


# Визуализация (точки lon/lat)
def plot_points(cities):
    plt.figure()
    states = sorted(set(c["state"] for c in cities))
    for st in states:
        xs = [c["coords"][1] for c in cities if c["state"] == st]  # lon
        ys = [c["coords"][0] for c in cities if c["state"] == st]  # lat
        plt.scatter(xs, ys, label=st)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Cities on lon/lat plane")
    plt.legend()
    plt.show()


def main():
    # добавляем расстояния для hi и lo
    add_distances(cities_hi, CAPITAL_USA)
    add_state_distances(cities_hi)

    add_distances(cities_lo, CAPITAL_USA)
    add_state_distances(cities_lo)

    # самый дальний от столицы США (hi)
    far = max(cities_hi, key=lambda c: c["dist_country_km"])
    print("Farthest from Washington, D.C. (hi):", far["city"], "-", far["state"], "-", round(far["dist_country_km"], 1), "km")

    # самый дальний от столицы своего штата (hi)
    far_state = max(cities_hi, key=lambda c: c["dist_state_km"])
    print("Farthest from own state capital (hi):", far_state["city"], "-", far_state["state"], "-", round(far_state["dist_state_km"], 1), "km")

    # ошибки от округления
    errors = compare_errors(cities_hi, cities_lo)
    print("\nTop 5 errors after rounding (percent):")
    for r in errors[:5]:
        print(r["city"], "-", r["state"], "country:", round(r["err_country_%"], 3), "%", "state:", round(r["err_state_%"], 3), "%")

    # часовые пояса
    tz_counts = timezones_per_state(cities_hi)
    print("\nTime zones per state:", tz_counts)

    # корреляция долгота - utc offset
    corr = longitude_offset_correlation(cities_hi)
    print("\nCorrelation (longitude vs UTC offset hours):", round(corr, 3))

    # график
    plot_points(cities_hi)


main()
