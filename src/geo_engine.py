import pandas as pd
import re
import time
import difflib 
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

class GeoAnalyzer:
    """
    Hybrid Geo-Engine V6 (Export-Data Optimized)
    Includes massive dictionary update based on '2026-01-05T11-32_export.csv'
    """
    
    def __init__(self):
        self.geolocator = Nominatim(user_agent="customer_hub_v6_export")
        
        # --- TIER 1: IATA CODES (Massive Expansion) ---
        self.iata_db = {
            # Major Global
            "LHR": "London", "LGW": "London", "LCY": "London", "STN": "London",
            "JFK": "New York", "EWR": "Newark", "LGA": "New York",
            "CDG": "Paris", "ORY": "Paris", "DXB": "Dubai", "DOH": "Doha",
            "HND": "Tokyo", "NRT": "Tokyo", "SIN": "Singapore", "HKG": "Hong Kong",
            "LAX": "Los Angeles", "SFO": "San Francisco", "ORD": "Chicago",
            "FRA": "Frankfurt", "MUC": "Munich", "AMS": "Amsterdam", "MAD": "Madrid",
            "BCN": "Barcelona", "FCO": "Rome", "MXP": "Milan", "GIG": "Rio De Janeiro",
            "GRU": "Sao Paulo", "SYD": "Sydney", "MEL": "Melbourne", "BKK": "Bangkok",
            "DEL": "Delhi", "BOM": "Mumbai", "YYZ": "Toronto", "YVR": "Vancouver",
            "CPT": "Cape Town", "JNB": "Johannesburg",
            
            # UK / Europe Expansion from Log
            "GLA": "Glasgow", "EDI": "Edinburgh", "MAN": "Manchester", "NCL": "Newcastle",
            "ABZ": "Aberdeen", "LBA": "Leeds", "BHX": "Birmingham", "BRS": "Bristol",
            "DUB": "Dublin", "ORK": "Cork", "SNN": "Shannon", "BFS": "Belfast",
            "ZRH": "Zurich", "GVA": "Geneva", "BSL": "Basel", "VIE": "Vienna",
            "PRG": "Prague", "BUD": "Budapest", "WAW": "Warsaw", "KRK": "Krakow",
            "ATH": "Athens", "SKG": "Thessaloniki", "HER": "Heraklion", "RHO": "Rhodes",
            "LCA": "Larnaca", "PFO": "Paphos", "MLA": "Malta", "IST": "Istanbul",
            "AYT": "Antalya", "DLM": "Dalaman", "BOD": "Bordeaux", "NCE": "Nice",
            "MRS": "Marseille", "LYS": "Lyon", "TLS": "Toulouse", "LIS": "Lisbon",
            "OPO": "Porto", "FAO": "Faro", "FNC": "Madeira", "MXP": "Milan",
            "LIN": "Milan", "VCE": "Venice", "PSA": "Pisa", "NAP": "Naples",
            "BLQ": "Bologna", "PMO": "Palermo", "CTA": "Catania", "BRI": "Bari",
            "TRN": "Turin", "VRN": "Verona", "OSL": "Oslo", "ARN": "Stockholm",
            "CPH": "Copenhagen", "HEL": "Helsinki", "KEF": "Reykjavik", "GOT": "Gothenburg",
            "SVG": "Stavanger", "BGO": "Bergen", "DUS": "Dusseldorf", "HAM": "Hamburg",
            "TXL": "Berlin", "BER": "Berlin", "STR": "Stuttgart", "CGN": "Cologne",
            "BRU": "Brussels", "LUX": "Luxembourg", "OTP": "Bucharest", "SOF": "Sofia",
            "ZAG": "Zagreb", "DBV": "Dubrovnik", "SPU": "Split", "TIA": "Tirana",
            
            # Americas Expansion
            "IAD": "Washington", "DCA": "Washington", "BWI": "Baltimore", "BOS": "Boston",
            "PHL": "Philadelphia", "MIA": "Miami", "FLL": "Fort Lauderdale", "MCO": "Orlando",
            "TPA": "Tampa", "ATL": "Atlanta", "DFW": "Dallas", "IAH": "Houston",
            "AUS": "Austin", "DEN": "Denver", "LAS": "Las Vegas", "PHX": "Phoenix",
            "SAN": "San Diego", "SEA": "Seattle", "PDX": "Portland", "SJC": "San Jose",
            "OAK": "Oakland", "YUL": "Montreal", "YYC": "Calgary", "YEG": "Edmonton",
            "YOW": "Ottawa", "MEX": "Mexico City", "CUN": "Cancun", "SJO": "San Jose",
            "PTY": "Panama City", "LIM": "Lima", "BOG": "Bogota", "SCL": "Santiago",
            "EZE": "Buenos Aires", "PUJ": "Punta Cana", "NAS": "Nassau", "BGI": "Bridgetown",
            "UVF": "St Lucia", "ANU": "Antigua", "GND": "Grenada", "MBJ": "Montego Bay",
            "KIN": "Kingston", "POS": "Port Of Spain",
            
            # Asia / ME / Africa Expansion
            "RUH": "Riyadh", "JED": "Jeddah", "KWI": "Kuwait", "BAH": "Bahrain",
            "AUH": "Abu Dhabi", "MCT": "Muscat", "AMM": "Amman", "TLV": "Tel Aviv",
            "CAI": "Cairo", "CMN": "Casablanca", "RAK": "Marrakech", "LOS": "Lagos",
            "ABV": "Abuja", "ACC": "Accra", "NBO": "Nairobi", "MRU": "Mauritius",
            "SEZ": "Seychelles", "MLE": "Male", "PEK": "Beijing", "PVG": "Shanghai",
            "CAN": "Guangzhou", "ICN": "Seoul", "KUL": "Kuala Lumpur", "MNL": "Manila",
            "CGK": "Jakarta", "SGN": "Ho Chi Minh City", "HAN": "Hanoi", "BLR": "Bangalore",
            "MAA": "Chennai", "HYD": "Hyderabad", "ISB": "Islamabad", "LHE": "Lahore",
            "KHI": "Karachi", "CMB": "Colombo"
        }

        # --- TIER 2: STATIC DB (Explicit Coordinates) ---
        self.static_db = {
            # --- UK & Ireland ---
            "london": {"lat": 51.5074, "lon": -0.1278, "iso": "GBR"},
            "gatwick": {"lat": 51.1537, "lon": -0.1821, "iso": "GBR"},
            "heathrow": {"lat": 51.4700, "lon": -0.4543, "iso": "GBR"},
            "manchester": {"lat": 53.4808, "lon": -2.2426, "iso": "GBR"},
            "edinburgh": {"lat": 55.9533, "lon": -3.1883, "iso": "GBR"},
            "glasgow": {"lat": 55.8642, "lon": -4.2518, "iso": "GBR"},
            "belfast": {"lat": 54.5973, "lon": -5.9301, "iso": "GBR"},
            "dublin": {"lat": 53.3498, "lon": -6.2603, "iso": "IRL"},
            "newcastle": {"lat": 54.9783, "lon": -1.6178, "iso": "GBR"},
            "aberdeen": {"lat": 57.1497, "lon": -2.0943, "iso": "GBR"},
            "leeds": {"lat": 53.8008, "lon": -1.5491, "iso": "GBR"},
            "leeds bradford": {"lat": 53.8659, "lon": -1.6606, "iso": "GBR"},
            "inverness": {"lat": 57.4778, "lon": -4.2247, "iso": "GBR"},
            "jersey": {"lat": 49.2144, "lon": -2.1313, "iso": "JEY"},
            
            # --- Europe ---
            "paris": {"lat": 48.8566, "lon": 2.3522, "iso": "FRA"},
            "nice": {"lat": 43.7102, "lon": 7.2620, "iso": "FRA"},
            "marseille": {"lat": 43.2965, "lon": 5.3698, "iso": "FRA"},
            "lyon": {"lat": 45.7640, "lon": 4.8357, "iso": "FRA"},
            "toulouse": {"lat": 43.6047, "lon": 1.4442, "iso": "FRA"},
            "bordeaux": {"lat": 44.8378, "lon": -0.5792, "iso": "FRA"},
            "amsterdam": {"lat": 52.3676, "lon": 4.9041, "iso": "NLD"},
            "rotterdam": {"lat": 51.9244, "lon": 4.4777, "iso": "NLD"},
            "brussels": {"lat": 50.8503, "lon": 4.3517, "iso": "BEL"},
            "luxembourg": {"lat": 49.6116, "lon": 6.1319, "iso": "LUX"},
            "frankfurt": {"lat": 50.1109, "lon": 8.6821, "iso": "DEU"},
            "munich": {"lat": 48.1351, "lon": 11.5820, "iso": "DEU"},
            "berlin": {"lat": 52.5200, "lon": 13.4050, "iso": "DEU"},
            "hamburg": {"lat": 53.5511, "lon": 9.9937, "iso": "DEU"},
            "dusseldorf": {"lat": 51.2277, "lon": 6.7735, "iso": "DEU"},
            "stuttgart": {"lat": 48.7758, "lon": 9.1829, "iso": "DEU"},
            "cologne": {"lat": 50.9375, "lon": 6.9603, "iso": "DEU"},
            "hannover": {"lat": 52.3759, "lon": 9.7320, "iso": "DEU"},
            "rome": {"lat": 41.9028, "lon": 12.4964, "iso": "ITA"},
            "milan": {"lat": 45.4642, "lon": 9.1900, "iso": "ITA"},
            "venice": {"lat": 45.4408, "lon": 12.3155, "iso": "ITA"},
            "naples": {"lat": 40.8518, "lon": 14.2681, "iso": "ITA"},
            "florence": {"lat": 43.7696, "lon": 11.2558, "iso": "ITA"},
            "bologna": {"lat": 44.4949, "lon": 11.3426, "iso": "ITA"},
            "pisa": {"lat": 43.7228, "lon": 10.4017, "iso": "ITA"},
            "turin": {"lat": 45.0703, "lon": 7.6869, "iso": "ITA"},
            "verona": {"lat": 45.4384, "lon": 10.9916, "iso": "ITA"},
            "bari": {"lat": 41.1171, "lon": 16.8719, "iso": "ITA"},
            "catania": {"lat": 37.5079, "lon": 15.0830, "iso": "ITA"},
            "brindisi": {"lat": 40.6327, "lon": 17.9418, "iso": "ITA"},
            "madrid": {"lat": 40.4168, "lon": -3.7038, "iso": "ESP"},
            "barcelona": {"lat": 41.3851, "lon": 2.1734, "iso": "ESP"},
            "seville": {"lat": 37.3891, "lon": -5.9845, "iso": "ESP"},
            "malaga": {"lat": 36.7213, "lon": -4.4214, "iso": "ESP"},
            "valencia": {"lat": 39.4699, "lon": -0.3763, "iso": "ESP"},
            "palma": {"lat": 39.5696, "lon": 2.6502, "iso": "ESP"},
            "palma de mallorca": {"lat": 39.5696, "lon": 2.6502, "iso": "ESP"},
            "ibiza": {"lat": 38.9067, "lon": 1.4206, "iso": "ESP"},
            "tenerife": {"lat": 28.2916, "lon": -16.6291, "iso": "ESP"},
            "fuerteventura": {"lat": 28.3587, "lon": -14.0536, "iso": "ESP"},
            "fuereventura": {"lat": 28.3587, "lon": -14.0536, "iso": "ESP"}, # Typos
            "lisbon": {"lat": 38.7223, "lon": -9.1393, "iso": "PRT"},
            "porto": {"lat": 41.1579, "lon": -8.6291, "iso": "PRT"},
            "faro": {"lat": 37.0194, "lon": -7.9304, "iso": "PRT"},
            "madeira": {"lat": 32.7607, "lon": -16.9595, "iso": "PRT"},
            "funchal": {"lat": 32.6505, "lon": -16.9085, "iso": "PRT"},
            "athens": {"lat": 37.9838, "lon": 23.7275, "iso": "GRC"},
            "santorini": {"lat": 36.3932, "lon": 25.4615, "iso": "GRC"},
            "mykonos": {"lat": 37.4467, "lon": 25.3289, "iso": "GRC"},
            "corfu": {"lat": 39.6243, "lon": 19.9217, "iso": "GRC"},
            "rhodes": {"lat": 36.4349, "lon": 28.2175, "iso": "GRC"},
            "heraklion": {"lat": 35.3387, "lon": 25.1442, "iso": "GRC"},
            "kos": {"lat": 36.8872, "lon": 27.2877, "iso": "GRC"},
            "zakinthos": {"lat": 37.7878, "lon": 20.8987, "iso": "GRC"},
            "cephalonia": {"lat": 38.1811, "lon": 20.4850, "iso": "GRC"},
            "kalamata": {"lat": 37.0420, "lon": 22.1142, "iso": "GRC"},
            "preveza": {"lat": 38.9575, "lon": 20.7565, "iso": "GRC"},
            "istanbul": {"lat": 41.0082, "lon": 28.9784, "iso": "TUR"},
            "dalaman": {"lat": 36.7659, "lon": 28.8037, "iso": "TUR"},
            "zurich": {"lat": 47.3769, "lon": 8.5417, "iso": "CHE"},
            "geneva": {"lat": 46.2044, "lon": 6.1432, "iso": "CHE"},
            "basel": {"lat": 47.5596, "lon": 7.5886, "iso": "CHE"},
            "vienna": {"lat": 48.2082, "lon": 16.3738, "iso": "AUT"},
            "salzburg": {"lat": 47.8095, "lon": 13.0550, "iso": "AUT"},
            "copenhagen": {"lat": 55.6761, "lon": 12.5683, "iso": "DNK"},
            "stockholm": {"lat": 59.3293, "lon": 18.0686, "iso": "SWE"},
            "sweden": {"lat": 59.3293, "lon": 18.0686, "iso": "SWE"}, # Country Mapping
            "gothenburg": {"lat": 57.7089, "lon": 11.9746, "iso": "SWE"},
            "oslo": {"lat": 59.9139, "lon": 10.7522, "iso": "NOR"},
            "bergen": {"lat": 60.3913, "lon": 5.3221, "iso": "NOR"},
            "stavanger": {"lat": 58.9690, "lon": 5.7331, "iso": "NOR"},
            "helsinki": {"lat": 60.1695, "lon": 24.9354, "iso": "FIN"},
            "oulu": {"lat": 65.0121, "lon": 25.4651, "iso": "FIN"},
            "reykjavik": {"lat": 64.1466, "lon": -21.9426, "iso": "ISL"},
            "keflavik": {"lat": 63.9961, "lon": -22.5610, "iso": "ISL"},
            "prague": {"lat": 50.0755, "lon": 14.4378, "iso": "CZE"},
            "budapest": {"lat": 47.4979, "lon": 19.0402, "iso": "HUN"},
            "warsaw": {"lat": 52.2297, "lon": 21.0122, "iso": "POL"},
            "krakow": {"lat": 50.0647, "lon": 19.9450, "iso": "POL"},
            "bucharest": {"lat": 44.4268, "lon": 26.1025, "iso": "ROU"},
            "sofia": {"lat": 42.6977, "lon": 23.3219, "iso": "BGR"},
            "zagreb": {"lat": 45.8150, "lon": 15.9819, "iso": "HRV"},
            "dubrovnik": {"lat": 42.6507, "lon": 18.0944, "iso": "HRV"},
            "moscow": {"lat": 55.7558, "lon": 37.6173, "iso": "RUS"},
            "st petersburg": {"lat": 59.9343, "lon": 30.3351, "iso": "RUS"},
            "kiev": {"lat": 50.4501, "lon": 30.5234, "iso": "UKR"},
            "tirana": {"lat": 41.3275, "lon": 19.8187, "iso": "ALB"},
            "malta": {"lat": 35.9375, "lon": 14.3754, "iso": "MLT"},
            "larnaca": {"lat": 34.9167, "lon": 33.6292, "iso": "CYP"},
            "paphos": {"lat": 34.7754, "lon": 32.4245, "iso": "CYP"},
            "gibraltar": {"lat": 36.1408, "lon": -5.3536, "iso": "GIB"},
            
            # --- Americas ---
            "new york": {"lat": 40.7128, "lon": -74.0060, "iso": "USA"},
            "newark": {"lat": 40.6895, "lon": -74.1745, "iso": "USA"},
            "washington": {"lat": 38.9072, "lon": -77.0369, "iso": "USA"},
            "washington dc": {"lat": 38.9072, "lon": -77.0369, "iso": "USA"},
            "washington dulles": {"lat": 38.9555, "lon": -77.4565, "iso": "USA"},
            "baltimore": {"lat": 39.2904, "lon": -76.6122, "iso": "USA"},
            "boston": {"lat": 42.3601, "lon": -71.0589, "iso": "USA"},
            "philadelphia": {"lat": 39.9526, "lon": -75.1652, "iso": "USA"},
            "chicago": {"lat": 41.8781, "lon": -87.6298, "iso": "USA"},
            "atlanta": {"lat": 33.7490, "lon": -84.3880, "iso": "USA"},
            "miami": {"lat": 25.7617, "lon": -80.1918, "iso": "USA"},
            "orlando": {"lat": 28.5383, "lon": -81.3792, "iso": "USA"},
            "tampa": {"lat": 27.9506, "lon": -82.4572, "iso": "USA"},
            "fort lauderdale": {"lat": 26.1224, "lon": -80.1373, "iso": "USA"},
            "lauderdale": {"lat": 26.1224, "lon": -80.1373, "iso": "USA"},
            "houston": {"lat": 29.7604, "lon": -95.3698, "iso": "USA"},
            "dallas": {"lat": 32.7767, "lon": -96.7970, "iso": "USA"},
            "austin": {"lat": 30.2672, "lon": -97.7431, "iso": "USA"},
            "new orleans": {"lat": 29.9511, "lon": -90.0715, "iso": "USA"},
            "denver": {"lat": 39.7392, "lon": -104.9903, "iso": "USA"},
            "phoenix": {"lat": 33.4484, "lon": -112.0740, "iso": "USA"},
            "las vegas": {"lat": 36.1699, "lon": -115.1398, "iso": "USA"},
            "los angeles": {"lat": 34.0522, "lon": -118.2437, "iso": "USA"},
            "san diego": {"lat": 32.7157, "lon": -117.1611, "iso": "USA"},
            "san francisco": {"lat": 37.7749, "lon": -122.4194, "iso": "USA"},
            "san jose": {"lat": 37.3382, "lon": -121.8863, "iso": "USA"}, # Assuming CA usually
            "oakland": {"lat": 37.8044, "lon": -122.2711, "iso": "USA"},
            "seattle": {"lat": 47.6062, "lon": -122.3321, "iso": "USA"},
            "anchorage": {"lat": 61.2181, "lon": -149.9003, "iso": "USA"},
            "toronto": {"lat": 43.6510, "lon": -79.3470, "iso": "CAN"},
            "vancouver": {"lat": 49.2827, "lon": -123.1207, "iso": "CAN"},
            "montreal": {"lat": 45.5017, "lon": -73.5673, "iso": "CAN"},
            "calgary": {"lat": 51.0447, "lon": -114.0719, "iso": "CAN"},
            "edmonton": {"lat": 53.5461, "lon": -113.4938, "iso": "CAN"},
            "ottawa": {"lat": 45.4215, "lon": -75.6972, "iso": "CAN"},
            "mexico": {"lat": 19.4326, "lon": -99.1332, "iso": "MEX"},
            "mexico city": {"lat": 19.4326, "lon": -99.1332, "iso": "MEX"},
            "cancun": {"lat": 21.1619, "lon": -86.8515, "iso": "MEX"},
            "nassau": {"lat": 25.0443, "lon": -77.3504, "iso": "BHS"},
            "bermuda": {"lat": 32.3078, "lon": -64.7505, "iso": "BMU"},
            "barbados": {"lat": 13.1939, "lon": -59.5432, "iso": "BRB"},
            "bridgetown": {"lat": 13.1060, "lon": -59.6316, "iso": "BRB"},
            "antigua": {"lat": 17.0608, "lon": -61.7964, "iso": "ATG"},
            "st lucia": {"lat": 13.9094, "lon": -60.9789, "iso": "LCA"},
            "saint lucia": {"lat": 13.9094, "lon": -60.9789, "iso": "LCA"},
            "grenada": {"lat": 12.1165, "lon": -61.6790, "iso": "GRD"},
            "punta cana": {"lat": 18.5601, "lon": -68.3725, "iso": "DOM"},
            "kingston": {"lat": 17.9714, "lon": -76.7936, "iso": "JAM"},
            "grand cayman": {"lat": 19.3222, "lon": -81.2409, "iso": "CYM"},
            "port of spain": {"lat": 10.6549, "lon": -61.5019, "iso": "TTO"},
            "sao paulo": {"lat": -23.5505, "lon": -46.6333, "iso": "BRA"},
            "rio de janeiro": {"lat": -22.9068, "lon": -43.1729, "iso": "BRA"},
            "buenos aires": {"lat": -34.6037, "lon": -58.3816, "iso": "ARG"},
            "santiago": {"lat": -33.4489, "lon": -70.6693, "iso": "CHL"},
            "lima": {"lat": -12.0464, "lon": -77.0428, "iso": "PER"},
            
            # --- Middle East / Africa / Asia ---
            "dubai": {"lat": 25.2048, "lon": 55.2708, "iso": "ARE"},
            "abu dhabi": {"lat": 24.4539, "lon": 54.3773, "iso": "ARE"},
            "doha": {"lat": 25.2854, "lon": 51.5310, "iso": "QAT"},
            "bahrain": {"lat": 26.0667, "lon": 50.5577, "iso": "BHR"},
            "kuwait": {"lat": 29.3759, "lon": 47.9774, "iso": "KWT"},
            "riyadh": {"lat": 24.7136, "lon": 46.6753, "iso": "SAU"},
            "jeddah": {"lat": 21.5433, "lon": 39.1728, "iso": "SAU"},
            "amman": {"lat": 31.9454, "lon": 35.9284, "iso": "JOR"},
            "amman jordan": {"lat": 31.9454, "lon": 35.9284, "iso": "JOR"},
            "beirut": {"lat": 33.8938, "lon": 35.5018, "iso": "LBN"},
            "lebanon": {"lat": 33.8938, "lon": 35.5018, "iso": "LBN"}, # Country Map
            "tel aviv": {"lat": 32.0853, "lon": 34.7818, "iso": "ISR"},
            "tehran": {"lat": 35.6892, "lon": 51.3890, "iso": "IRN"},
            "islamabad": {"lat": 33.6844, "lon": 73.0479, "iso": "PAK"},
            "cairo": {"lat": 30.0444, "lon": 31.2357, "iso": "EGY"},
            "marrakech": {"lat": 31.6295, "lon": -7.9811, "iso": "MAR"},
            "tunis": {"lat": 36.8065, "lon": 10.1815, "iso": "TUN"},
            "johannesburg": {"lat": -26.2041, "lon": 28.0473, "iso": "ZAF"},
            "cape town": {"lat": -33.9249, "lon": 18.4241, "iso": "ZAF"},
            "durban": {"lat": -29.8587, "lon": 31.0218, "iso": "ZAF"},
            "nairobi": {"lat": -1.2921, "lon": 36.8219, "iso": "KEN"},
            "accra": {"lat": 5.6037, "lon": -0.1870, "iso": "GHA"},
            "abuja": {"lat": 9.0765, "lon": 7.3986, "iso": "NGA"},
            "mauritius": {"lat": -20.3484, "lon": 57.5522, "iso": "MUS"},
            "seychelles": {"lat": -4.6796, "lon": 55.4920, "iso": "SYC"},
            "mahe": {"lat": -4.6796, "lon": 55.4920, "iso": "SYC"},
            "male": {"lat": 4.1755, "lon": 73.5093, "iso": "MDV"},
            "mumbai": {"lat": 19.0760, "lon": 72.8777, "iso": "IND"},
            "delhi": {"lat": 28.6139, "lon": 77.2090, "iso": "IND"},
            "bangalore": {"lat": 12.9716, "lon": 77.5946, "iso": "IND"},
            "chennai": {"lat": 13.0827, "lon": 80.2707, "iso": "IND"},
            "hyderabad": {"lat": 17.3850, "lon": 78.4867, "iso": "IND"},
            "singapore": {"lat": 1.3521, "lon": 103.8198, "iso": "SGP"},
            "bangkok": {"lat": 13.7563, "lon": 100.5018, "iso": "THA"},
            "kuala lumpur": {"lat": 3.1390, "lon": 101.6869, "iso": "MYS"},
            "hong kong": {"lat": 22.3193, "lon": 114.1694, "iso": "HKG"},
            "beijing": {"lat": 39.9042, "lon": 116.4074, "iso": "CHN"},
            "shanghai": {"lat": 31.2304, "lon": 121.4737, "iso": "CHN"},
            "tokyo": {"lat": 35.6762, "lon": 139.6503, "iso": "JPN"},
            "seoul": {"lat": 37.5665, "lon": 126.9780, "iso": "KOR"},
            "sydney": {"lat": -33.8688, "lon": 151.2093, "iso": "AUS"},
            "melbourne": {"lat": -37.8136, "lon": 144.9631, "iso": "AUS"}
        }

    def analyze_location_data(self, df: pd.DataFrame, location_col: str, use_api=False, progress_callback=None):
        geo_df = df.copy()
        if geo_df[location_col].dropna().empty: return geo_df

        # --- STEP 1: CLEANING ---
        def clean_location(val):
            val = str(val).strip()
            # 0. Null check
            if val.lower() == 'nan' or val == '': return None
            
            # 1. Route Parser ("London to Paris")
            match = re.search(r"^(.*?)(?:\s+to\s+|-|\s+via\s+)", val, re.IGNORECASE)
            if match: val = match.group(1).strip()
            
            # 2. IATA Code Check (Upper case 3 letters)
            if len(val) == 3 and val.upper() in self.iata_db:
                return self.iata_db[val.upper()].lower()
            
            # 3. Regex Noise Removal
            val = re.sub(r'(?i)\b(airport|intl|international|hub|city|ft|fort)\b', '', val)
            val = re.sub(r'\(.*?\)', '', val)
            val = re.sub(r',.*', '', val) # Remove ", Florida"
            
            # 4. Unicode Normalization
            val = val.replace('İ', 'I').replace('ü', 'u').replace('ã', 'a').replace('é', 'e')
            
            return val.strip().lower()

        geo_df['__clean_loc__'] = geo_df[location_col].apply(clean_location)
        unique_cities = geo_df['__clean_loc__'].dropna().unique()
        total_unique = len(unique_cities)
        
        mapping_dict = {}
        
        for idx, city in enumerate(unique_cities):
            if not city: continue
            
            if progress_callback:
                progress_callback(idx / total_unique, f"Resolving: {city.title()}...")
            
            # --- TIER 1: EXACT MATCH ---
            if city in self.static_db:
                mapping_dict[city] = self.static_db[city]
                continue
                
            # --- TIER 2: SUBSTRING MATCH ---
            found_sub = False
            for db_key in self.static_db:
                if db_key in city: 
                    mapping_dict[city] = self.static_db[db_key]
                    found_sub = True
                    break
            if found_sub: continue

            # --- TIER 3: FUZZY MATCH ---
            close_matches = difflib.get_close_matches(city, self.static_db.keys(), n=1, cutoff=0.75)
            if close_matches:
                mapping_dict[city] = self.static_db[close_matches[0]]
                continue

            # --- TIER 4: API (CONDITIONAL) ---
            if use_api:
                coords = self._get_coords_from_api(city)
                if coords: mapping_dict[city] = coords
            else:
                mapping_dict[city] = None

        # Map back results
        def get_lat(city): return mapping_dict.get(city)['lat'] if mapping_dict.get(city) else None
        def get_lon(city): return mapping_dict.get(city)['lon'] if mapping_dict.get(city) else None
        
        geo_df['lat'] = geo_df['__clean_loc__'].apply(get_lat)
        geo_df['lon'] = geo_df['__clean_loc__'].apply(get_lon)
        geo_df['Mapped_Location'] = geo_df['__clean_loc__'].str.title()
        
        return geo_df.drop(columns=['__clean_loc__'])

    def _get_coords_from_api(self, city_name):
        try:
            time.sleep(1.1) 
            location = self.geolocator.geocode(city_name, timeout=5)
            if location:
                data = {"lat": location.latitude, "lon": location.longitude, "iso": "UNK"}
                self.static_db[city_name] = data 
                return data
        except:
            return None
        return None