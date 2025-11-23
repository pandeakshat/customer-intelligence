import pandas as pd
import re

class GeoAnalyzer:
    """
    Extracts geospatial insights from location columns (Route, Country, City).
    """
    
    # Static DB for Route Parsing (Legacy Support)
    CITY_COORDS = {
        "london": {"lat": 51.5074, "lon": -0.1278, "iso": "GBR"},
        "heathrow": {"lat": 51.4700, "lon": -0.4543, "iso": "GBR"},
        "gatwick": {"lat": 51.1537, "lon": -0.1821, "iso": "GBR"},
        "stuttgart": {"lat": 48.7758, "lon": 9.1829, "iso": "DEU"},
        "new york": {"lat": 40.7128, "lon": -74.0060, "iso": "USA"},
        "jfk": {"lat": 40.6413, "lon": -73.7781, "iso": "USA"},
        "brussels": {"lat": 50.8503, "lon": 4.3517, "iso": "BEL"},
        "paris": {"lat": 48.8566, "lon": 2.3522, "iso": "FRA"},
        "frankfurt": {"lat": 50.1109, "lon": 8.6821, "iso": "DEU"},
        "doha": {"lat": 25.2854, "lon": 51.5310, "iso": "QAT"},
        "sydney": {"lat": -33.8688, "lon": 151.2093, "iso": "AUS"},
        "tokyo": {"lat": 35.6762, "lon": 139.6503, "iso": "JPN"},
        "los angeles": {"lat": 34.0522, "lon": -118.2437, "iso": "USA"},
        "chicago": {"lat": 41.8781, "lon": -87.6298, "iso": "USA"},
        "singapore": {"lat": 1.3521, "lon": 103.8198, "iso": "SGP"},
        "dubai": {"lat": 25.2048, "lon": 55.2708, "iso": "ARE"}
    }

    def analyze_location_data(self, df: pd.DataFrame, location_col: str):
        """
        Smart processor: determines if column is 'Route' or 'Region/Country'.
        Returns a dataframe ready for plotting.
        """
        geo_df = df.copy()
        
        # 1. Check if it looks like a Route (contains " to " or "-")
        sample_val = str(geo_df[location_col].iloc[0])
        is_route = bool(re.search(r"(\sto\s)|-", sample_val))
        
        if is_route:
            return self._process_routes(geo_df, location_col)
        else:
            return self._process_regions(geo_df, location_col)

    def _process_routes(self, df, col):
        """ Legacy Route Parser (Extract Origin -> Lat/Lon) """
        def get_coords(route_str):
            if not isinstance(route_str, str): return None, None, None
            match = re.search(r"^(.*?)(?:\s+to\s+|-|\s+via\s+)", route_str, re.IGNORECASE)
            if match:
                origin = match.group(1).strip().lower()
                if origin in self.CITY_COORDS:
                    data = self.CITY_COORDS[origin]
                    return data['lat'], data['lon'], origin.title()
                for city, data in self.CITY_COORDS.items():
                    if city in origin:
                        return data['lat'], data['lon'], city.title()
            return None, None, None

        coords = df[col].apply(get_coords)
        df['lat'] = [x[0] for x in coords]
        df['lon'] = [x[1] for x in coords]
        df['Mapped_Location'] = [x[2] for x in coords]
        return df.dropna(subset=['lat'])

    def _process_regions(self, df, col):
        """
        Simple Region/Country Handler.
        For V1, we assume standard country names or use the column as is for Choropleth.
        """
        df['Mapped_Location'] = df[col]
        df['Geo_Type'] = 'Region' 
        return df