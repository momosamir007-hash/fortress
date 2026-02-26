import requests
import os
import streamlit as st
from datetime import datetime, timedelta

class FixturesFetcher:
    def __init__(self):
        try:
            self.api_key = st.secrets["FOOTBALL_DATA_API_KEY"]
        except Exception:
            self.api_key = os.getenv("FOOTBALL_DATA_API_KEY")

    def get_upcoming_matches(self):
        if not self.api_key or self.api_key == "مفتاح_football_data_هنا":
            return []
            
        headers = {'X-Auth-Token': self.api_key}
        competitions = ['2021', '2016'] # البريميرليج والشامبيونشيب
        matches = []
        
        # جلب مباريات الأسبوع القادم
        date_from = datetime.now().strftime('%Y-%m-%d')
        date_to = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
        
        for comp in competitions:
            url = f"https://api.football-data.org/v4/competitions/{comp}/matches?status=SCHEDULED&dateFrom={date_from}&dateTo={date_to}"
            try:
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    for match in data.get('matches', []):
                        home = match['homeTeam']['name'].replace(" FC", "")
                        away = match['awayTeam']['name'].replace(" FC", "")
                        match_time = match['utcDate'][:16].replace("T", " ")
                        matches.append({"home": home, "away": away, "time": match_time})
            except Exception:
                pass
        return matches
