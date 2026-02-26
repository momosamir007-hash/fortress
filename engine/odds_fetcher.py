import requests
import os
import streamlit as st

class OddsFetcher:
    def __init__(self):
        # محاولة جلب المفتاح بأمان
        try:
            self.api_key = st.secrets["ODDS_API_KEY"]
        except Exception:
            self.api_key = os.getenv("ODDS_API_KEY")

    def _is_match(self, name1, name2):
        # توحيد الأسماء لتجنب مشكلة اختلاف التسميات بين المنصات
        n1, n2 = name1.lower().replace(" fc", ""), name2.lower().replace(" fc", "")
        aliases = {
            "man utd": "manchester united", "man city": "manchester city",
            "wolves": "wolverhampton", "nott'm forest": "nottingham",
            "spurs": "tottenham", "sheff utd": "sheffield", "luton": "luton town"
        }
        n1 = aliases.get(n1, n1)
        n2 = aliases.get(n2, n2)
        return (n1 in n2) or (n2 in n1)

    def get_odds(self, home_team, away_team):
        if not self.api_key or self.api_key == "ضع_مفتاح_the_odds_هنا":
            return None, "مفتاح API غير موجود"

        # البحث في البريميرليج والشامبيونشيب
        sports = ['soccer_epl', 'soccer_efl_championship']
        for sport in sports:
            url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds/?apiKey={self.api_key}&regions=eu,uk&markets=h2h"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    matches = response.json()
                    for match in matches:
                        # إذا وجدنا المباراة المطلوبة
                        if self._is_match(home_team, match['home_team']) and self._is_match(away_team, match['away_team']):
                            if match['bookmakers']:
                                # نختار أول شركة متاحة (غالباً Pinnacle أو Bet365)
                                bookie = match['bookmakers'][0] 
                                outcomes = bookie['markets'][0]['outcomes']
                                odds = {'home': 0, 'draw': 0, 'away': 0}
                                for out in outcomes:
                                    if out['name'] == match['home_team']: odds['home'] = out['price']
                                    elif out['name'] == match['away_team']: odds['away'] = out['price']
                                    elif out['name'].lower() == 'draw': odds['draw'] = out['price']
                                return odds, bookie['title']
            except Exception as e:
                return None, f"خطأ في الاتصال: {e}"
                
        return None, "المباراة لم تُطرح بعد في مكاتب المراهنات أو انتهت"
