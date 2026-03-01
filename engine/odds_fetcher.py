import requests
import os
import streamlit as st
from difflib import SequenceMatcher

class OddsFetcher:
    def __init__(self):
        try:
            self.api_key = st.secrets["ODDS_API_KEY"]
        except Exception:
            self.api_key = os.getenv("ODDS_API_KEY")

    def _is_match(self, name1, name2):
        n1 = name1.lower().replace(" fc", "").strip()
        n2 = name2.lower().replace(" fc", "").strip()
        
        # القاموس الشامل لربط أسماء قاعدة البيانات بالأسماء الرسمية في The Odds API
        aliases = {
            "man utd": "manchester united", 
            "man city": "manchester city",
            "wolves": "wolverhampton wanderers", 
            "nott'm forest": "nottingham forest",
            "nottingham": "nottingham forest",
            "spurs": "tottenham hotspur", 
            "tottenham": "tottenham hotspur",
            "sheff utd": "sheffield united", 
            "luton": "luton town",
            "brighton": "brighton and hove albion",
            "leicester": "leicester city",
            "leeds": "leeds united",
            "ipswich": "ipswich town",
            "west ham": "west ham united",
            "newcastle": "newcastle united",
            "aston villa": "aston villa"
        }
        
        n1 = aliases.get(n1, n1)
        n2 = aliases.get(n2, n2)
        
        # 1. فحص التطابق بنسبة 75%
        similarity = SequenceMatcher(None, n1, n2).ratio()
        if similarity >= 0.75:
            return True
            
        # 2. فحص التطابق الجزئي الآمن (يمنع تطابق الكلمات القصيرة جداً مثل Man)
        if (n1 in n2 or n2 in n1) and len(n1) > 5 and len(n2) > 5:
            return True
            
        return False

    def get_odds(self, home_team, away_team):
        if not self.api_key or self.api_key == "ضع_مفتاح_the_odds_هنا":
            return None, "مفتاح API غير موجود"

        sports = ['soccer_epl', 'soccer_efl_championship']
        for sport in sports:
            url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds/?apiKey={self.api_key}&regions=eu,uk&markets=h2h"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    matches = response.json()
                    for match in matches:
                        if self._is_match(home_team, match['home_team']) and self._is_match(away_team, match['away_team']):
                            if match['bookmakers']:
                                target_bookie = None
                                preferred_bookies = ['onexbet', '1xbet', 'bet365', 'pinnacle', 'williamhill', 'unibet_eu', 'betfair_ex_eu']
                                
                                for pref in preferred_bookies:
                                    for bookie in match['bookmakers']:
                                        if bookie['key'].lower() == pref:
                                            target_bookie = bookie
                                            break
                                    if target_bookie:
                                        break 
                                        
                                if not target_bookie:
                                    target_bookie = match['bookmakers'][0]

                                outcomes = target_bookie['markets'][0]['outcomes']
                                odds = {'home': 0, 'draw': 0, 'away': 0}
                                for out in outcomes:
                                    if out['name'] == match['home_team']: odds['home'] = out['price']
                                    elif out['name'] == match['away_team']: odds['away'] = out['price']
                                    elif out['name'].lower() == 'draw': odds['draw'] = out['price']
                                
                                return odds, target_bookie['title']
            except Exception as e:
                return None, f"خطأ في الاتصال: {e}"
                
        return None, "المباراة لم تُطرح بعد في مكاتب المراهنات أو انتهت"
