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
        # 💡 تنظيف قوي للأسماء: حذف AFC وحذف FC لتسهيل المطابقة
        n1 = name1.lower().replace(" fc", "").replace("afc ", "").strip()
        n2 = name2.lower().replace(" fc", "").replace("afc ", "").strip()
        
        # القاموس الشامل للحالات الاستثنائية
        aliases = {
            "man utd": "manchester united", 
            "man united": "manchester united",
            "man city": "manchester city",
            "wolves": "wolverhampton wanderers", 
            "wolverhampton": "wolverhampton wanderers",
            "nott'm forest": "nottingham forest",
            "nottingham": "nottingham forest",
            "spurs": "tottenham hotspur", 
            "tottenham": "tottenham hotspur",
            "sheff utd": "sheffield united", 
            "sheff wed": "sheffield wednesday",
            "luton": "luton town",
            "brighton": "brighton and hove albion",
            "leicester": "leicester city",
            "leeds": "leeds united",
            "ipswich": "ipswich town",
            "west ham": "west ham united",
            "newcastle": "newcastle united",
            "aston villa": "aston villa",
            "qpr": "queens park rangers"
        }
        
        n1 = aliases.get(n1, n1)
        n2 = aliases.get(n2, n2)

        # 1. المطابقة النصية المباشرة
        if n1 in n2 or n2 in n1:
            return True
            
        # 2. المطابقة بنسبة 70% للأسماء المعقدة
        similarity = SequenceMatcher(None, n1, n2).ratio()
        if similarity >= 0.70:
            return True
            
        return False

    def get_odds(self, home_team, away_team):
        if not self.api_key or self.api_key == "ضع_مفتاح_the_odds_هنا":
            return None, "مفتاح The Odds API غير متوفر."

        sports = ['soccer_epl', 'soccer_efl_championship']
        api_error_msg = ""
        
        for sport in sports:
            url = f'https://api.the-odds-api.com/v4/sports/{sport}/odds/?apiKey={self.api_key}&regions=eu,uk&markets=h2h'
            try:
                response = requests.get(url)
                
                # التقاط خطأ استنفاد الطلبات أو خطأ المفتاح
                if response.status_code != 200:
                    api_error_msg = f"خطأ من الخادم ({response.status_code}): قد تكون باقة الطلبات نفدت أو المفتاح غير صالح."
                    continue 
                    
                matches = response.json()
                for match in matches:
                    api_home = match['home_team']
                    api_away = match['away_team']
                    
                    if self._is_match(home_team, api_home) and self._is_match(away_team, api_away):
                        if not match.get('bookmakers'):
                            return None, "المباراة تم العثور عليها لكن لا توجد شركات مراهنة طرحت الكوتا لها بعد."
                            
                        target_bookie = None
                        preferred_bookies = ['bet365', '1xbet', 'pinnacle', 'williamhill', 'betfair_ex_eu', 'unibet_eu']
                        
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
                            elif out['name'].lower() in ['draw', 'x']: odds['draw'] = out['price']
                        
                        if odds['home'] != 0 and odds['away'] != 0:
                            return odds, target_bookie['title']
            except Exception as e:
                return None, f"عطل برمجي أثناء جلب الكوتا: {str(e)}"
                
        if api_error_msg:
            return None, api_error_msg
            
        return None, f"لم يتم العثور على مباراة ({home_team} ضد {away_team}) في سوق المراهنات."

    def get_available_matches(self):
        """تجلب قائمة بكل المباريات المتاحة حالياً في مكاتب المراهنات لعرضها في القائمة المنسدلة"""
        if not self.api_key or self.api_key == "ضع_مفتاح_the_odds_هنا":
            return []
            
        matches_list = []
        sports = ['soccer_epl', 'soccer_efl_championship']
        
        for sport in sports:
            url = f'https://api.the-odds-api.com/v4/sports/{sport}/odds/?apiKey={self.api_key}&regions=eu,uk&markets=h2h'
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    matches = response.json()
                    for match in matches:
                        matches_list.append(f"{match['home_team']} vs {match['away_team']}")
            except Exception:
                pass
                
        return matches_list
