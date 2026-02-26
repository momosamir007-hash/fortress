import pandas as pd
import numpy as np
import re
import requests

class DataProcessor:
    def __init__(self):
        self.seasons = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
        # سحب بيانات البريميرليج والشامبيونشيب معاً لرفع دقة التدريب
        self.url_templates = [
            "https://raw.githubusercontent.com/openfootball/england/master/{}/1-premierleague.txt",
            "https://raw.githubusercontent.com/openfootball/england/master/{}/2-championship.txt"
        ]
        
    def clean_team_name(self, name):
        if not name or pd.isna(name): return "Unknown"
        s = str(name).strip()
        s = re.sub(r'\(.*?\)', '', s)
        s = re.sub(r'[^a-zA-Z\s]', '', s)
        return s.strip()

    def fetch_github_data(self):
        matches = []
        for season in self.seasons:
            for template in self.url_templates:
                url = template.format(season)
                try:
                    response = requests.get(url, timeout=15)
                    if response.status_code == 200:
                        lines = response.text.split('\n')
                        for line in lines:
                            line = line.strip()
                            if not line or line.startswith('#') or line.startswith('[') or line.startswith('Matchday') or line.startswith('Round'):
                                continue
                            
                            match_score = re.search(r'\b(\d+)\s*-\s*(\d+)\b', line)
                            if match_score:
                                team1_raw = line[:match_score.start()].split('(')[0].strip()
                                team2_raw = line[match_score.end():].split('(')[0].strip()
                                
                                matches.append({
                                    'team1': self.clean_team_name(team1_raw),
                                    'team2': self.clean_team_name(team2_raw),
                                    'goals1': int(match_score.group(1)),
                                    'goals2': int(match_score.group(2))
                                })
                except Exception as e:
                    pass # تجاهل الأخطاء الصامتة لتسريع الجلب
                
        if not matches:
            raise ValueError("لم يتم جلب أي بيانات! يرجى التحقق من اتصال الإنترنت.")
        return pd.DataFrame(matches)

    def extract_features(self, df):
        team_stats = {}
        h2h_stats = {}
        features = []
        
        for idx, row in df.iterrows():
            t1, t2 = row['team1'], row['team2']
            g1, g2 = int(row['goals1']), int(row['goals2'])
            
            result = 2 if g1 > g2 else (1 if g1 == g2 else 0)
            
            pair = tuple(sorted([t1, t2]))
            if pair not in h2h_stats:
                h2h_stats[pair] = {'t1_wins': 0, 't2_wins': 0, 'draws': 0}
                
            h2h_t1_adv = h2h_stats[pair]['t1_wins'] - h2h_stats[pair]['t2_wins'] if pair[0] == t1 else h2h_stats[pair]['t2_wins'] - h2h_stats[pair]['t1_wins']

            features.append({
                'team1': t1, 'team2': t2,
                'h_atk': team_stats.get(t1, {'atk': 1.0})['atk'],
                'h_def': team_stats.get(t1, {'def': 1.0})['def'],
                'h_pts': team_stats.get(t1, {'pts': 1.0})['pts'],
                'a_atk': team_stats.get(t2, {'atk': 1.0})['atk'],
                'a_def': team_stats.get(t2, {'def': 1.0})['def'],
                'a_pts': team_stats.get(t2, {'pts': 1.0})['pts'],
                'h2h_adv': h2h_t1_adv,
                'result': result
            })
            
            alpha = 0.3 
            for t in [t1, t2]:
                if t not in team_stats: team_stats[t] = {'atk': 1.0, 'def': 1.0, 'pts': 1.0}
                
            team_stats[t1]['atk'] = (alpha * g1) + ((1 - alpha) * team_stats[t1]['atk'])
            team_stats[t1]['def'] = (alpha * g2) + ((1 - alpha) * team_stats[t1]['def'])
            team_stats[t1]['pts'] = (alpha * (3 if result==2 else 1 if result==1 else 0)) + ((1 - alpha) * team_stats[t1]['pts'])
            
            team_stats[t2]['atk'] = (alpha * g2) + ((1 - alpha) * team_stats[t2]['atk'])
            team_stats[t2]['def'] = (alpha * g1) + ((1 - alpha) * team_stats[t2]['def'])
            team_stats[t2]['pts'] = (alpha * (3 if result==0 else 1 if result==1 else 0)) + ((1 - alpha) * team_stats[t2]['pts'])
            
            if result == 2: h2h_stats[pair]['t1_wins' if pair[0] == t1 else 't2_wins'] += 1
            elif result == 0: h2h_stats[pair]['t2_wins' if pair[0] == t1 else 't1_wins'] += 1
            else: h2h_stats[pair]['draws'] += 1

        self.latest_team_stats = team_stats
        self.latest_h2h_stats = h2h_stats
        return pd.DataFrame(features)

    def get_match_features(self, t1, t2):
        h_stats = self.latest_team_stats.get(t1, {'atk': 1.0, 'def': 1.0, 'pts': 1.0})
        a_stats = self.latest_team_stats.get(t2, {'atk': 1.0, 'def': 1.0, 'pts': 1.0})
        
        pair = tuple(sorted([t1, t2]))
        h2h = self.latest_h2h_stats.get(pair, {'t1_wins': 0, 't2_wins': 0, 'draws': 0})
        h2h_t1_adv = h2h['t1_wins'] - h2h['t2_wins'] if pair[0] == t1 else h2h['t2_wins'] - h2h['t1_wins']
        
        return np.array([[h_stats['atk'], h_stats['def'], h_stats['pts'], 
                          a_stats['atk'], a_stats['def'], a_stats['pts'], h2h_t1_adv]])
