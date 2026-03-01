import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self):
        self.seasons = []
        for year in range(1993, 2026):
            start_yr = str(year)[-2:].zfill(2)
            end_yr = str(year + 1)[-2:].zfill(2)
            self.seasons.append(f"{start_yr}{end_yr}")
            
        self.leagues = ["E0", "E1"]
        self.base_url = "https://www.football-data.co.uk/mmz4281/{}/{}.csv"

    def fetch_data(self):
        dfs = []
        for season in self.seasons:
            for league in self.leagues:
                url = self.base_url.format(season, league)
                try:
                    df = pd.read_csv(url, on_bad_lines='skip')
                    df = df[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].dropna()
                    df.rename(columns={'HomeTeam': 'team1', 'AwayTeam': 'team2', 'FTHG': 'goals1', 'FTAG': 'goals2'}, inplace=True)
                    dfs.append(df)
                except Exception:
                    pass
        
        if not dfs:
            raise ValueError("لم يتم جلب أي بيانات! يرجى التحقق من الاتصال بالإنترنت.")
            
        raw_df = pd.concat(dfs, ignore_index=True)
        raw_df['team1'] = raw_df['team1'].astype(str).str.strip()
        raw_df['team2'] = raw_df['team2'].astype(str).str.strip()
        return raw_df

    def extract_features(self, df):
        team_stats = {}
        team_recent = {}
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

            for t in [t1, t2]:
                if t not in team_recent:
                    team_recent[t] = {'scored': [], 'conceded': []}

            h_scored_5 = np.mean(team_recent[t1]['scored']) if len(team_recent[t1]['scored']) > 0 else 1.0
            h_conceded_5 = np.mean(team_recent[t1]['conceded']) if len(team_recent[t1]['conceded']) > 0 else 1.0
            
            a_scored_5 = np.mean(team_recent[t2]['scored']) if len(team_recent[t2]['scored']) > 0 else 1.0
            a_conceded_5 = np.mean(team_recent[t2]['conceded']) if len(team_recent[t2]['conceded']) > 0 else 1.0

            features.append({
                'team1': t1, 'team2': t2,
                'h_atk': team_stats.get(t1, {'atk': 1.0})['atk'],
                'h_def': team_stats.get(t1, {'def': 1.0})['def'],
                'h_pts': team_stats.get(t1, {'pts': 1.0})['pts'],
                'h_avg_scored_5': h_scored_5,
                'h_avg_conceded_5': h_conceded_5,
                'a_atk': team_stats.get(t2, {'atk': 1.0})['atk'],
                'a_def': team_stats.get(t2, {'def': 1.0})['def'],
                'a_pts': team_stats.get(t2, {'pts': 1.0})['pts'],
                'a_avg_scored_5': a_scored_5,
                'a_avg_conceded_5': a_conceded_5,
                'h2h_adv': h2h_t1_adv,
                'result': result,
                'h_goals': g1,  
                'a_goals': g2   
            })
            
            alpha = 0.3 
            for t in [t1, t2]:
                if t not in team_stats:
                    if len(team_stats) == 0:
                        team_stats[t] = {'atk': 1.0, 'def': 1.0, 'pts': 1.0}
                    else:
                        avg_atk = sum(s['atk'] for s in team_stats.values()) / len(team_stats)
                        avg_def = sum(s['def'] for s in team_stats.values()) / len(team_stats)
                        avg_pts = sum(s['pts'] for s in team_stats.values()) / len(team_stats)
                        team_stats[t] = {'atk': avg_atk * 0.8, 'def': avg_def * 0.8, 'pts': avg_pts * 0.8}
                
            team_stats[t1]['atk'] = (alpha * g1) + ((1 - alpha) * team_stats[t1]['atk'])
            team_stats[t1]['def'] = (alpha * g2) + ((1 - alpha) * team_stats[t1]['def'])
            team_stats[t1]['pts'] = (alpha * (3 if result==2 else 1 if result==1 else 0)) + ((1 - alpha) * team_stats[t1]['pts'])
            
            team_stats[t2]['atk'] = (alpha * g2) + ((1 - alpha) * team_stats[t2]['atk'])
            team_stats[t2]['def'] = (alpha * g1) + ((1 - alpha) * team_stats[t2]['def'])
            team_stats[t2]['pts'] = (alpha * (3 if result==0 else 1 if result==1 else 0)) + ((1 - alpha) * team_stats[t2]['pts'])
            
            team_recent[t1]['scored'].append(g1)
            team_recent[t1]['conceded'].append(g2)
            team_recent[t2]['scored'].append(g2)
            team_recent[t2]['conceded'].append(g1)
            
            team_recent[t1]['scored'] = team_recent[t1]['scored'][-5:]
            team_recent[t1]['conceded'] = team_recent[t1]['conceded'][-5:]
            team_recent[t2]['scored'] = team_recent[t2]['scored'][-5:]
            team_recent[t2]['conceded'] = team_recent[t2]['conceded'][-5:]

            if result == 2: h2h_stats[pair]['t1_wins' if pair[0] == t1 else 't2_wins'] += 1
            elif result == 0: h2h_stats[pair]['t2_wins' if pair[0] == t1 else 't1_wins'] += 1
            else: h2h_stats[pair]['draws'] += 1

        self.latest_team_stats = team_stats
        self.latest_team_recent = team_recent
        self.latest_h2h_stats = h2h_stats
        return pd.DataFrame(features)

    def get_match_features(self, t1, t2):
        h_stats = self.latest_team_stats.get(t1, {'atk': 1.0, 'def': 1.0, 'pts': 1.0})
        a_stats = self.latest_team_stats.get(t2, {'atk': 1.0, 'def': 1.0, 'pts': 1.0})
        
        h_recent = self.latest_team_recent.get(t1, {'scored': [1.0], 'conceded': [1.0]})
        a_recent = self.latest_team_recent.get(t2, {'scored': [1.0], 'conceded': [1.0]})
        
        h_scored_5 = np.mean(h_recent['scored']) if h_recent['scored'] else 1.0
        h_conceded_5 = np.mean(h_recent['conceded']) if h_recent['conceded'] else 1.0
        a_scored_5 = np.mean(a_recent['scored']) if a_recent['scored'] else 1.0
        a_conceded_5 = np.mean(a_recent['conceded']) if a_recent['conceded'] else 1.0

        pair = tuple(sorted([t1, t2]))
        h2h = self.latest_h2h_stats.get(pair, {'t1_wins': 0, 't2_wins': 0, 'draws': 0})
        h2h_t1_adv = h2h['t1_wins'] - h2h['t2_wins'] if pair[0] == t1 else h2h['t2_wins'] - h2h['t1_wins']
        
        return np.array([[
            h_stats['atk'], h_stats['def'], h_stats['pts'], h_scored_5, h_conceded_5,
            a_stats['atk'], a_stats['def'], a_stats['pts'], a_scored_5, a_conceded_5, 
            h2h_adv
        ]])
