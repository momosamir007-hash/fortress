import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self):
        # توليد كل المواسم من 1993 حتى 2026 آلياً
        self.seasons = []
        for year in range(1993, 2026):
            start_yr = str(year)[-2:].zfill(2)
            end_yr = str(year + 1)[-2:].zfill(2)
            self.seasons.append(f"{start_yr}{end_yr}")
            
        # E0 = الدوري الإنجليزي الممتاز | E1 = الشامبيونشيب
        self.leagues = ["E0", "E1"]
        self.base_url = "https://www.football-data.co.uk/mmz4281/{}/{}.csv"

    def fetch_data(self):
        dfs = []
        for season in self.seasons:
            for league in self.leagues:
                url = self.base_url.format(season, league)
                try:
                    df = pd.read_csv(url, on_bad_lines='skip')
                    # 💡 أضفنا جلب عمود التاريخ (Date) للتحليل الزمني
                    if 'Date' in df.columns:
                        df = df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].dropna()
                        df.rename(columns={'HomeTeam': 'team1', 'AwayTeam': 'team2', 'FTHG': 'goals1', 'FTAG': 'goals2'}, inplace=True)
                        dfs.append(df)
                except Exception:
                    pass
        
        if not dfs:
            raise ValueError("لم يتم جلب أي بيانات! يرجى التحقق من الاتصال بالإنترنت.")
            
        raw_df = pd.concat(dfs, ignore_index=True)
        raw_df['team1'] = raw_df['team1'].astype(str).str.strip()
        raw_df['team2'] = raw_df['team2'].astype(str).str.strip()
        
        # 💡 التعديل الأهم لمنع تسريب البيانات: ترتيب المباريات زمنياً بشكل صارم
        raw_df['Date'] = pd.to_datetime(raw_df['Date'], dayfirst=True, errors='coerce')
        raw_df = raw_df.sort_values('Date').dropna(subset=['Date']).reset_index(drop=True)
        
        return raw_df

    def extract_features(self, df):
        team_stats = {}
        team_recent = {}
        h2h_stats = {}
        team_last_match_date = {} # 💡 تتبع تاريخ آخر مباراة لحساب الإرهاق
        team_matches_played = {}  # 💡 تتبع عدد المباريات (رقم الجولة) لحساب الدوافع
        features = []
        
        for idx, row in df.iterrows():
            t1, t2 = row['team1'], row['team2']
            g1, g2 = int(row['goals1']), int(row['goals2'])
            match_date = row['Date']
            
            result = 2 if g1 > g2 else (1 if g1 == g2 else 0)
            pair = tuple(sorted([t1, t2]))
            
            if pair not in h2h_stats:
                h2h_stats[pair] = {'t1_wins': 0, 't2_wins': 0, 'draws': 0}
                
            h2h_t1_adv = h2h_stats[pair]['t1_wins'] - h2h_stats[pair]['t2_wins'] if pair[0] == t1 else h2h_stats[pair]['t2_wins'] - h2h_stats[pair]['t1_wins']

            for t in [t1, t2]:
                if t not in team_recent:
                    team_recent[t] = {'scored': [], 'conceded': []}
                if t not in team_last_match_date:
                    team_last_match_date[t] = match_date - pd.Timedelta(days=7) # افتراضي أسبوع راحة
                if t not in team_matches_played:
                    team_matches_played[t] = 0

            # 💡 حساب عامل الإرهاق (أيام الراحة) - نضع حداً أقصى 14 يوماً لتجاهل التوقفات الصيفية
            h_rest_days = min((match_date - team_last_match_date[t1]).days, 14)
            a_rest_days = min((match_date - team_last_match_date[t2]).days, 14)
            
            # 💡 الجولة الحالية
            h_matchweek = min(team_matches_played[t1] + 1, 38)
            a_matchweek = min(team_matches_played[t2] + 1, 38)

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
                'h_rest_days': h_rest_days,     # 🚀 ميزة الإرهاق الجديدة
                'h_matchweek': h_matchweek,     # 🚀 ميزة الدوافع الجديدة
                'a_atk': team_stats.get(t2, {'atk': 1.0})['atk'],
                'a_def': team_stats.get(t2, {'def': 1.0})['def'],
                'a_pts': team_stats.get(t2, {'pts': 1.0})['pts'],
                'a_avg_scored_5': a_scored_5,
                'a_avg_conceded_5': a_conceded_5,
                'a_rest_days': a_rest_days,     # 🚀 ميزة الإرهاق الجديدة
                'a_matchweek': a_matchweek,     # 🚀 ميزة الدوافع الجديدة
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

            # تحديث التواريخ والمباريات الملعوبة
            team_last_match_date[t1] = match_date
            team_last_match_date[t2] = match_date
            team_matches_played[t1] = h_matchweek
            team_matches_played[t2] = a_matchweek

            if result == 2: h2h_stats[pair]['t1_wins' if pair[0] == t1 else 't2_wins'] += 1
            elif result == 0: h2h_stats[pair]['t2_wins' if pair[0] == t1 else 't1_wins'] += 1
            else: h2h_stats[pair]['draws'] += 1

        self.latest_team_stats = team_stats
        self.latest_team_recent = team_recent
        self.latest_h2h_stats = h2h_stats
        # للتبسيط في المباريات المباشرة القادمة، سنفترض 7 أيام راحة إذا لم تكن معلومة
        self.latest_team_last_match = team_last_match_date
        self.latest_team_matches_played = team_matches_played
        
        return pd.DataFrame(features)

    def get_match_features(self, t1, t2):
        """تجهيز المصفوفة الرقمية للآلة (XGBoost)"""
        h_stats = self.latest_team_stats.get(t1, {'atk': 1.0, 'def': 1.0, 'pts': 1.0})
        a_stats = self.latest_team_stats.get(t2, {'atk': 1.0, 'def': 1.0, 'pts': 1.0})
        
        h_recent = self.latest_team_recent.get(t1, {'scored': [1.0], 'conceded': [1.0]})
        a_recent = self.latest_team_recent.get(t2, {'scored': [1.0], 'conceded': [1.0]})
        
        h_scored_5 = np.mean(h_recent['scored']) if h_recent['scored'] else 1.0
        h_conceded_5 = np.mean(h_recent['conceded']) if h_recent['conceded'] else 1.0
        a_scored_5 = np.mean(a_recent['scored']) if a_recent['scored'] else 1.0
        a_conceded_5 = np.mean(a_recent['conceded']) if a_recent['conceded'] else 1.0

        # افتراض 7 أيام راحة للمباريات المباشرة القادمة إذا لم تكن متوفرة
        h_rest_days = 7
        a_rest_days = 7
        h_matchweek = min(self.latest_team_matches_played.get(t1, 0) + 1, 38)
        a_matchweek = min(self.latest_team_matches_played.get(t2, 0) + 1, 38)

        pair = tuple(sorted([t1, t2]))
        h2h = self.latest_h2h_stats.get(pair, {'t1_wins': 0, 't2_wins': 0, 'draws': 0})
        h2h_t1_adv = h2h['t1_wins'] - h2h['t2_wins'] if pair[0] == t1 else h2h['t2_wins'] - h2h['t1_wins']
        
        # ⚠️ تنبيه: المصفوفة الآن تتكون من 15 ميزة
        return np.array([[
            h_stats['atk'], h_stats['def'], h_stats['pts'], h_scored_5, h_conceded_5, h_rest_days, h_matchweek,
            a_stats['atk'], a_stats['def'], a_stats['pts'], a_scored_5, a_conceded_5, a_rest_days, a_matchweek,
            h2h_adv 
        ]])

    def get_detailed_h2h(self, home_team, away_team):
        pair = tuple(sorted([home_team, away_team]))
        if pair not in self.latest_h2h_stats:
            return {'home_wins': 0, 'away_wins': 0, 'draws': 0, 'total': 0}
        stats = self.latest_h2h_stats[pair]
        if pair[0] == home_team:
            h_wins, a_wins = stats['t1_wins'], stats['t2_wins']
        else:
            h_wins, a_wins = stats['t2_wins'], stats['t1_wins']
        draws = stats['draws']
        return {'home_wins': h_wins, 'away_wins': a_wins, 'draws': draws, 'total': h_wins + a_wins + draws}
