import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self):
        # المواسم (مثال: 2425 تعني 2024-2025)
        self.seasons = ["1920", "2021", "2122", "2223", "2324", "2425"]
        # E0 = الدوري الإنجليزي الممتاز | E1 = الشامبيونشيب
        self.leagues = ["E0", "E1"]
        self.base_url = "https://www.football-data.co.uk/mmz4281/{}/{}.csv"

    def fetch_data(self):
        dfs = []
        for season in self.seasons:
            for league in self.leagues:
                url = self.base_url.format(season, league)
                try:
                    # قراءة ملف CSV مباشرة من الإنترنت بضغطة زر
                    df = pd.read_csv(url, on_bad_lines='skip')
                    # أخذ الأعمدة المهمة فقط (الفرق والأهداف) وحذف الأسطر الفارغة
                    df = df[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].dropna()
                    # إعادة تسمية الأعمدة لتتوافق مع محرك XGBoost الخاص بنا
                    df.rename(columns={'HomeTeam': 'team1', 'AwayTeam': 'team2', 'FTHG': 'goals1', 'FTAG': 'goals2'}, inplace=True)
                    dfs.append(df)
                except Exception:
                    pass # تجاهل الروابط في حال عدم توفر الموسم بعد
        
        if not dfs:
            raise ValueError("لم يتم جلب أي بيانات! يرجى التحقق من الاتصال بالإنترنت.")
            
        raw_df = pd.concat(dfs, ignore_index=True)
        # تنظيف الأسماء من أي مسافات فارغة لضمان عدم تكرار الفرق
        raw_df['team1'] = raw_df['team1'].astype(str).str.strip()
        raw_df['team2'] = raw_df['team2'].astype(str).str.strip()
        return raw_df

    def extract_features(self, df):
        team_stats = {}
        h2h_stats = {}
        features = []
        
        for idx, row in df.iterrows():
            t1, t2 = row['team1'], row['team2']
            g1, g2 = int(row['goals1']), int(row['goals2'])
            
            # النتيجة: 2 = فوز الأرض، 1 = تعادل، 0 = فوز الضيف
            result = 2 if g1 > g2 else (1 if g1 == g2 else 0)
            
            # المواجهات المباشرة H2H
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
            
            # المتوسط المتحرك (EMA)
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

