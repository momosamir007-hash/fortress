import xgboost as xgb
import pandas as pd

class FortressML:
    def __init__(self):
        # المحرك الأساسي (تصنيف الفرصة المزدوجة)
        self.model = xgb.XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05, 
            eval_metric='mlogloss', device='cpu'
        )
        # المحركين الجديدين (لتوقع الأهداف xG) - تم رفع العمق (max_depth) قليلاً للاستفادة من البيانات الجديدة
        self.model_reg_h = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, device='cpu')
        self.model_reg_a = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, device='cpu')

    def train(self, df):
        # تم إضافة الميزات الأربعة الجديدة التي تخص آخر 5 مباريات
        X = df[[
            'h_atk', 'h_def', 'h_pts', 'h_avg_scored_5', 'h_avg_conceded_5', 
            'a_atk', 'a_def', 'a_pts', 'a_avg_scored_5', 'a_avg_conceded_5', 
            'h2h_adv'
        ]]
        y_cls = df['result']
        y_h_goals = df['h_goals']
        y_a_goals = df['a_goals']
        
        # تدريب العقول الثلاثة في نفس الوقت
        self.model.fit(X, y_cls)
        self.model_reg_h.fit(X, y_h_goals)
        self.model_reg_a.fit(X, y_a_goals)

    def predict_match_probs(self, X):
        return self.model.predict_proba(X)[0]
        
    def predict_xg(self, X):
        h_xg = max(0, self.model_reg_h.predict(X)[0])
        a_xg = max(0, self.model_reg_a.predict(X)[0])
        return h_xg, a_xg
