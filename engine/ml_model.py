import xgboost as xgb
import pandas as pd

class FortressML:
    def __init__(self):
        # المحرك الأساسي (تصنيف الفرصة المزدوجة)
        self.model = xgb.XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05, 
            eval_metric='mlogloss', device='cpu'
        )
        # المحركين الجديدين (لتوقع الأهداف xG)
        self.model_reg_h = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, device='cpu')
        self.model_reg_a = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, device='cpu')

    def train(self, df):
        # الميزات التي تتعلم منها الآلة
        X = df[['h_atk', 'h_def', 'h_pts', 'a_atk', 'a_def', 'a_pts', 'h2h_adv']]
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
        # توقع الأهداف (نستخدم max(0) لضمان عدم وجود أهداف بالسالب)
        h_xg = max(0, self.model_reg_h.predict(X)[0])
        a_xg = max(0, self.model_reg_a.predict(X)[0])
        return h_xg, a_xg
