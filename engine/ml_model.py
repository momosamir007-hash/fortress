import xgboost as xgb

class FortressML:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            device='cpu',
            random_state=42
        )
        self.is_trained = False
        
    def train(self, df_features):
        X = df_features[['h_atk', 'h_def', 'h_pts', 'a_atk', 'a_def', 'a_pts', 'h2h_adv']]
        y = df_features['result']
        self.model.fit(X, y)
        self.is_trained = True
        
    def predict_match_probs(self, match_features):
        if not self.is_trained:
            raise Exception("النموذج غير مدرب بعد!")
        return self.model.predict_proba(match_features)[0]
