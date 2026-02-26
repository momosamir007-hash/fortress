import streamlit as st
import numpy as np
from engine.data_processor import DataProcessor
from engine.ml_model import FortressML
from engine.llm_expert import OracleLLM

st.set_page_config(page_title="المحرك الحصين V13", page_icon="🏰", layout="wide")

@st.cache_resource
def load_and_train_engine():
    dp = DataProcessor()
    raw_df = dp.fetch_data()
    features_df = dp.extract_features(raw_df)
    
    ml = FortressML()
    ml.train(features_df)
    
    teams = sorted(list(set(raw_df['team1'].unique()) | set(raw_df['team2'].unique())))
    return dp, ml, teams

st.title("🏰 المحرك الحصين V13.0")
st.markdown("نظام التوقع الهجين (XGBoost + Groq LLM + الحوسبة السحابية للبيانات)")
st.markdown("---")

try:
    with st.spinner("جاري جلب البيانات من GitHub وتدريب نموذج الذكاء الاصطناعي..."):
        dp, ml, teams = load_and_train_engine()
except Exception as e:
    st.error(f"خطأ في تحميل البيانات: {e}")
    st.stop()

st.sidebar.header("⚙️ إعدادات الذكاء الاصطناعي")
confidence_threshold = st.sidebar.slider("عتبة التدخل (الفرق بين الاحتمالات %)", 5, 30, 15)

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("فريق الأرض (Home)", teams, index=teams.index("Arsenal") if "Arsenal" in teams else 0)
with col2:
    away_team = st.selectbox("الفريق الضيف (Away)", teams, index=teams.index("Chelsea") if "Chelsea" in teams else 1)

if home_team == away_team:
    st.warning("الرجاء اختيار فريقين مختلفين!")
else:
    if st.button("🚀 تحليل المواجهة والتوقع", use_container_width=True):
        match_x = dp.get_match_features(home_team, away_team)
        probs = ml.predict_match_probs(match_x)
        sorted_probs = np.sort(probs)[::-1]
        diff = (sorted_probs[0] - sorted_probs[1]) * 100
        
        st.subheader("📊 قراءة المحرك الرياضي (XGBoost)")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric(f"فوز {home_team} (أرض)", f"{probs[2]*100:.1f}%")
        m_col2.metric("تعادل", f"{probs[1]*100:.1f}%")
        m_col3.metric(f"فوز {away_team} (ضيف)", f"{probs[0]*100:.1f}%")
        
        st.markdown("---")
        
        if diff < confidence_threshold:
            st.info(f"⚠️ المباراة معقدة (فارق الاحتمالات {diff:.1f}%). جاري استدعاء Groq كحكم مساعد...")
            try:
                oracle = OracleLLM(provider="groq")
                final_decision = oracle.get_double_chance(home_team, away_team, probs)
                source = "🤖 تم الحسم بواسطة Groq LLM"
            except Exception as e:
                st.error(f"فشل الاتصال بـ API: {e}")
                final_decision = "تعذر التوقع الآمن"
                source = "خطأ"
        else:
            st.success(f"✅ المباراة محسومة إحصائياً (فارق {diff:.1f}%).")
            sorted_indices = np.argsort(probs)[::-1]
            top2 = {sorted_indices[0], sorted_indices[1]}
            if top2 == {2, 1}: final_decision = "الأرض أو تعادل"
            elif top2 == {0, 1}: final_decision = "الضيف أو تعادل"
            else: final_decision = "أرض أو ضيف"
            source = "🌲 تم الحسم بواسطة XGBoost"

        st.markdown(f"### 🎯 التوقع النهائي (فرصة مزدوجة): **{final_decision}**")
        st.caption(source)
