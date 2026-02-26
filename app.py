import streamlit as st
import numpy as np
import time
from engine.data_processor import DataProcessor
from engine.ml_model import FortressML
from engine.llm_expert import OracleLLM

st.set_page_config(page_title="المحرك الحصين V13", page_icon="🏰", layout="wide")

# 1. جلب البيانات وتدريب النموذج الأساسي (يُحفظ في الذاكرة)
@st.cache_resource
def load_and_train_engine():
    dp = DataProcessor()
    raw_df = dp.fetch_data()
    features_df = dp.extract_features(raw_df)
    
    ml = FortressML()
    ml.train(features_df)
    
    teams = sorted(list(set(raw_df['team1'].unique()) | set(raw_df['team2'].unique())))
    match_count = len(raw_df)
    return dp, ml, teams, match_count, features_df # أضفنا features_df لنستخدمها في الفحص الرجعي

st.title("🏰 المحرك الحصين V13.0")
st.markdown("نظام التوقع الهجين (XGBoost + Groq LLM + الحوسبة السحابية)")
st.markdown("---")

try:
    with st.spinner("جاري طحن البيانات التاريخية من 1993 وتدريب الذكاء الاصطناعي..."):
        dp, ml, teams, match_count, features_df = load_and_train_engine()
        st.sidebar.success(f"📚 حجم قاعدة التدريب الكلية: {match_count} مباراة")
except Exception as e:
    st.error(f"خطأ في تحميل البيانات: {e}")
    st.stop()

st.sidebar.header("⚙️ إعدادات الذكاء الاصطناعي")
confidence_threshold = st.sidebar.slider("عتبة التدخل للـ LLM (الفرق %)", 5, 30, 15)

# 2. إنشاء التبويبات (Tabs) لفصل الوظائف
tab1, tab2 = st.tabs(["🔮 التوقع المباشر (مباراة بمباراة)", "📈 الفحص الرجعي (Backtest)"])

# ==========================================
# التبويب الأول: التوقع المباشر
# ==========================================
with tab1:
    st.subheader("توقع نتيجة مباراة قادمة")
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
            
            st.markdown("#### 📊 قراءة المحرك الرياضي (XGBoost)")
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

# ==========================================
# التبويب الثاني: الفحص الرجعي (Backtest)
# ==========================================
with tab2:
    st.subheader("اختبار دقة النموذج على المواسم الماضية")
    st.info("في هذا الاختبار، سنقوم باقتطاع المواسم الأخيرة وإخفائها عن النموذج. سيتدرب النموذج على الماضي السحيق فقط، ثم نختبره على المواسم المخفية لنرى مدى دقته الحقيقية في توقع مباريات لم يرها قط.")
    
    seasons_to_hide = st.slider("كم موسماً تريد إخفاءه واختبار النموذج عليه؟", min_value=1, max_value=10, value=5)
    matches_per_season = 932 # تقريباً (البريميرليج + الشامبيونشيب)
    
    if st.button("⚙️ بدء الفحص الرجعي الآن", type="primary", use_container_width=True):
        matches_to_hide = seasons_to_hide * matches_per_season
        
        if matches_to_hide >= len(features_df):
            st.error("❌ عدد المواسم المطلوب اختبارها يتجاوز حجم قاعدة البيانات المتوفرة!")
        else:
            with st.spinner("جاري تجهيز بيئة الاختبار المعزولة..."):
                start_time = time.time()
                
                # 1. فصل البيانات
                split_idx = len(features_df) - matches_to_hide
                train_df = features_df.iloc[:split_idx]
                test_df = features_df.iloc[split_idx:]
                
                # 2. تدريب نموذج جديد (أعمى عن المواسم الأخيرة)
                backtest_ml = FortressML()
                backtest_ml.train(train_df)
                
                # 3. تجهيز بيانات الاختبار للتوقع السريع (Vectorization)
                X_test = test_df[['h_atk', 'h_def', 'h_pts', 'a_atk', 'a_def', 'a_pts', 'h2h_adv']]
                y_test = test_df['result'].values
                
                # 4. التوقع وحساب الدقة
                probs = backtest_ml.model.predict_proba(X_test)
                # استخراج أعلى احتمالين لكل مباراة
                top2_indices = np.argsort(probs, axis=1)[:, -2:]
                
                # حساب الإجابات الصحيحة (الفرصة المزدوجة)
                correct = sum([1 for i in range(len(y_test)) if y_test[i] in top2_indices[i]])
                accuracy = (correct / len(y_test)) * 100
                
                end_time = time.time()
                
                # 5. عرض النتائج بشكل رسومي أنيق
                st.markdown("### 📊 نتائج الاختبار الرجعي (Backtest Results)")
                
                col_res1, col_res2, col_res3 = st.columns(3)
                col_res1.metric("مباريات التدريب (الماضي)", f"{len(train_df)}")
                col_res2.metric("مباريات الاختبار (المستقبل)", f"{len(test_df)}")
                col_res3.metric("الوقت المستغرق", f"{(end_time - start_time):.2f} ثانية")
                
                st.divider()
                
                col_acc1, col_acc2 = st.columns(2)
                col_acc1.metric("✅ التوقعات الصحيحة (فرصة مزدوجة)", f"{correct} من {len(test_df)}")
                col_acc2.metric("🎯 نسبة الدقة الحقيقية", f"{accuracy:.2f}%")
                
                if accuracy > 70:
                    st.balloons()
                    st.success("🔥 دقة استثنائية! المحرك أثبت صلابته الإحصائية على المدى الطويل.")
                elif accuracy > 60:
                    st.info("👍 دقة جيدة جداً، النموذج يحقق أرباحاً إحصائية في نظام الفرصة المزدوجة.")
                else:
                    st.warning("⚠️ النموذج يحتاج إلى بيانات أو ميزات إضافية لرفع الدقة.")
