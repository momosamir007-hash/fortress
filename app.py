import streamlit as st
import numpy as np
import time
from engine.data_processor import DataProcessor
from engine.ml_model import FortressML
from engine.odds_fetcher import OddsFetcher
from engine.fixtures_fetcher import FixturesFetcher
from engine.multi_agent_board import MultiAgentBoard

st.set_page_config(page_title="المحرك الحصين V14", page_icon="🏰", layout="wide")

@st.cache_resource
def load_and_train_engine():
    dp = DataProcessor()
    raw_df = dp.fetch_data()
    features_df = dp.extract_features(raw_df)
    
    ml = FortressML()
    ml.train(features_df)
    
    teams = sorted(list(set(raw_df['team1'].unique()) | set(raw_df['team2'].unique())))
    match_count = len(raw_df)
    return dp, ml, teams, match_count, features_df

st.title("🏰 المحرك الحصين V14 (غرفة العمليات الذكية)")
st.markdown("XGBoost + Cerebras Experts + Groq Manager + Live Odds & Fixtures")
st.markdown("---")

try:
    with st.spinner("جاري طحن البيانات التاريخية وتهيئة غرفة العمليات..."):
        dp, ml, teams, match_count, features_df = load_and_train_engine()
        st.sidebar.success(f"📚 حجم قاعدة التدريب الكلية: {match_count} مباراة")
except Exception as e:
    st.error(f"خطأ في تحميل البيانات: {e}")
    st.stop()

# --- إعدادات القائمة الجانبية ---
st.sidebar.header("⚙️ إعدادات الذكاء الاصطناعي")
confidence_threshold = st.sidebar.slider("عتبة التدخل للـ LLM (الفرق %)", 5, 30, 15)

st.sidebar.markdown("---")
st.sidebar.header("💰 إعدادات الاستثمار (إدارة المخاطر)")
st.sidebar.info("الرادار سيتجاهل أي فريق نسبة فوزه في الآلة أقل من هذا الرقم، لحمايتك من الفخاخ عالية المخاطرة.")
min_win_prob = st.sidebar.slider("الحد الأدنى لنسبة الفوز المقبولة (%)", 10, 80, 40) / 100.0

tab1, tab2 = st.tabs(["🔮 التوقع المباشر وغرفة العمليات", "📈 الفحص الرجعي (Backtest)"])

# ==========================================
# التبويب الأول: التوقع المباشر وغرفة العمليات
# ==========================================
with tab1:
    st.subheader("📡 سحب المباريات المباشرة")
    
    # محاولة جلب المباريات من API
    fetcher = FixturesFetcher()
    live_matches = fetcher.get_upcoming_matches()
    
    # طريقة اختيار المباراة
    input_method = st.radio("طريقة اختيار المباراة:", ["اختيار من مباريات الجولة القادمة", "إدخال يدوي"])
    
    home_team, away_team = teams[0], teams[1]
    
    if input_method == "اختيار من مباريات الجولة القادمة" and live_matches:
        match_options = [f"{m['home']} vs {m['away']} - {m['time']}" for m in live_matches]
        selected_match = st.selectbox("اختر مباراة لتحليلها:", match_options)
        
        # استخراج الأسماء الخام من الاختيار
        raw_home = selected_match.split(" vs ")[0].strip()
        raw_away = selected_match.split(" vs ")[1].split(" - ")[0].strip()
        
        # دالة ذكية لمطابقة أسماء الـ API مع أسماء قاعدة بياناتنا (XGBoost)
        def get_closest_team(raw_name, teams_list):
            for t in teams_list:
                if t.lower() in raw_name.lower() or raw_name.lower() in t.lower():
                    return t
            return teams_list[0] 
            
        home_team = get_closest_team(raw_home, teams)
        away_team = get_closest_team(raw_away, teams)
        st.info(f"تمت المطابقة مع قاعدة البيانات: **{home_team}** ضد **{away_team}**")
        
    else:
        if not live_matches and input_method == "اختيار من مباريات الجولة القادمة":
            st.warning("لا توجد مباريات قادمة حالياً في الـ API أو يوجد خطأ في الاتصال، تم التحويل للإدخال اليدوي.")
        col1, col2 = st.columns(2)
        with col1: home_team = st.selectbox("فريق الأرض (Home)", teams, index=teams.index("Arsenal") if "Arsenal" in teams else 0)
        with col2: away_team = st.selectbox("الفريق الضيف (Away)", teams, index=teams.index("Chelsea") if "Chelsea" in teams else 1)

    if st.button("🚀 بدء تحليل المواجهة واجتماع الخبراء", use_container_width=True):
        if home_team == away_team:
            st.warning("الرجاء اختيار فريقين مختلفين!")
        else:
            # 1. استخراج الأرقام من المحرك الرياضي
            match_x = dp.get_match_features(home_team, away_team)
            probs = ml.predict_match_probs(match_x)
            h_xg, a_xg = ml.predict_xg(match_x)
            
            # 2. جلب كوتا السوق (Odds)
            odds_fetcher = OddsFetcher()
            odds_data, bookie_name = odds_fetcher.get_odds(home_team, away_team)
            
            # --- عرض الأرقام الخام ---
            st.markdown("#### 📊 القراءة الرقمية الصارمة (XGBoost)")
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric(f"فوز {home_team} (أرض)", f"{probs[2]*100:.1f}%")
            m_col2.metric("تعادل", f"{probs[1]*100:.1f}%")
            m_col3.metric(f"فوز {away_team} (ضيف)", f"{probs[0]*100:.1f}%")
            
            # --- كاشف القيمة الاستثمارية (العائد الرياضي) ---
            st.divider()
            st.subheader("💰 كاشف القيمة الاستثمارية (إدارة المخاطر)")
            if odds_data:
                st.info(f"🏦 **الشركة المرجعية:** {bookie_name} | **الكوتا:** 🏠 ({odds_data['home']}) | 🤝 ({odds_data['draw']}) | ✈️ ({odds_data['away']})")
                ev_home = (probs[2] * odds_data['home']) - 1
                ev_draw = (probs[1] * odds_data['draw']) - 1
                ev_away = (probs[0] * odds_data['away']) - 1
                best_ev = max(ev_home, ev_draw, ev_away)
                
                if best_ev > 0.05: 
                    if best_ev == ev_home: bet_target, odds_val, machine_prob = f"فوز {home_team}", odds_data['home'], probs[2]
                    elif best_ev == ev_draw: bet_target, odds_val, machine_prob = "التعادل", odds_data['draw'], probs[1]
                    else: bet_target, odds_val, machine_prob = f"فوز {away_team}", odds_data['away'], probs[0]
                    
                    if machine_prob >= min_win_prob:
                        st.success(f"🔥 **فرصة استثمارية آمنة!** ننصح بالرهان على: **{bet_target}**. فرصة الفريق ({machine_prob*100:.1f}%)، والكوتا ({odds_val}). العائد المتوقع: **+{best_ev*100:.1f}%**")
                    else:
                        st.warning(f"🛡️ **تم حجب مخاطرة!** يوجد ربح في الرهان على **{bet_target}** بكوتا ({odds_val})، لكن فرصة الفوز الحقيقية ({machine_prob*100:.1f}%) أقل من حد الأمان الخاص بك.")
                else:
                    st.warning("⚠️ **لا توجد قيمة استثمارية!** الكوتا منخفضة جداً ولا تبرر المخاطرة، تجنب الرهان هنا.")
            else:
                st.warning("لم يتم العثور على كوتا حية لهذه المباراة في الأسواق المعتمدة.")

            # --- غرفة العمليات (Cerebras + Groq) ---
            st.divider()
            st.subheader("🏛️ اجتماع مجلس الخبراء (Live AI Debate)")
            with st.spinner("جاري استدعاء الخبراء عبر معالجات Cerebras السريعة جداً لمناقشة المباراة..."):
                board = MultiAgentBoard()
                s_rep, t_rep, v_rep, manager_decision = board.run_board_meeting(
                    home_team, away_team, h_xg, a_xg, probs, odds_data
                )
                
                exp_col1, exp_col2, exp_col3 = st.columns(3)
                with exp_col1:
                    st.info(f"**📊 الخبير الإحصائي:**\n\n{s_rep}")
                with exp_col2:
                    st.warning(f"**⚽ الخبير التكتيكي:**\n\n{t_rep}")
                with exp_col3:
                    st.success(f"**💰 خبير الاستثمار:**\n\n{v_rep}")
                    
            st.divider()
            st.markdown("### 👑 التقرير النهائي للمدير (Groq)")
            st.success(f"{manager_decision}")

# ==========================================
# التبويب الثاني: الفحص الرجعي (Backtest)
# ==========================================
with tab2:
    st.subheader("اختبار دقة النموذج الإحصائي على المواسم الماضية")
    st.info("في هذا الاختبار، سنقوم باقتطاع المواسم الأخيرة وإخفائها عن النموذج. سيتدرب النموذج على الماضي السحيق فقط، ثم نختبره على المواسم المخفية لنرى مدى دقته الحقيقية (النظام يعتمد على إحصائيات XGBoost هنا للسرعة).")
    
    seasons_to_hide = st.slider("كم موسماً تريد إخفاءه واختبار النموذج عليه؟", min_value=1, max_value=10, value=5)
    matches_per_season = 932 
    
    if st.button("⚙️ بدء الفحص الرجعي الآن", type="primary", use_container_width=True):
        matches_to_hide = seasons_to_hide * matches_per_season
        
        if matches_to_hide >= len(features_df):
            st.error("❌ عدد المواسم المطلوب اختبارها يتجاوز حجم قاعدة البيانات المتوفرة!")
        else:
            with st.spinner("جاري تجهيز بيئة الاختبار المعزولة والطحن العكسي للبيانات..."):
                start_time = time.time()
                
                split_idx = len(features_df) - matches_to_hide
                train_df = features_df.iloc[:split_idx]
                test_df = features_df.iloc[split_idx:]
                
                backtest_ml = FortressML()
                backtest_ml.train(train_df)
                
                X_test = test_df[['h_atk', 'h_def', 'h_pts', 'a_atk', 'a_def', 'a_pts', 'h2h_adv']]
                y_test = test_df['result'].values
                
                probs = backtest_ml.model.predict_proba(X_test)
                top2_indices = np.argsort(probs, axis=1)[:, -2:]
                
                correct = sum([1 for i in range(len(y_test)) if y_test[i] in top2_indices[i]])
                accuracy = (correct / len(y_test)) * 100
                
                end_time = time.time()
                
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
