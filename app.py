import streamlit as st
import numpy as np
import time
import requests
from engine.data_processor import DataProcessor
from engine.ml_model import FortressML
from engine.odds_fetcher import OddsFetcher
from engine.fixtures_fetcher import FixturesFetcher
from engine.multi_agent_board import MultiAgentBoard
from engine.team_dictionary import TeamDictionary
from engine.llm_expert import OracleLLM

# -------- ثوابت -------- #
MATCHES_PER_SEASON = 932

# -------- إعدادات الصفحة -------- #
st.set_page_config(page_title="المحرك الحصين V14", page_icon="🏰", layout="wide")

st.markdown("""
<style>
.rtl-text { 
    direction: rtl; 
    text-align: right; 
    font-family: 'Arial', sans-serif; 
    line-height: 1.6; 
}
div[data-testid="stMetricValue"] { 
    font-size: 1.8rem; 
}
div[data-testid="stAlert"] { 
    direction: rtl; 
    text-align: right; 
}
</style>
""", unsafe_allow_html=True)

# -------- تليجرام -------- #
def send_telegram_alert(bot_token, chat_id, message):
    if not bot_token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id, 
        "text": message, 
        "parse_mode": "Markdown"
    }
    try:
        resp = requests.post(url, json=payload, timeout=5)
        resp.raise_for_status()
    except Exception as e:
        st.sidebar.warning(f"⚠️ فشل إرسال تنبيه تليجرام: {e}")

# -------- تحميل المحرك -------- #
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
st.markdown("<div class='rtl-text'>XGBoost + Cerebras Debate + Groq Manager + Live Odds & Fixtures</div>", unsafe_allow_html=True)
st.markdown("---")

try:
    with st.spinner("جاري طحن البيانات التاريخية وتهيئة غرفة العمليات..."):
        dp, ml, teams, match_count, features_df = load_and_train_engine()
        st.sidebar.success(f"📚 حجم قاعدة التدريب الكلية: {match_count} مباراة")
except Exception as e:
    st.error(f"خطأ في تحميل البيانات: {e}")
    st.stop()

# -------- القائمة الجانبية -------- #
st.sidebar.header("⚙️ إعدادات الذكاء الاصطناعي")
confidence_threshold = st.sidebar.slider("عتبة التدخل للـ LLM (الفرق %)", 5, 30, 15)

st.sidebar.markdown("---")

st.sidebar.header("💰 إعدادات الاستثمار (إدارة المخاطر)")
st.sidebar.info("الرادار سيتجاهل أي فريق نسبة فوزه في الآلة أقل من هذا الرقم.")
min_win_prob = st.sidebar.slider("الحد الأدنى لنسبة الفوز المقبولة (%)", 10, 80, 40) / 100.0

st.sidebar.markdown("---")

st.sidebar.header("📱 إشعارات تليجرام (اختياري)")
st.sidebar.info("ضع بيانات البوت الخاص بك لتلقي تنبيهات بالفرص الاستثمارية القوية.")
tg_token = st.sidebar.text_input("Bot Token", type="password")
tg_chat_id = st.sidebar.text_input("Chat ID")

tab1, tab2 = st.tabs(["🔮 غرفة العمليات والمناظرة", "📈 الفحص الرجعي (Backtest)"])

# ========== التبويب الأول ========== #
with tab1:
    st.subheader("📡 سحب المباريات المباشرة")
    fetcher = FixturesFetcher()
    live_matches = fetcher.get_upcoming_matches()
    
    input_method = st.radio("طريقة اختيار المباراة:", ["اختيار من مباريات الجولة القادمة", "إدخال يدوي"])
    home_team, away_team = teams[0], teams[1]
    
    if input_method == "اختيار من مباريات الجولة القادمة" and live_matches:
        match_options = [f"{m['home']} vs {m['away']} - {m['time']}" for m in live_matches]
        selected_match = st.selectbox("اختر مباراة لتحليلها:", match_options)
        
        raw_home = selected_match.split(" vs ")[0].strip()
        raw_away = selected_match.split(" vs ")[1].split(" - ")[0].strip()
        
        home_team = TeamDictionary.get_closest_team(raw_home, teams)
        away_team = TeamDictionary.get_closest_team(raw_away, teams)
        
        if not home_team or not away_team:
            st.error("❌ لم نتمكن من مطابقة أسماء الفرق. استخدم الإدخال اليدوي.")
            st.stop()
        st.info(f"تمت المطابقة: **{home_team}** ضد **{away_team}**")
    else:
        col1, col2 = st.columns(2)
        with col1:
            home_team = st.selectbox("فريق الأرض (Home)", teams, index=teams.index("Arsenal") if "Arsenal" in teams else 0)
        with col2:
            away_team = st.selectbox("الفريق الضيف (Away)", teams, index=teams.index("Chelsea") if "Chelsea" in teams else 1)

    if st.button("🚀 بدء المناظرة وتحليل المواجهة", use_container_width=True):
        if home_team == away_team:
            st.warning("الرجاء اختيار فريقين مختلفين!")
        else:
            # 1. المحرك الرياضي والمواجهات المباشرة
            match_x = dp.get_match_features(home_team, away_team)
            probs = ml.predict_match_probs(match_x)
            h_xg, a_xg = ml.predict_xg(match_x)
            
            try:
                h2h_data = dp.get_detailed_h2h(home_team, away_team)
                
                # عرض إحصائيات H2H بشكل أنيق
                st.markdown("### 📜 تاريخ المواجهات المباشرة (H2H)")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("إجمالي اللقاءات", h2h_data.get('total', 0))
                c2.metric(f"فوز {home_team}", h2h_data.get('home_wins', 0))
                c3.metric(f"فوز {away_team}", h2h_data.get('away_wins', 0))
                c4.metric("تعادل", h2h_data.get('draws', 0))
            except AttributeError:
                h2h_data = None
                st.warning("لم يتم العثور على دالة `get_detailed_h2h` في DataProcessor، تم تخطي عرض المواجهات المباشرة التفصيلية.")

            
            # 2. كوتا السوق
            odds_fetcher = OddsFetcher()
            odds_data, bookie_name = odds_fetcher.get_odds(home_team, away_team)
            
            st.markdown("#### 📊 القراءة الرقمية (XGBoost)")
            m1, m2, m3 = st.columns(3)
            m1.metric(f"فوز {home_team}", f"{probs[2]*100:.1f}%")
            m2.metric("تعادل", f"{probs[1]*100:.1f}%")
            m3.metric(f"فوز {away_team}", f"{probs[0]*100:.1f}%")
            
            # --- دمج OracleLLM لتوقع النتيجة الدقيقة ---
            st.markdown("#### 🤖 توقع النتيجة الدقيقة (Oracle LLM)")
            try:
                oracle = OracleLLM()
                with st.spinner("جاري استشارة الأوراكل لقراءة النتيجة النهائية..."):
                    exact_score = oracle.get_exact_score(home_team, away_team, h_xg, a_xg, probs)
                    double_chance = oracle.get_double_chance(home_team, away_team, probs)
                    
                    c_score, c_dc = st.columns(2)
                    c_score.info(f"🎯 **النتيجة المتوقعة:** {exact_score}")
                    c_dc.success(f"🛡️ **الخيار الآمن:** {double_chance}")
            except Exception as e:
                st.warning(f"⚠️ تعذر الاتصال بخوادم Groq لتوقع النتيجة الدقيقة: {e}")
            
            # 3. القيمة الاستثمارية
            st.divider()
            telegram_msg = ""
            
            if odds_data:
                st.caption(f"📌 مصدر الكوتا: **{bookie_name}**")
                ev_home = (probs[2] * odds_data['home']) - 1
                ev_draw = (probs[1] * odds_data['draw']) - 1
                ev_away = (probs[0] * odds_data['away']) - 1
                
                best_label = np.argmax([ev_away, ev_draw, ev_home])
                best_ev = [ev_away, ev_draw, ev_home][best_label]
                
                if best_ev > 0.05:
                    if best_label == 2:
                        bt, ov, mp = f"فوز {home_team}", odds_data['home'], probs[2]
                    elif best_label == 1:
                        bt, ov, mp = "التعادل", odds_data['draw'], probs[1]
                    else:
                        bt, ov, mp = f"فوز {away_team}", odds_data['away'], probs[0]
                        
                    if mp >= min_win_prob:
                        invest_text = (f"💰 **قيمة استثمارية مكتشفة:** رهان على **{bt}** " 
                                       f"بكوتا ({ov}). العائد المتوقع: +{best_ev*100:.1f}%")
                        st.success(invest_text)
                        telegram_msg = (f"🚨 **تنبيه استثماري جديد** 🚨\n" 
                                        f"المباراة: {home_team} 🆚 {away_team}\n\n" 
                                        f"{invest_text}\n\n")
                    else:
                        st.warning(f"🛡️ **تم حجب مخاطرة:** فرصة ({bt}) جيدة مالياً " 
                                   f"ولكنها ضعيفة إحصائياً ({mp*100:.1f}%).")
                else:
                    st.info("لا توجد قيمة استثمارية واضحة في هذه المباراة.")
            else:
                st.warning("⚠️ لم نتمكن من جلب كوتا السوق لهذه المباراة.")
                
            # 4. المناظرة
            st.divider()
            st.subheader("🏛️ اجتماع مجلس الخبراء (Live AI Debate)")
            try:
                with st.spinner("جاري إدارة المناظرة بين الخبراء..."):
                    board = MultiAgentBoard(confidence_threshold=confidence_threshold)
                    
                    s_rep, t_rep, v_rep, debate_content, manager_decision = board.run_board_meeting(
                        home_team, away_team, h_xg, a_xg, probs, odds_data, h2h_data if h2h_data else None
                    )
                    
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.info("**📊 الإحصائي:**")
                        st.markdown(f"<div class='rtl-text'>{s_rep}</div>", unsafe_allow_html=True)
                    with c2:
                        st.warning("**⚽ التكتيكي:**")
                        st.markdown(f"<div class='rtl-text'>{t_rep}</div>", unsafe_allow_html=True)
                    with c3:
                        st.success("**💰 المالي:**")
                        st.markdown(f"<div class='rtl-text'>{v_rep}</div>", unsafe_allow_html=True)
                        
                    st.divider()
                    st.markdown("### 👑 القرار النهائي للمدير (خلاصة المناظرة)")
                    st.error(f"**{manager_decision}**")
            except Exception as e:
                st.error(f"⚠️ فشل الاتصال بخوادم الذكاء الاصطناعي: {e}")
                
            if telegram_msg and tg_token and tg_chat_id:
                telegram_msg += f"👑 **قرار المدير النهائي:**\n{manager_decision}"
                send_telegram_alert(tg_token, tg_chat_id, telegram_msg)

# ========== التبويب الثاني (الفحص الرجعي الشامل) ========== #
with tab2:
    st.subheader("📊 اختبار دقة النموذج الشامل على المواسم الماضية")
    seasons_to_hide = st.slider("سنوات الاختبار (مواسم)", 1, 10, 5)
    
    if st.button("⚙️ بدء الفحص الرجعي الشامل", type="primary", use_container_width=True):
        matches_to_hide = seasons_to_hide * MATCHES_PER_SEASON
        
        if matches_to_hide >= len(features_df):
            st.error(f"بيانات غير كافية. المتاح: {len(features_df)} مباراة، المطلوب: {matches_to_hide}")
        else:
            with st.spinner("جاري طحن البيانات واختبار الخوارزميات..."):
                split_idx = len(features_df) - matches_to_hide
                train_df = features_df.iloc[:split_idx]
                test_df = features_df.iloc[split_idx:]
                
                backtest_ml = FortressML()
                backtest_ml.train(train_df)
                
                # تحديد الميزات الـ 11 بالترتيب الدقيق كما تدرب عليها النموذج في ml_model.py
                feature_cols = [
                    'h_atk', 'h_def', 'h_pts', 'h_avg_scored_5', 'h_avg_conceded_5', 
                    'a_atk', 'a_def', 'a_pts', 'a_avg_scored_5', 'a_avg_conceded_5', 
                    'h2h_adv'
                ]

                # التحقق من وجود جميع الأعمدة في test_df
                missing_cols = set(feature_cols) - set(test_df.columns)
                if missing_cols:
                    st.error(f"الأعمدة المفقودة في بيانات الاختبار: {missing_cols}")
                    st.stop()
                
                # أخذ الميزات الرقمية الصحيحة للاختبار
                X_test = test_df[feature_cols]
                y_test = test_df['result'].values
                actual_h_goals = test_df['h_goals'].values
                actual_a_goals = test_df['a_goals'].values
                
                probs_test = backtest_ml.model.predict_proba(X_test)
                pred_h_goals = np.round(np.clip(backtest_ml.model_reg_h.predict(X_test), 0, None))
                pred_a_goals = np.round(np.clip(backtest_ml.model_reg_a.predict(X_test), 0, None))
                
                total_matches = len(y_test)
                
                # أ. الفرصة المزدوجة
                top2_indices = np.argsort(probs_test, axis=1)[:, -2:]
                correct_dc = sum(1 for i in range(total_matches) if y_test[i] in top2_indices[i])
                acc_dc = (correct_dc / total_matches) * 100
                
                # ب. الربح المباشر (1X2)
                top1_indices = np.argmax(probs_test, axis=1)
                correct_direct = np.sum(y_test == top1_indices)
                acc_direct = (correct_direct / total_matches) * 100
                
                # ج. النتيجة الدقيقة
                correct_exact = np.sum((pred_h_goals == actual_h_goals) & (pred_a_goals == actual_a_goals))
                acc_exact = (correct_exact / total_matches) * 100
                
                # د. أهداف (Over/Under 2.5)
                pred_total = pred_h_goals + pred_a_goals
                actual_total = actual_h_goals + actual_a_goals
                correct_ou25 = np.sum((pred_total > 2.5) == (actual_total > 2.5))
                acc_ou25 = (correct_ou25 / total_matches) * 100
                
                st.success(f"✅ تم التدريب على **{len(train_df)}** مباراة، واختبار الآلة على **{total_matches}** مباراة حقيقية.")
                
                col1, col2 = st.columns(2)
                col3, col4 = st.columns(2)
                with col1:
                    st.metric("🛡️ الفرصة المزدوجة (Double Chance)", f"{acc_dc:.2f}%")
                    st.progress(min(acc_dc / 100, 1.0))
                with col2:
                    st.metric("🎯 الربح المباشر (Match Winner)", f"{acc_direct:.2f}%")
                    st.progress(min(acc_direct / 100, 1.0))
                with col3:
                    st.metric("⚽ الأهداف (Over/Under 2.5)", f"{acc_ou25:.2f}%")
                    st.progress(min(acc_ou25 / 100, 1.0))
                with col4:
                    st.metric("🔮 النتيجة الدقيقة (Exact Score)", f"{acc_exact:.2f}%")
                    st.progress(min(acc_exact / 100, 1.0))
                
                st.info("💡 يتم الآن حساب كافة المقاييس بناءً على الميزات المتاحة.")
