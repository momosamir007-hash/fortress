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

# ---------------- إعدادات الصفحة و CSS المخصص (للمحاذاة RTL) ---------------- #
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
    /* إجبار جميع صناديق التنبيهات على المحاذاة من اليمين لليسار */
    div[data-testid="stAlert"] {
        direction: rtl;
        text-align: right;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- دالة إرسال تنبيهات تليجرام ---------------- #
def send_telegram_alert(bot_token, chat_id, message):
    if not bot_token or not chat_id:
        return # تجاهل الإرسال إذا لم يقم المستخدم بإدخال البيانات
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print(f"فشل إرسال التنبيه إلى تليجرام: {e}")

# ---------------- دوال التحميل والتدريب ---------------- #
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

# ---------------- إعدادات القائمة الجانبية ---------------- #
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

# ==========================================
# التبويب الأول: غرفة العمليات (المناظرة المباشرة)
# ==========================================
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
        st.info(f"تمت المطابقة: **{home_team}** ضد **{away_team}**")
        
    else:
        col1, col2 = st.columns(2)
        with col1: home_team = st.selectbox("فريق الأرض (Home)", teams, index=teams.index("Arsenal") if "Arsenal" in teams else 0)
        with col2: away_team = st.selectbox("الفريق الضيف (Away)", teams, index=teams.index("Chelsea") if "Chelsea" in teams else 1)

    if st.button("🚀 بدء المناظرة وتحليل المواجهة", use_container_width=True):
        if home_team == away_team:
            st.warning("الرجاء اختيار فريقين مختلفين!")
        else:
            # 1. المحرك الرياضي
            match_x = dp.get_match_features(home_team, away_team)
            probs = ml.predict_match_probs(match_x)
            h_xg, a_xg = ml.predict_xg(match_x)
            
            # 2. كوتا السوق
            odds_fetcher = OddsFetcher()
            odds_data, bookie_name = odds_fetcher.get_odds(home_team, away_team)
            
            st.markdown("#### 📊 القراءة الرقمية (XGBoost)")
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric(f"فوز {home_team}", f"{probs[2]*100:.1f}%")
            m_col2.metric("تعادل", f"{probs[1]*100:.1f}%")
            m_col3.metric(f"فوز {away_team}", f"{probs[0]*100:.1f}%")
            
            # 3. القيمة الاستثمارية (وتهيئة رسالة تليجرام)
            st.divider()
            telegram_msg = ""
            
            if odds_data:
                ev_home = (probs[2] * odds_data['home']) - 1
                ev_draw = (probs[1] * odds_data['draw']) - 1
                ev_away = (probs[0] * odds_data['away']) - 1
                best_ev = max(ev_home, ev_draw, ev_away)
                
                if best_ev > 0.05:
                    if best_ev == ev_home: bt, ov, mp = f"فوز {home_team}", odds_data['home'], probs[2]
                    elif best_ev == ev_draw: bt, ov, mp = "التعادل", odds_data['draw'], probs[1]
                    else: bt, ov, mp = f"فوز {away_team}", odds_data['away'], probs[0]
                    
                    if mp >= min_win_prob:
                        invest_text = f"💰 **قيمة استثمارية مكتشفة:** رهان على **{bt}** بكوتا ({ov}). العائد المتوقع: +{best_ev*100:.1f}%"
                        st.success(invest_text)
                        telegram_msg += f"🚨 **تنبيه استثماري جديد** 🚨\nالمباراة: {home_team} 🆚 {away_team}\n\n{invest_text}\n\n"
                    else:
                        st.warning(f"🛡️ **تم حجب مخاطرة:** فرصة ({bt}) جيدة مالياً ولكنها ضعيفة إحصائياً ({mp*100:.1f}%).")

            # 4. غرفة العمليات والمناظرة (Cerebras + Groq)
            st.divider()
            st.subheader("🏛️ اجتماع مجلس الخبراء (Live AI Debate)")
            
            with st.spinner("جاري إدارة المناظرة بين الخبراء عبر Cerebras..."):
                board = MultiAgentBoard()
                s_rep, t_rep, v_rep, debate_content, manager_decision = board.run_board_meeting(
                    home_team, away_team, h_xg, a_xg, probs, odds_data
                )
                
                exp_col1, exp_col2, exp_col3 = st.columns(3)
                with exp_col1: 
                    st.info("**📊 الإحصائي:**")
                    st.markdown(f"<div class='rtl-text'>{s_rep}</div>", unsafe_allow_html=True)
                with exp_col2: 
                    st.warning("**⚽ التكتيكي:**")
                    st.markdown(f"<div class='rtl-text'>{t_rep}</div>", unsafe_allow_html=True)
                with exp_col3: 
                    st.success("**💰 المالي:**")
                    st.markdown(f"<div class='rtl-text'>{v_rep}</div>", unsafe_allow_html=True)
            
            with st.expander("📺 شاهد تفاصيل المناظرة المباشرة بين الخبراء"):
                st.markdown(f"<div class='rtl-text'>{debate_content}</div>", unsafe_allow_html=True)
                    
            st.divider()
            st.markdown("### 👑 القرار النهائي للمدير (خلاصة المناظرة)")
            st.error(f"**{manager_decision}**")
            
            # إرسال التنبيه إلى تليجرام إذا كانت هناك فرصة قوية
            if telegram_msg and tg_token and tg_chat_id:
                telegram_msg += f"👑 **قرار المدير النهائي:**\n{manager_decision}"
                send_telegram_alert(tg_token, tg_chat_id, telegram_msg)

# ==========================================
# التبويب الثاني: الفحص الرجعي (Backtest)
# ==========================================
with tab2:
    st.subheader("اختبار دقة النموذج على المواسم الماضية")
    seasons_to_hide = st.slider("سنوات الاختبار", 1, 10, 5)
    
    if st.button("⚙️ بدء الفحص الرجعي", type="primary", use_container_width=True):
        matches_to_hide = seasons_to_hide * 932
        if matches_to_hide >= len(features_df):
            st.error("بيانات غير كافية")
        else:
            with st.spinner("جاري الحساب..."):
                split_idx = len(features_df) - matches_to_hide
                train_df = features_df.iloc[:split_idx]
                test_df = features_df.iloc[split_idx:]
                
                backtest_ml = FortressML()
                backtest_ml.train(train_df)
                
                X_test = test_df[['h_atk', 'h_def', 'h_pts', 'a_atk', 'a_def', 'a_pts', 'h2h_adv']]
                y_test = test_df['result'].values
                
                probs_test = backtest_ml.model.predict_proba(X_test)
                top2_indices = np.argsort(probs_test, axis=1)[:, -2:]
                
                correct = sum([1 for i in range(len(y_test)) if y_test[i] in top2_indices[i]])
                accuracy = (correct / len(y_test)) * 100
                
                st.metric("🎯 نسبة الدقة الحقيقية (Double Chance)", f"{accuracy:.2f}%")
                st.progress(accuracy / 100)
