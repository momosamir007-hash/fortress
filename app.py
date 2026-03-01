import streamlit as st
import numpy as np
import time
import requests
from scipy.stats import poisson  # 💡 إضافة مكتبة التوزيع الاحتمالي
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

# -------- دوال رياضية متقدمة (بواسون) -------- #
def calculate_goal_lines(h_xg, a_xg):
    """
    تحسب احتمالات كل خطوط الأهداف (Over/Under) باستخدام توزيع بواسون
    بناءً على الأهداف المتوقعة (xG) من نموذج XGBoost.
    """
    h_probs = [poisson.pmf(i, h_xg) for i in range(7)]
    a_probs = [poisson.pmf(i, a_xg) for i in range(7)]
    
    exact_score_probs = np.outer(h_probs, a_probs)
    
    lines = {}
    for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
        under_prob = 0.0
        for h_goals in range(7):
            for a_goals in range(7):
                if h_goals + a_goals < line:
                    under_prob += exact_score_probs[h_goals, a_goals]
        
        over_prob = 1.0 - under_prob
        lines[f"O/U {line}"] = {
            "Over": over_prob,
            "Under": under_prob
        }
    return lines, exact_score_probs

# -------- تليجرام -------- #
def send_telegram_alert(bot_token, chat_id, message):
    if not bot_token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
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
            match_x = dp.get_match_features(home_team, away_team)
            probs = ml.predict_match_probs(match_x)
            h_xg, a_xg = ml.predict_xg(match_x)
            
            try:
                h2h_data = dp.get_detailed_h2h(home_team, away_team)
                st.markdown("### 📜 تاريخ المواجهات المباشرة (H2H)")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("إجمالي اللقاءات", h2h_data.get('total', 0))
                c2.metric(f"فوز {home_team}", h2h_data.get('home_wins', 0))
                c3.metric(f"فوز {away_team}", h2h_data.get('away_wins', 0))
                c4.metric("تعادل", h2h_data.get('draws', 0))
            except AttributeError:
                h2h_data = None
                st.warning("لم يتم العثور على دالة المواجهات المباشرة، تم التخطي.")

            odds_fetcher = OddsFetcher()
            odds_data, bookie_name = odds_fetcher.get_odds(home_team, away_team)
            
            st.markdown("#### 📊 القراءة الرقمية (XGBoost)")
            m1, m2, m3 = st.columns(3)
            m1.metric(f"فوز {home_team}", f"{probs[2]*100:.1f}%")
            m2.metric("تعادل", f"{probs[1]*100:.1f}%")
            m3.metric(f"فوز {away_team}", f"{probs[0]*100:.1f}%")
            
            # ==========================================
            # ⚽ رادار الأهداف الشامل (Poisson Distribution)
            # ==========================================
            st.divider()
            st.markdown("### ⚽ رادار الأهداف الشامل (بناءً على xG)")
            
            goal_lines, exact_score_matrix = calculate_goal_lines(h_xg, a_xg)
            most_likely_h, most_likely_a = np.unravel_index(exact_score_matrix.argmax(), exact_score_matrix.shape)
            math_exact_score = f"{most_likely_h} - {most_likely_a}"
            math_exact_prob = exact_score_matrix[most_likely_h, most_likely_a] * 100
            
            st.info(f"🎯 **النتيجة الدقيقة الأقرب رياضياً:** {math_exact_score} (بنسبة ثقة {math_exact_prob:.1f}%)")
            
            col_g1, col_g2, col_g3 = st.columns(3)
            lines_to_show = ["O/U 1.5", "O/U 2.5", "O/U 3.5"]
            cols = [col_g1, col_g2, col_g3]
            
            for col, line in zip(cols, lines_to_show):
                over_p = goal_lines[line]['Over'] * 100
                under_p = goal_lines[line]['Under'] * 100
                
                if over_p > 65.0:
                    rec = f"🔥 أكثر (Over) {over_p:.1f}%"
                elif under_p > 65.0:
                    rec = f"❄️ أقل (Under) {under_p:.1f}%"
                else:
                    rec = f"⚖️ متوازن (O:{over_p:.0f}% | U:{under_p:.0f}%)"
                    
                col.metric(f"خط الأهداف {line}", rec)

            st.markdown("#### 🤖 الرؤية التكتيكية (Oracle LLM)")
            try:
                oracle = OracleLLM()
                with st.spinner("جاري استشارة الأوراكل لقراءة السيناريو التكتيكي..."):
                    llm_double_chance = oracle.get_double_chance(home_team, away_team, probs)
                    st.success(f"🛡️ **الخيار الآمن (LLM):** {llm_double_chance}")
            except Exception as e:
                st.warning(f"⚠️ تعذر الاتصال بخوادم Groq: {e}")
            
            # ==========================================
            # القيمة الاستثمارية والمناظرة
            # ==========================================
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
                        telegram_msg = (f"🚨 **تنبيه استثماري جديد** 🚨\nالمباراة: {home_team} 🆚 {away_team}\n\n{invest_text}\n\n")
                    else:
                        st.warning(f"🛡️ **تم حجب مخاطرة:** فرصة ({bt}) جيدة مالياً ولكنها ضعيفة إحصائياً ({mp*100:.1f}%).")
                else:
                    st.info("لا توجد قيمة استثمارية واضحة في هذه المباراة.")
            else:
                st.warning("⚠️ لم نتمكن من جلب كوتا السوق لهذه المباراة.")
                
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
    st.subheader("📊 اختبار دقة النموذج الشامل (مع فلتر الفوضى الاستثماري)")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        seasons_to_hide = st.slider("سنوات الاختبار (مواسم)", 1, 10, 5)
    with col_s2:
        chaos_filter_pct = st.slider("فلتر الفوضى (تجاهل المباريات التي فرق الثقة فيها أقل من %)", 0, 40, 15)
        chaos_filter = chaos_filter_pct / 100.0
    
    if st.button("⚙️ بدء الفحص الرجعي الشامل", type="primary", use_container_width=True):
        matches_to_hide = seasons_to_hide * MATCHES_PER_SEASON
        
        if matches_to_hide >= len(features_df):
            st.error(f"بيانات غير كافية. المتاح: {len(features_df)} مباراة، المطلوب: {matches_to_hide}")
        else:
            with st.spinner("جاري طحن البيانات وفلترة المباريات الفوضوية..."):
                split_idx = len(features_df) - matches_to_hide
                train_df = features_df.iloc[:split_idx]
                test_df = features_df.iloc[split_idx:]
                
                backtest_ml = FortressML()
                backtest_ml.train(train_df)
                
                feature_cols = [
                    'h_atk', 'h_def', 'h_pts', 'h_avg_scored_5', 'h_avg_conceded_5', 'h_rest_days', 'h_matchweek',
                    'a_atk', 'a_def', 'a_pts', 'a_avg_scored_5', 'a_avg_conceded_5', 'a_rest_days', 'a_matchweek',
                    'h2h_adv'
                ]

                missing_cols = set(feature_cols) - set(test_df.columns)
                if missing_cols:
                    st.error(f"الأعمدة المفقودة في بيانات الاختبار: {missing_cols}")
                    st.stop()
                
                X_test = test_df[feature_cols].values
                y_test_raw = test_df['result'].values
                actual_h_goals_raw = test_df['h_goals'].values
                actual_a_goals_raw = test_df['a_goals'].values
                
                probs_test_raw = backtest_ml.model.predict_proba(X_test)
                pred_h_goals_raw = np.round(np.clip(backtest_ml.model_reg_h.predict(X_test), 0, None))
                pred_a_goals_raw = np.round(np.clip(backtest_ml.model_reg_a.predict(X_test), 0, None))
                
                # تطبيق فلتر الفوضى الاستثماري
                sorted_probs = np.sort(probs_test_raw, axis=1)
                prob_diffs = sorted_probs[:, -1] - sorted_probs[:, -2] 
                
                valid_indices = np.where(prob_diffs >= chaos_filter)[0]
                ignored_matches = len(y_test_raw) - len(valid_indices)
                
                if len(valid_indices) == 0:
                    st.warning("⚠️ فلتر الفوضى صارم جداً! تم استبعاد كل المباريات. قم بتقليل النسبة.")
                    st.stop()
                
                y_test = y_test_raw[valid_indices]
                probs_test = probs_test_raw[valid_indices]
                actual_h_goals = actual_h_goals_raw[valid_indices]
                actual_a_goals = actual_a_goals_raw[valid_indices]
                pred_h_goals = pred_h_goals_raw[valid_indices]
                pred_a_goals = pred_a_goals_raw[valid_indices]
                
                total_matches_filtered = len(y_test)

                # حساب المقاييس
                top2_indices = np.argsort(probs_test, axis=1)[:, -2:]
                correct_dc = sum(1 for i in range(total_matches_filtered) if y_test[i] in top2_indices[i])
                acc_dc = (correct_dc / total_matches_filtered) * 100
                
                top1_indices = np.argmax(probs_test, axis=1)
                correct_direct = np.sum(y_test == top1_indices)
                acc_direct = (correct_direct / total_matches_filtered) * 100
                
                correct_exact = np.sum((pred_h_goals == actual_h_goals) & (pred_a_goals == actual_a_goals))
                acc_exact = (correct_exact / total_matches_filtered) * 100
                
                pred_total = pred_h_goals + pred_a_goals
                actual_total = actual_h_goals + actual_a_goals
                correct_ou25 = np.sum((pred_total > 2.5) == (actual_total > 2.5))
                acc_ou25 = (correct_ou25 / total_matches_filtered) * 100
                
                st.success(f"✅ تم التدريب على **{len(train_df)}** مباراة سابقة.")
                st.warning(f"🛡️ **نتيجة فلتر الفوضى:** تم تجاهل **{ignored_matches}** مباراة خطيرة، واعتماد **{total_matches_filtered}** مباراة استثمارية آمنة.")
                
                col1, col2 = st.columns(2)
                col3, col4 = st.columns(2)
                with col1:
                    st.metric("🛡️ الفرصة المزدوجة (للمباريات الآمنة)", f"{acc_dc:.2f}%")
                    st.progress(min(acc_dc / 100, 1.0))
                with col2:
                    st.metric("🎯 الربح المباشر (للمباريات الآمنة)", f"{acc_direct:.2f}%")
                    st.progress(min(acc_direct / 100, 1.0))
                with col3:
                    st.metric("⚽ الأهداف (Over/Under 2.5)", f"{acc_ou25:.2f}%")
                    st.progress(min(acc_ou25 / 100, 1.0))
                with col4:
                    st.metric("🔮 النتيجة الدقيقة (Exact Score)", f"{acc_exact:.2f}%")
                    st.progress(min(acc_exact / 100, 1.0))
                
                st.info("💡 جرب تغيير (سلايدر فلتر الفوضى) ولاحظ كيف ترتفع دقة الفرصة المزدوجة كلما أصبحت الآلة أكثر انتقائية!")
