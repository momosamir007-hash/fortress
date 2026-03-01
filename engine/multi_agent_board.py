import requests
import os
import json
import streamlit as st
import concurrent.futures
from groq import Groq

# ============================================================
# 1. إدارة قاموس التعريب والأسماء
# ============================================================
def load_team_dictionary(filepath="teams_dictionary.json"):
    """تحميل قاموس أسماء الفرق من ملف JSON الخارجي"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}

ARABIC_TEAM_NAMES = load_team_dictionary()

def translate_team(english_name):
    """البحث عن اسم الفريق وتعريبه بدقة"""
    clean_eng_name = english_name.strip().lower()
    for key, arabic_name in ARABIC_TEAM_NAMES.items():
        if key.strip().lower() == clean_eng_name:
            return arabic_name
    return english_name

def get_secret(key_name):
    """جلب المفتاح من أسرار Streamlit أو متغيرات البيئة"""
    try:
        if hasattr(st, 'secrets') and key_name in st.secrets:
            return st.secrets[key_name]
    except:
        pass
    return os.getenv(key_name)

# ============================================================
# 2. إعدادات النماذج (Cerebras & Groq)
# ============================================================
PRIMARY_MODEL = "llama3.1-70b"
FALLBACK_MODEL = "llama3.1-8b"
FAST_MODEL = "llama3.1-8b"
GROQ_MODEL = "llama-3.3-70b-versatile"

class MultiAgentBoard:
    def __init__(self, confidence_threshold=15):
        self.cerebras_key = get_secret("CEREBRAS_API_KEY")
        self.groq_key = get_secret("GROQ_API_KEY")
        self.confidence_threshold = confidence_threshold
        self.groq_client = None
        
        if self.groq_key:
            try:
                self.groq_client = Groq(api_key=self.groq_key)
            except Exception as e:
                st.warning(f"⚠️ فشل تهيئة عميل Groq: {e}")

    def ask_cerebras_expert(self, system_prompt, user_prompt, model_id):
        """الاتصال بخوادم Cerebras لجلب تحليل الخبير"""
        if not self.cerebras_key:
            return "❌ مفتاح Cerebras مفقود."
            
        url = "https://api.cerebras.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cerebras_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 800
        }

        try:
            r = requests.post(url, headers=headers, json=payload, timeout=30)
            if r.status_code == 200:
                return r.json()['choices'][0]['message']['content']
        except:
            pass
        return "❌ تعذر جلب التحليل حالياً."

    def run_board_meeting(self, home_team_eng, away_team_eng, h_xg, a_xg, probs, odds_data, h2h_details):
        """إدارة المناظرة الكبرى بين الخبراء"""
        home_team = translate_team(home_team_eng)
        away_team = translate_team(away_team_eng)
        
        # تحويل بيانات H2H إلى سياق نصي للخبير التكتيكي
        h2h_context = f"""
        سجل المواجهات المباشرة التاريخي:
        - إجمالي المباريات: {h2h_details['total']}
        - فوز {home_team}: {h2h_details['home_wins']}
        - فوز {away_team}: {h2h_details['away_wins']}
        - التعادلات: {h2h_details['draws']}
        """

        # بناء السياق العام للمباراة
        main_context = (
            f"المباراة: {home_team} ضد {away_team}.\n"
            f"الإحصائيات الحالية:\n"
            f"- فوز الأرض: {probs[2]*100:.1f}% | تعادل: {probs[1]*100:.1f}% | فوز الضيف: {probs[0]*100:.1f}%\n"
            f"- الأهداف المتوقعة: {h_xg:.2f} - {a_xg:.2f}\n"
            f"{h2h_context}"
        )

        # تعريف أدوار الخبراء
        experts = [
            {"role": "إحصائي", "sys": "أنت محلل بيانات رياضي. ركز على الأرقام والنسب المئوية فقط بالعربية.", "model": FAST_MODEL},
            {"role": "تكتيكي", "sys": "أنت محلل تكتيكي. ركز على H2H والعقد النفسية وسيناريو اللعب بالعربية.", "model": PRIMARY_MODEL},
            {"role": "مالي", "sys": "أنت خبير مخاطر. قيم القيمة الاستثمارية بناءً على الكوتا والنسب بالعربية.", "model": FAST_MODEL}
        ]

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(self.ask_cerebras_expert, e["sys"], main_context, e["model"]) for e in experts]
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                # حماية ضد الأخطاء
                if "❌" in res or "⏰" in res:
                    results.append("بيانات هذا الخبير غير متوفرة لهذه المواجهة.")
                else:
                    results.append(res)

        stat_rep, tactic_rep, finance_rep = results[0], results[1], results[2]

        # صياغة المناظرة النهائية
        debate_prompt = f"أدر مناظرة فنية حادة بين هؤلاء الثلاثة: الإحصائي ({stat_rep}), التكتيكي ({tactic_rep}), والمالي ({finance_rep}). ركز على نقاط الخلاف."
        debate_text = self.ask_cerebras_expert("أنت مخرج استوديو تحليلي محترف.", debate_prompt, PRIMARY_MODEL)

        # قرار المدير النهائي عبر Groq
        decision = self._get_manager_decision(home_team, away_team, debate_text)
        
        return stat_rep, tactic_rep, finance_rep, debate_text, decision

    def _get_manager_decision(self, home_team, away_team, debate_text):
        if not self.groq_client:
            return "⚠️ قرار المدير غير متاح (مفتاح Groq مفقود)."
            
        manager_prompt = (
            f"بصفتك مدير غرفة العمليات، حلل المناظرة التالية لمباراة {home_team} ضد {away_team}:\n"
            f"{debate_text}\n\n"
            f"أصدر قرارك النهائي بالتنسيق التالي:\n"
            f"النتيجة المتوقعة: [X-Y]\n"
            f"التوقع المزدوج: [مثلاً الأرض أو تعادل]\n"
            f"نسبة الثقة: [X]%\n"
            f"الخلاصة: [سطرين]"
        )
        try:
            response = self.groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": manager_prompt}],
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ خطأ المدير: {str(e)[:50]}"
