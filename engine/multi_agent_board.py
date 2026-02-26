import requests
import os
import streamlit as st
import concurrent.futures
from groq import Groq

# دالة آمنة لجلب المفاتيح بشكل منفصل لمنع تداخل الأخطاء
def get_secret(key_name):
    try:
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
    return os.getenv(key_name)

class MultiAgentBoard:
    def __init__(self):
        self.cerebras_key = get_secret("CEREBRAS_API_KEY")
        self.groq_key = get_secret("GROQ_API_KEY")
        
        if not self.groq_key:
            st.warning("تنبيه: مفتاح GROQ_API_KEY غير موجود.")
        else:
            self.groq_client = Groq(api_key=self.groq_key)

    def ask_cerebras_expert(self, system_prompt, user_prompt):
        if not self.cerebras_key:
            return "❌ خطأ: مفتاح CEREBRAS_API_KEY مفقود من الإعدادات."
            
        url = "https://api.cerebras.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cerebras_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "llama3.1-70b", # النموذج السريع من Cerebras
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 150
        }
        try:
            response = requests.post(url, headers=headers, json=data)
            # إذا نجح الاتصال
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            # إذا رفض الخادم الاتصال، نطبع الخطأ لنعرف السبب
            else:
                return f"❌ خطأ من الخادم ({response.status_code}): {response.text}"
        except Exception as e:
            return f"❌ فشل الاتصال بالإنترنت: {str(e)}"

    def run_board_meeting(self, home_team, away_team, h_xg, a_xg, probs, odds_data):
        odds_text = f"كوتا السوق: {odds_data}" if odds_data else "غير متوفرة"
        context = f"المباراة: {home_team} ضد {away_team}\nفرص الآلة: فوز الأرض {probs[2]*100:.1f}%, تعادل {probs[1]*100:.1f}%, فوز الضيف {probs[0]*100:.1f}%\nأهداف متوقعة: {home_team} [{h_xg:.2f}] - [{a_xg:.2f}] {away_team}\n{odds_text}"

        # تجهيز شخصيات الخبراء
        experts = [
            ("أنت خبير إحصائي رياضي صارم. لا تشاهد المباريات، تعتمد فقط على الأرقام الاحتمالية المرفقة. لخص رأيك في سطرين.", context),
            ("أنت محلل تكتيكي خبير بالدوري الإنجليزي. حلل السياق التكتيكي وعوامل الأرض والتاريخ بناءً على الأرقام المعطاة. لخص في سطرين.", context),
            ("أنت صائد فرص استثمارية (Value Bettor). تحذر من الفخاخ وتقارن احتمالات الآلة بكوتا السوق لاصطياد الأخطاء. لخص نصيحتك في سطرين.", context)
        ]

        # تشغيل الخبراء بالتوازي
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.ask_cerebras_expert, p[0], p[1]) for p in experts]
            results = [f.result() for f in futures]

        stats_report, tactical_report, value_report = results

        # إرسال التقارير لمدير Groq لاتخاذ القرار
        manager_prompt = f"""
        أنت المدير العام (Groq). أمامك تقارير خبرائك حول مباراة ({home_team} ضد {away_team}):
        1. الإحصائي: {stats_report}
        2. التكتيكي: {tactical_report}
        3. المستثمر: {value_report}
        
        استنتج قرارك النهائي وأجبني بالصيغة التالية فقط:
        النتيجة الدقيقة: X-Y
        التوقع المزدوج: (اكتب الخيار الآمن)
        الخلاصة: (سطر واحد يلخص سبب اختيارك)
        """
        try:
            if hasattr(self, 'groq_client'):
                manager_decision = self.groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": manager_prompt}],
                    temperature=0.1
                ).choices[0].message.content
            else:
                manager_decision = "تعذر التوقع، مفتاح Groq مفقود."
        except Exception as e:
            manager_decision = f"فشل المدير في اتخاذ القرار: {str(e)}"

        return stats_report, tactical_report, value_report, manager_decision
