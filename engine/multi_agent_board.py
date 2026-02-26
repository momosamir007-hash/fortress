import requests
import os
import streamlit as st
import concurrent.futures
from groq import Groq

class MultiAgentBoard:
    def __init__(self):
        try:
            self.cerebras_key = st.secrets["CEREBRAS_API_KEY"]
            self.groq_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            self.cerebras_key = os.getenv("CEREBRAS_API_KEY")
            self.groq_key = os.getenv("GROQ_API_KEY")
            
        self.groq_client = Groq(api_key=self.groq_key)

    def ask_cerebras_expert(self, system_prompt, user_prompt):
        url = "https://api.cerebras.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cerebras_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "llama3.1-70b", # نموذج سريع جداً وذكي
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 150
        }
        try:
            response = requests.post(url, headers=headers, json=data).json()
            return response['choices'][0]['message']['content']
        except Exception as e:
            return "تعذر جلب التحليل."

    def run_board_meeting(self, home_team, away_team, h_xg, a_xg, probs, odds_data):
        odds_text = f"كوتا السوق: {odds_data}" if odds_data else "غير متوفرة"
        context = f"المباراة: {home_team} ضد {away_team}\nفرص الآلة: فوز الأرض {probs[2]*100:.1f}%, تعادل {probs[1]*100:.1f}%, فوز الضيف {probs[0]*100:.1f}%\nأهداف متوقعة: {home_team} [{h_xg:.2f}] - [{a_xg:.2f}] {away_team}\n{odds_text}"

        # تجهيز شخصيات الخبراء
        experts = [
            ("أنت خبير إحصائي رياضي صارم. لا تشاهد المباريات، تعتمد فقط على الأرقام والاحتمالات الرياضية. لخص رأيك في سطرين.", context),
            ("أنت محلل تكتيكي خبير بالدوري الإنجليزي. حلل السياق التكتيكي وعوامل الأرض والتاريخ بناءً على الأرقام المعطاة. لخص في سطرين.", context),
            ("أنت صائد فرص استثمارية (Value Bettor). تحذر من الفخاخ وتقارن احتمالات الآلة بكوتا السوق لاصطياد الأخطاء. لخص نصيحتك في سطرين.", context)
        ]

        # تشغيل الخبراء الثلاثة في نفس الوقت (التوازي) لضمان السرعة عبر Cerebras
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
            manager_decision = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": manager_prompt}],
                temperature=0.1
            ).choices[0].message.content
        except Exception:
            manager_decision = "تعذر الوصول للمدير."

        return stats_report, tactical_report, value_report, manager_decision
