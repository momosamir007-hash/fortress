import requests
import os
import streamlit as st
import concurrent.futures
from groq import Groq

def get_secret(key_name):
    try:
        if key_name in st.secrets:
            return st.secrets[key_name]
    except:
        pass
    return os.getenv(key_name)

class MultiAgentBoard:
    def __init__(self):
        self.cerebras_key = get_secret("CEREBRAS_API_KEY")
        self.groq_key = get_secret("GROQ_API_KEY")
        
        if self.groq_key:
            self.groq_client = Groq(api_key=self.groq_key)

    def ask_cerebras_expert(self, system_prompt, user_prompt, model_id):
        if not self.cerebras_key:
            return "❌ مفتاح Cerebras مفقود."
            
        url = "https://api.cerebras.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cerebras_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model_id, 
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 200
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=20)
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                return f"⚠️ خطأ {response.status_code} مع موديل {model_id}"
        except Exception as e:
            return f"❌ خطأ اتصال: {str(e)}"

    def run_board_meeting(self, home_team, away_team, h_xg, a_xg, probs, odds_data):
        context = (f"مباراة {home_team} vs {away_team}. "
                   f"الاحتمالات: فوز الأرض {probs[2]*100:.1f}%, تعادل {probs[1]*100:.1f}%, ضيف {probs[0]*100:.1f}%. "
                   f"الأهداف المتوقعة: {h_xg:.2f}-{a_xg:.2f}. الكوتا: {odds_data}")

        # تخصيص الموديلات بناءً على القائمة التي ظهرت في هاتفك
        # سنستخدم Llama للتحليل السريع و Qwen للتحليل المعمق
        experts = [
            ("أنت محلل إحصائي جاف. قدم تحليلاً للأرقام في سطرين.", context, "llama3.1-8b"),
            ("أنت محلل تكتيكي للدوري الإنجليزي. قدم رؤية فنية في سطرين.", context, "qwen-3-235b-a22b-instruct-2507"),
            ("أنت خبير مالي. قدم نصيحة بناءً على القيمة في سطرين.", context, "llama3.1-8b")
        ]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # نمرر الـ model_id لكل خبير بشكل مستقل
            futures = [executor.submit(self.ask_cerebras_expert, exp[0], exp[1], exp[2]) for exp in experts]
            results = [f.result() for f in futures]

        s_rep, t_rep, v_rep = results

        manager_prompt = f"""
        بصفتك المدير العام (Groq)، استخلص القرار النهائي من تقارير خبرائك ({home_team} vs {away_team}):
        إحصائياً: {s_rep}
        تكتيكياً: {t_rep}
        مالياً: {v_rep}
        
        المطلوب باختصار شديد:
        1. النتيجة الدقيقة
        2. التوقع المزدوج الآمن
        3. كلمة الختام (لماذا هذا القرار؟).
        """
        try:
            decision = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": manager_prompt}],
                temperature=0.1
            ).choices[0].message.content
        except:
            decision = "فشل المدير في دمج التحليلات."

        return s_rep, t_rep, v_rep, decision
