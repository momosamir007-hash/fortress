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

        self.groq_client = None
        if self.groq_key:
            self.groq_client = Groq(api_key=self.groq_key)

    # ---------------- CEREBRAS CALL ---------------- #

    def ask_cerebras_expert(self, system_prompt, user_prompt, model_id):

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
            "max_tokens": 200
        }

        # retry logic
        for attempt in range(3):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=60)

                if r.status_code == 200:
                    return r.json()['choices'][0]['message']['content']

                # fallback to faster model
                payload["model"] = "llama3.1-8b"
                r2 = requests.post(url, headers=headers, json=payload, timeout=30)

                if r2.status_code == 200:
                    return r2.json()['choices'][0]['message']['content']

            except Exception:
                continue

        return "❌ فشل الاتصال بالنموذج."

    # ---------------- BOARD MEETING ---------------- #

    def run_board_meeting(self, home_team, away_team, h_xg, a_xg, probs, odds_data):

        context = (
            f"مباراة {home_team} ضد {away_team}. "
            f"الاحتمالات: أرض {probs[2]*100:.1f}%، تعادل {probs[1]*100:.1f}%، ضيف {probs[0]*100:.1f}%. "
            f"xG: {h_xg:.2f}-{a_xg:.2f}. الكوتا: {odds_data}"
        )

        experts = [
            ("أنت محلل إحصائي. لخص في سطرين.", context, "llama3.1-8b"),
            ("أنت محلل تكتيكي. لخص في سطرين.", context, "qwen-3-235b-a22b-instruct-2507"),
            ("أنت مستشار مالي للمراهنات. لخص في سطرين.", context, "llama3.1-8b")
        ]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.ask_cerebras_expert, e[0], e[1], e[2])
                for e in experts
            ]
            results = [f.result() for f in futures]

        s_rep, t_rep, v_rep = results

        # ---------- MANAGER DECISION ---------- #

        if not self.groq_client:
            return s_rep, t_rep, v_rep, "⚠️ مفتاح Groq غير موجود."

        manager_prompt = f"""
اتخذ قرارك النهائي:

إحصائي:
{s_rep}

تكتيكي:
{t_rep}

مالي:
{v_rep}

أجب فقط:
النتيجة: X-Y
التوقع المزدوج:
الخلاصة:
"""

        try:
            decision = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": manager_prompt}],
                temperature=0.1
            ).choices[0].message.content

        except Exception:
            decision = "❌ فشل المدير في التحليل."

        return s_rep, t_rep, v_rep, decision
