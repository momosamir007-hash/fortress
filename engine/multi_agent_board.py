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
        f"المباراة: {home_team} ضد {away_team}. "
        f"نسب الفوز: أرض {probs[2]*100:.1f}% | تعادل {probs[1]*100:.1f}% | ضيف {probs[0]*100:.1f}%. "
        f"xG: {h_xg:.2f} مقابل {a_xg:.2f}. الكوتا: {odds_data}"
    )

    # ---------- ROUND 1: OPINIONS ---------- #

    experts = [
        (
        """أنت محلل إحصائي في مناظرة تلفزيونية.
قدم رأيك بالتفصيل:

- ماذا تقول الأرقام؟
- ماذا تعني xG؟
- من الأقرب للفوز رقمياً ولماذا؟

اكتب بأسلوب نقاشي واضح.""",
        context,
        "llama3.1-8b"
        ),

        (
        """أنت محلل تكتيكي عالمي في مناظرة.
قدم تحليلك التكتيكي بالتفصيل:

- كيف ستسير المباراة داخل الملعب؟
- نقاط القوة والضعف
- تأثير الأرض والجمهور
- السيناريو الأقرب

اكتب بأسلوب نقاشي واضح.""",
        context,
        "qwen-3-235b-a22b-instruct-2507"
        ),

        (
        """أنت خبير مراهنات في مناظرة.
قدم رأيك المالي:

- هل توجد قيمة مراهنة؟
- مستوى المخاطرة
- أفضل خيار منطقي

اكتب بأسلوب نقاشي.""",
        context,
        "llama3.1-8b"
        )
    ]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(self.ask_cerebras_expert, e[0], e[1], e[2]) for e in experts]
        results = [f.result() for f in futures]

    stat, tactic, finance = results

    # ---------- ROUND 2: DEBATE ---------- #

    debate_prompt = f"""
نحن في مناظرة بين ثلاثة خبراء.

رأي الإحصائي:
{stat}

رأي التكتيكي:
{tactic}

رأي المالي:
{finance}

اكتب مناظرة حقيقية حيث:
- كل خبير يعلق على رأي الآخرين
- يوضح أين يتفق وأين يختلف
- يشرح لماذا يعتقد أن رأيه أقوى

اكتب الحوار بأسلوب نقاشي واضح مع عناوين لكل خبير.
"""

    debate_text = self.ask_cerebras_expert(
        "أنت مخرج برنامج رياضي يكتب نص مناظرة واقعية.",
        debate_prompt,
        "qwen-3-235b-a22b-instruct-2507"
    )

    # ---------- FINAL DECISION ---------- #

    manager_prompt = f"""
بصفتك مدير النقاش، لخص نتيجة المناظرة التالية:

{debate_text}

وأعط القرار النهائي:

- النتيجة المتوقعة
- الخيار الآمن
- مستوى الثقة
- لماذا هذا القرار هو خلاصة المناظرة
"""

    try:
        decision = self.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": manager_prompt}],
            temperature=0.1
        ).choices[0].message.content
    except:
        decision = "فشل المدير في اتخاذ القرار."

    return stat, tactic, finance, debate_text, decision
