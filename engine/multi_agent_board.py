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

    # ---------------- استدعاء Cerebras مع نظام المحاولات والتبديل الذكي ---------------- #
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
            "max_tokens": 400
        }

        for attempt in range(2):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=45)
                if r.status_code == 200:
                    return r.json()['choices'][0]['message']['content']
                
                # التبديل التلقائي للموديل الأسرع في حال فشل الموديل الضخم أو انشغاله
                payload["model"] = "llama3.1-8b"
                r2 = requests.post(url, headers=headers, json=payload, timeout=20)
                if r2.status_code == 200:
                    return r2.json()['choices'][0]['message']['content']
            except Exception:
                continue
        return "❌ تعذر جلب التحليل من الخبير نتيجة ضغط على الخادم."

    # ---------------- إدارة اجتماع مجلس الخبراء والمناظرة المتقدمة ---------------- #
    def run_board_meeting(self, home_team, away_team, h_xg, a_xg, probs, odds_data):
        
        # تحويل البيانات إلى صيغة "صاحب الأرض" و "الفريق الضيف" لضمان ثبات اللغة العربية
        context = (
            f"تحليل المباراة: (صاحب الأرض) ضد (الفريق الضيف).\n"
            f"بيانات الخوارزمية:\n"
            f"- احتمال فوز صاحب الأرض: {probs[2]*100:.1f}%\n"
            f"- احتمال التعادل: {probs[1]*100:.1f}%\n"
            f"- احتمال فوز الفريق الضيف: {probs[0]*100:.1f}%\n"
            f"- الأهداف المتوقعة (xG): صاحب الأرض ({h_xg:.2f}) - الفريق الضيف ({a_xg:.2f})\n"
            f"- كوتا السوق المتوفرة: {odds_data}"
        )

        experts = [
            (
                "أنت محلل إحصائي رياضي فذ. قدم تحليلاً رقمياً دقيقاً بناءً على النسب المعطاة. "
                "قارن بين كفاءة 'صاحب الأرض' و'الفريق الضيف' بالعربية الفصحى فقط.",
                context, "llama3.1-8b"
            ),
            (
                "أنت محلل تكتيكي عالمي. بناءً على نسب الفوز و xG، صف سيناريو المباراة التكتيكي "
                "بين 'صاحب الأرض' و'الفريق الضيف'. ركز على الصراعات الميدانية بالعربية الفصحى.",
                context, "qwen-3-235b-a22b-instruct-2507"
            ),
            (
                "أنت خبير مالي في أسواق المراهنات. قارن بين احتمالات الآلة وكوتا السوق المرفقة. "
                "أين تكمن القيمة؟ هل هي مع 'صاحب الأرض' أم 'الفريق الضيف'؟ اكتب بالعربية.",
                context, "llama3.1-8b"
            )
        ]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.ask_cerebras_expert, e[0], e[1], e[2]) for e in experts]
            results = [f.result() for f in futures]

        stat, tactic, finance = results

        # ---------------- مرحلة المناظرة (التحليل المقابل والاشتباك الفكري) ---------------- #
        debate_prompt = f"""
        أدر مناظرة ساخنة واحترافية بين ثلاثة خبراء باللغة العربية:
        
        رأي الإحصائي: {stat}
        رأي التكتيكي: {tactic}
        رأي المالي: {finance}
        
        المطلوب: اجعل الخبراء يشتبكون في الحوار حول 'صاحب الأرض' و'الفريق الضيف'. 
        يجب أن يعلق كل منهم على فجوات رأي الآخر بأسلوب استوديو تحليلي عالمي.
        """

        debate_text = self.ask_cerebras_expert(
            "أنت مخرج استوديو رياضي بارع يكتب نصوص المناظرات العربية الاحترافية والمنظمة.",
            debate_prompt,
            "qwen-3-235b-a22b-instruct-2507"
        )

        # ---------------- القرار النهائي للمدير (التوليف وربط الأسماء الحقيقية) ---------------- #
        if not self.groq_client:
            return stat, tactic, finance, debate_text, "⚠️ فشل الوصول للمدير النهائي (Groq)."

        manager_prompt = f"""
        بصفتك مدير غرفة العمليات الاستراتيجية، حلل المناظرة التالية للمباراة بين {home_team} و {away_team}:
        
        {debate_text}
        
        أصدر قرارك النهائي بالصيغة التالية بدقة (استخدم الأسماء الحقيقية للفرق):
        1. النتيجة المتوقعة: (X-Y)
        2. التوقع المزدوج (Double Chance): (مثلاً: {home_team} أو تعادل)
        3. نسبة الثقة في التوقع: (حددها كنسبة مئوية بناءً على اتفاق الخبراء)
        4. الخيار الآمن: (أفضل رهان منطقي)
        5. الخلاصة: (اشرح لماذا، بربط تحليل الخبراء بأسماء الفرق {home_team} و {away_team}).
        """

        try:
            decision = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": manager_prompt}],
                temperature=0.1
            ).choices[0].message.content
        except Exception:
            decision = "❌ فشل المدير في معالجة بيانات المناظرة."

        return stat, tactic, finance, debate_text, decision
