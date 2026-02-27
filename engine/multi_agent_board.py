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

    # ---------------- استدعاء Cerebras مع نظام المحاولات والتبديل ---------------- #
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
            "max_tokens": 300
        }

        for attempt in range(2):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=45)
                if r.status_code == 200:
                    return r.json()['choices'][0]['message']['content']
                
                # التبديل التلقائي للموديل الأسرع في حال فشل الموديل الضخم
                payload["model"] = "llama3.1-8b"
                r2 = requests.post(url, headers=headers, json=payload, timeout=20)
                if r2.status_code == 200:
                    return r2.json()['choices'][0]['message']['content']
            except Exception:
                continue
        return "❌ تعذر جلب التحليل من الخبير."

    # ---------------- إدارة اجتماع مجلس الخبراء والمناظرة ---------------- #
    def run_board_meeting(self, home_team, away_team, h_xg, a_xg, probs, odds_data):
        
        # تحويل البيانات إلى صيغة "صاحب الأرض" و "الفريق الضيف" لمنع تداخل اللغات
        context = (
            f"المباراة: (صاحب الأرض) ضد (الفريق الضيف).\n"
            f"البيانات الرقمية:\n"
            f"- فرصة فوز صاحب الأرض: {probs[2]*100:.1f}%\n"
            f"- فرصة التعادل: {probs[1]*100:.1f}%\n"
            f"- فرصة فوز الفريق الضيف: {probs[0]*100:.1f}%\n"
            f"- الأهداف المتوقعة (xG): صاحب الأرض ({h_xg:.2f}) - الفريق الضيف ({a_xg:.2f})\n"
            f"- كوتا السوق: {odds_data}"
        )

        experts = [
            (
                "أنت محلل إحصائي رياضي. قدم تحليلاً رقمياً دقيقاً للمباراة باللغة العربية. "
                "تحدث عن 'صاحب الأرض' و'الفريق الضيف' حصراً ولا تستخدم أسماء إنجليزية.",
                context, "llama3.1-8b"
            ),
            (
                "أنت محلل تكتيكي خبير بالدوري الإنجليزي. صف سيناريو المباراة المتوقع بين "
                "'صاحب الأرض' و'الفريق الضيف' بالعربية الفصحى. ركز على الهيمنة والضغط الميداني.",
                context, "qwen-3-235b-a22b-instruct-2507"
            ),
            (
                "أنت مستشار مالي وخبير مراهنات. قارن بين أرقام الآلة وكوتا السوق المذكورة. "
                "هل هناك قيمة استثمارية لصالح 'صاحب الأرض' أم 'الفريق الضيف'؟ اكتب بالعربية.",
                context, "llama3.1-8b"
            )
        ]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.ask_cerebras_expert, e[0], e[1], e[2]) for e in experts]
            results = [f.result() for f in futures]

        stat, tactic, finance = results

        # ---------------- مرحلة المناظرة (التحليل المقابل) ---------------- #
        debate_prompt = f"""
        أدر مناظرة تفاعلية بين ثلاثة خبراء باللغة العربية حول المباراة القادمة:
        
        رأي الإحصائي: {stat}
        رأي التكتيكي: {tactic}
        رأي المالي: {finance}
        
        المطلوب: اجعلهم يتناقشون عن 'صاحب الأرض' و'الفريق الضيف'، يتبادلون وجهات النظر، ويعلقون على نقاط القوة والضعف المذكورة بأسلوب حواري شيق.
        """

        debate_text = self.ask_cerebras_expert(
            "أنت مخرج استوديو تحليلي رياضي بارع يكتب نصوص المناظرات العربية الاحترافية.",
            debate_prompt,
            "qwen-3-235b-a22b-instruct-2507"
        )

        # ---------------- القرار النهائي للمدير (الدمج والربط بالأسماء) ---------------- #
        if not self.groq_client:
            return stat, tactic, finance, debate_text, "⚠️ مفتاح Groq غير موجود لإصدار القرار النهائي."

        manager_prompt = f"""
        بصفتك مدير غرفة العمليات، استخلص الخلاصة النهائية من هذه المناظرة للمباراة التي تجمع بين {home_team} (صاحب الأرض) و {away_team} (الفريق الضيف).
        
        نص المناظرة:
        {debate_text}
        
        المطلوب منك هو إصدار التقرير النهائي بالصيغة التالية:
        النتيجة المتوقعة: (حدد النتيجة الرقمية)
        الخيار الآمن: (حدد الرهان الأضمن)
        الخلاصة: (اشرح لماذا تم اتخاذ هذا القرار بربط التحليل بأسماء الفرق الحقيقية {home_team} و {away_team} ليعرف المستخدم لمن تعود التوقعات).
        """

        try:
            decision = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": manager_prompt}],
                temperature=0.1
            ).choices[0].message.content
        except Exception:
            decision = "❌ فشل المدير في معالجة المناظرة."

        return stat, tactic, finance, debate_text, decision
