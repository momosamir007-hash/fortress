import requests
import os
import json
import streamlit as st
import concurrent.futures
from groq import Groq

# ---------------- 1. إدارة قاموس التعريب ---------------- #
def load_team_dictionary(filepath="teams_dictionary.json"):
    """تحميل قاموس أسماء الفرق من ملف JSON الخارجي"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {} # إرجاع قاموس فارغ إذا لم يتم إنشاء الملف بعد

# تحميل القاموس مرة واحدة عند استدعاء الملف
ARABIC_TEAM_NAMES = load_team_dictionary()

def translate_team(english_name):
    """البحث عن اسم الفريق وتعريبه، وإرجاع الاسم الأصلي إذا لم يتوفر"""
    for key, arabic_name in ARABIC_TEAM_NAMES.items():
        if key.lower() in english_name.lower():
            return arabic_name
    return english_name

# ---------------- 2. إدارة المفاتيح السرية ---------------- #
def get_secret(key_name):
    try:
        if key_name in st.secrets:
            return st.secrets[key_name]
    except:
        pass
    return os.getenv(key_name)

# ---------------- 3. فئة مجلس الخبراء (غرفة العمليات) ---------------- #
class MultiAgentBoard:
    def __init__(self):
        self.cerebras_key = get_secret("CEREBRAS_API_KEY")
        self.groq_key = get_secret("GROQ_API_KEY")
        self.groq_client = None
        if self.groq_key:
            self.groq_client = Groq(api_key=self.groq_key)

    def ask_cerebras_expert(self, system_prompt, user_prompt, model_id):
        """استدعاء نماذج Cerebras مع نظام التبديل الذكي في حال الفشل"""
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
            "temperature": 0.1, # حرارة منخفضة جداً لمنع الهلوسة والالتزام بالبيانات
            "max_tokens": 800
        }

        # محاولة الاتصال مرتين، مع التبديل لموديل أخف وأسرع في حال فشل الموديل الأساسي
        for attempt in range(2):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=45)
                if r.status_code == 200:
                    return r.json()['choices'][0]['message']['content']
                
                payload["model"] = "llama3.1-8b"
                r2 = requests.post(url, headers=headers, json=payload, timeout=20)
                if r2.status_code == 200:
                    return r2.json()['choices'][0]['message']['content']
            except Exception:
                continue
        return "❌ تعذر جلب التحليل من الخبير نتيجة ضغط على الخادم."

    def run_board_meeting(self, home_team_eng, away_team_eng, h_xg, a_xg, probs, odds_data):
        
        # تعريب أسماء الفرق لضمان عدم تداخل اللغتين في واجهة المستخدم
        home_team = translate_team(home_team_eng)
        away_team = translate_team(away_team_eng)

        # تجهيز السياق الموحد باللغة العربية فقط
        context = (
            f"المباراة: {home_team} (صاحب الأرض) ضد {away_team} (الفريق الضيف).\n"
            f"البيانات المعتمدة للمباراة:\n"
            f"- فرصة فوز {home_team}: {probs[2]*100:.1f}%\n"
            f"- فرصة التعادل: {probs[1]*100:.1f}%\n"
            f"- فرصة فوز {away_team}: {probs[0]*100:.1f}%\n"
            f"- الأهداف المتوقعة: {home_team} ({h_xg:.2f}) | {away_team} ({a_xg:.2f})\n"
            f"- كوتا السوق المتوفرة: {odds_data}\n"
            f"- ملاحظة تكتيكية وتاريخية: {home_team} يمتلك أفضلية في المواجهات المباشرة (H2H) عبر السنوات."
        )

        # ---------------- 4. هندسة الأوامر الصارمة للخبراء ---------------- #
        experts = [
            # 1. المحلل الإحصائي (يعتمد على الأرقام فقط)
            (
                "أنت محلل بيانات رياضي صارم. دورك هو سرد قراءة للبيانات المتوفرة فقط دون أي تنظير تكتيكي. "
                "قواعد صارمة:\n"
                "1. لا تخترع أي أرقام من خارج السياق.\n"
                "2. اكتب فقرة قصيرة ومباشرة باللغة العربية الفصحى.\n"
                "3. استخدم أسماء الفرق العربية المرفقة ولا تستخدم الإنجليزية إطلاقاً.",
                context, "llama3.1-8b"
            ),
            # 2. المحلل التكتيكي والنفسي (ممنوع من استخدام الأرقام نهائياً)
            (
                "أنت محلل تكتيكي ونفسي خبير في كرة القدم. دورك هو قراءة سيناريو المباراة استناداً للتفوق التاريخي وأفضلية الملعب. "
                "قواعد صارمة جداً:\n"
                "1. يُمنع منعاً باتاً كتابة أي رقم، نسبة مئوية، أو إحصائية في ردك (حتى الأصفار).\n"
                "2. ركز على الضغط العالي، الاستحواذ، العامل النفسي، وعقدة المواجهات المباشرة.\n"
                "3. اكتب باللغة العربية الفصحى فقط وبدون أي كلمات إنجليزية.",
                context, "qwen-3-235b-a22b-instruct-2507"
            ),
            # 3. الخبير المالي والمراهنات (يبحث عن القيمة)
            (
                "أنت خبير مراهنات وتقييم مخاطر مالية. دورك استخراج 'القيمة' عبر مقارنة احتمالات الخوارزمية مع كوتا السوق. "
                "قواعد صارمة:\n"
                "1. حدد بوضوح أين تكمن القيمة الآمنة للمستثمر.\n"
                "2. تجنب العاطفة وركز على العائد المالي مقابل الخطر الرياضي.\n"
                "3. اكتب باللغة العربية فقط.",
                context, "llama3.1-8b"
            )
        ]

        # تشغيل المحللين الثلاثة بالتوازي لتسريع الاستجابة
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.ask_cerebras_expert, e[0], e[1], e[2]) for e in experts]
            results = [f.result() for f in futures]

        stat, tactic, finance = results

        # ---------------- 5. إدارة المناظرة (الاشتباك الفكري) ---------------- #
        debate_prompt = f"""
        أدر مناظرة فنية قصيرة وحادة باللغة العربية بين الخبراء حول مباراة {home_team} و {away_team}:
        - الإحصائي: {stat}
        - التكتيكي: {tactic}
        - المالي: {finance}
        
        المطلوب: صغ نقاشاً احترافياً يركز على الصدام بين 'لغة الأرقام' و 'الواقع النفسي والتكتيكي'. 
        هل ستكسر 'المفاجأة' الأرقام؟ أم أن '{home_team}' سيكرر سيطرته التاريخية؟
        تأكد من أن النص كامل باللغة العربية ومكتوب بأسلوب يسهل قراءته من اليمين لليسار.
        """

        debate_text = self.ask_cerebras_expert(
            "أنت مخرج استوديو تحليلي. تصيغ مناظرات حماسية واحترافية خالية من الحشو المفرط.",
            debate_prompt,
            "qwen-3-235b-a22b-instruct-2507"
        )

        # ---------------- 6. القرار النهائي للمدير (Groq) ---------------- #
        if not self.groq_client:
            return stat, tactic, finance, debate_text, "⚠️ فشل الوصول للمدير النهائي (Groq)."

        manager_prompt = f"""
        بصفتك مدير غرفة العمليات، حلل المناظرة التالية لمباراة {home_team} ضد {away_team}:
        {debate_text}
        
        أصدر قرارك الاستثماري والرياضي النهائي حصرياً بهذا التنسيق (باللغة العربية فقط وبدون أي إضافات):
        النتيجة المتوقعة: [أدخل النتيجة هنا]
        التوقع المزدوج: [أدخل التوقع هنا]
        نسبة الثقة: [أدخل النسبة هنا]%
        الخيار الآمن: [أدخل الخيار المالي هنا]
        الخلاصة: [سطرين يشرحان القرار النهائي بناءً على تلاقي التحليل المالي والتكتيكي]
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
