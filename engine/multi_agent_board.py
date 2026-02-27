import requests
import os
import json
import streamlit as st
import concurrent.futures
from groq import Groq

# ============================================================
# 1. إدارة قاموس التعريب
# ============================================================
def load_team_dictionary(filepath="teams_dictionary.json"):
    """تحميل قاموس أسماء الفرق من ملف JSON الخارجي"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning("⚠️ ملف قاموس الفرق غير موجود. سيتم استخدام الأسماء الإنجليزية.")
        return {}
    except json.JSONDecodeError:
        st.error("❌ ملف قاموس الفرق تالف. تحقق من صيغة JSON.")
        return {}

ARABIC_TEAM_NAMES = load_team_dictionary()

def translate_team(english_name):
    """ البحث عن اسم الفريق وتعريبه بدقة. """
    # تنظيف النص لتفادي أخطاء المسافات
    clean_eng_name = english_name.strip().lower()
    
    # أولاً: تطابق كامل (الأدق)
    for key, arabic_name in ARABIC_TEAM_NAMES.items():
        if key.strip().lower() == clean_eng_name:
            return arabic_name

    # ثانياً: تطابق جزئي (احتياطي) مع أولوية للاسم الأطول
    best_match = None
    best_len = 0
    for key, arabic_name in ARABIC_TEAM_NAMES.items():
        if key.strip().lower() in clean_eng_name and len(key) > best_len:
            best_match = arabic_name
            best_len = len(key)
            
    return best_match if best_match else english_name

# ============================================================
# 2. إدارة المفاتيح السرية
# ============================================================
def get_secret(key_name):
    """جلب المفتاح من أسرار Streamlit أو متغيرات البيئة"""
    try:
        if hasattr(st, 'secrets') and key_name in st.secrets:
            return st.secrets[key_name]
    except FileNotFoundError:
        pass
    except Exception as e:
        st.sidebar.warning(f"⚠️ خطأ في قراءة المفتاح {key_name}: {e}")
    return os.getenv(key_name)

# ============================================================
# 3. فئة مجلس الخبراء (غرفة العمليات)
# ============================================================
PRIMARY_MODEL = "qwen-3-235b-a22b-instruct-2507"
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
        if not self.cerebras_key:
            return "❌ مفتاح Cerebras مفقود. أضفه في الإعدادات."
            
        url = "https://api.cerebras.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cerebras_key}",
            "Content-Type": "application/json"
        }
        base_payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 800
        }

        models_to_try = [model_id, FALLBACK_MODEL]
        if model_id == FALLBACK_MODEL:
            models_to_try = [FALLBACK_MODEL]

        for current_model in models_to_try:
            try:
                payload = {**base_payload, "model": current_model}
                timeout = 45 if current_model != FALLBACK_MODEL else 20
                
                # تم إضافة stream=False لضمان استقرار الاتصال
                r = requests.post(
                    url, headers=headers, json=payload, timeout=timeout, stream=False
                )
                
                if r.status_code == 200:
                    data = r.json()
                    if 'choices' in data and len(data['choices']) > 0:
                        return data['choices'][0]['message']['content']
                elif r.status_code == 429:
                    continue 
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                continue
            except Exception as e:
                st.warning(f"⚠️ خطأ غير متوقع مع {current_model}: {e}")
                continue
                
        return "❌ تعذر جلب التحليل من الخبير. جميع النماذج مشغولة حالياً."

    @staticmethod
    def _build_context(home_team, away_team, h_xg, a_xg, probs, odds_data):
        if probs[2] > probs[0]:
            h2h_note = f"{home_team} يملك الأفضلية الإحصائية بناءً على البيانات."
        elif probs[0] > probs[2]:
            h2h_note = f"{away_team} يملك الأفضلية الإحصائية بناءً على البيانات."
        else:
            h2h_note = "لا توجد أفضلية واضحة لأي فريق بناءً على البيانات."
            
        odds_text = "غير متوفرة حالياً"
        if odds_data:
            odds_text = (
                f"فوز {home_team}: {odds_data.get('home', 'N/A')} | "
                f"تعادل: {odds_data.get('draw', 'N/A')} | "
                f"فوز {away_team}: {odds_data.get('away', 'N/A')}"
            )
            
        context = (
            f"المباراة: {home_team} (صاحب الأرض) ضد {away_team} (الضيف).\n"
            f"البيانات المعتمدة:\n"
            f"- فرصة فوز {home_team}: {probs[2] * 100:.1f}%\n"
            f"- فرصة التعادل: {probs[1] * 100:.1f}%\n"
            f"- فرصة فوز {away_team}: {probs[0] * 100:.1f}%\n"
            f"- الأهداف المتوقعة: {home_team} ({h_xg:.2f}) | {away_team} ({a_xg:.2f})\n"
            f"- كوتا السوق: {odds_text}\n"
            f"- الأفضلية: {h2h_note}"
        )
        return context

    @staticmethod
    def _define_experts(context):
        return [
            {
                "system": (
                    "أنت محلل بيانات رياضي صارم. دورك سرد قراءة للبيانات المتوفرة فقط.\n"
                    "قواعد صارمة:\n"
                    "1. لا تخترع أي أرقام من خارج السياق.\n"
                    "2. اكتب فقرة قصيرة ومباشرة بالعربية الفصحى.\n"
                    "3. استخدم أسماء الفرق العربية فقط."
                ),
                "user": context,
                "model": FAST_MODEL
            },
            {
                "system": (
                    "أنت محلل تكتيكي ونفسي خبير في كرة القدم. دورك قراءة سيناريو المباراة.\n"
                    "قواعد صارمة جداً:\n"
                    "1. يُمنع كتابة أي رقم أو نسبة مئوية.\n"
                    "2. ركز على: الضغط العالي، الاستحواذ، العامل النفسي، عقدة المواجهات.\n"
                    "3. اكتب بالعربية الفصحى فقط."
                ),
                "user": context,
                "model": PRIMARY_MODEL
            },
            {
                "system": (
                    "أنت خبير مراهنات وتقييم مخاطر مالية. دورك استخراج 'القيمة' عبر مقارنة "
                    "احتمالات الخوارزمية مع كوتا السوق.\n"
                    "قواعد صارمة:\n"
                    "1. حدد بوضوح أين تكمن القيمة الآمنة.\n"
                    "2. تجنب العاطفة وركز على العائد مقابل الخطر.\n"
                    "3. اكتب بالعربية فقط."
                ),
                "user": context,
                "model": FAST_MODEL
            }
        ]

    def run_board_meeting(self, home_team_eng, away_team_eng, h_xg, a_xg, probs, odds_data):
        home_team = translate_team(home_team_eng)
        away_team = translate_team(away_team_eng)
        context = self._build_context(home_team, away_team, h_xg, a_xg, probs, odds_data)
        experts = self._define_experts(context)

        # تم إزالة علامة التعليق (Uncomment) لتعريف المتغير وتفادي خطأ NameError
        error_fallback = "⚠️ تعذر الحصول على تحليل هذا الخبير."
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(
                    self.ask_cerebras_expert, e["system"], e["user"], e["model"]
                ): i for i, e in enumerate(experts)
            }
            results = [error_fallback] * 3
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result(timeout=60)
                except concurrent.futures.TimeoutError:
                    results[idx] = "⏰ انتهت مهلة الخبير."
                except Exception as e:
                    results[idx] = f"❌ خطأ: {str(e)[:100]}"
                    
        stat, tactic, finance = results

        # تم تمرير السياق للمخرج لكي لا يقوم بتأليف معلومات خاطئة
        debate_prompt = (
            f"بناءً على الأرقام التالية:\n{context}\n\n"
            f"أدر مناظرة فنية قصيرة وحادة بالعربية بين الخبراء:\n"
            f"- الإحصائي: {stat}\n"
            f"- التكتيكي: {tactic}\n"
            f"- المالي: {finance}\n\n"
            f"المطلوب: صغ نقاشاً احترافياً يركز على الصدام بين 'لغة الأرقام' و'الواقع التكتيكي والنفسي'. "
            f"اكتب بالعربية فقط بأسلوب سهل القراءة."
        )
        
        debate_text = self.ask_cerebras_expert(
            "أنت مخرج استوديو تحليلي. تصيغ مناظرات احترافية خالية من الحشو.",
            debate_prompt,
            PRIMARY_MODEL
        )

        decision = self._get_manager_decision(home_team, away_team, debate_text)
        return stat, tactic, finance, debate_text, decision

    def _get_manager_decision(self, home_team, away_team, debate_text):
        if not self.groq_client:
            return "⚠️ مفتاح Groq مفقود. أضفه في الإعدادات لتفعيل قرار المدير النهائي."
            
        # إضافة تعليمات توجيهية أقوى للمدير للحفاظ على المنطقية
        manager_prompt = (
            f"بصفتك مدير غرفة العمليات الصارم، حلل المناظرة التالية لمباراة {home_team} ضد {away_team}:\n"
            f"{debate_text}\n\n"
            f"توجيه هام: إذا كان هناك تضارب بين المحللين، انحز للبيانات المضمونة وقم بخفض نسبة الثقة.\n\n"
            f"أصدر قرارك النهائي بهذا التنسيق بالعربية (كل عنصر في سطر جديد):\n"
            f"النتيجة المتوقعة: [النتيجة]\n"
            f"التوقع المزدوج: [التوقع]\n"
            f"نسبة الثقة: [النسبة]%\n"
            f"الخيار الآمن: [الخيار المالي]\n"
            f"الخلاصة: [سطرين يشرحان القرار النهائي]"
        )
        try:
            response = self.groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": manager_prompt}],
                temperature=0.1,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ فشل المدير في معالجة البيانات: {str(e)[:150]}"
