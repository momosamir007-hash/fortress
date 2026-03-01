import requests
import os
import json
import re
import streamlit as st
import concurrent.futures
from groq import Groq

# ============================================================
# 1. إدارة قاموس التعريب
# ============================================================

def load_team_dictionary(filepath="teams_dictionary.json"):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning("⚠️ ملف قاموس الفرق غير موجود.")
        return {}
    except json.JSONDecodeError:
        st.error("❌ ملف قاموس الفرق تالف.")
        return {}

ARABIC_TEAM_NAMES = load_team_dictionary()

def translate_team(english_name):
    clean_eng_name = english_name.strip().lower()
    for key, arabic_name in ARABIC_TEAM_NAMES.items():
        if key.strip().lower() == clean_eng_name:
            return arabic_name
    best_match = None
    best_len = 0
    for key, arabic_name in ARABIC_TEAM_NAMES.items():
        if key.strip().lower() in clean_eng_name and len(key) > best_len:
            best_match = arabic_name
            best_len = len(key)
    return best_match if best_match else english_name

def get_secret(key_name):
    try:
        if hasattr(st, 'secrets') and key_name in st.secrets:
            return st.secrets[key_name]
    except FileNotFoundError:
        pass
    except Exception as e:
        st.sidebar.warning(f"⚠️ خطأ في قراءة المفتاح {key_name}: {e}")
    return os.getenv(key_name)


# ============================================================
# 2. إعدادات النماذج
# ============================================================

PRIMARY_MODEL = "qwen-3-235b-a22b-instruct-2507"
FALLBACK_MODEL = "llama3.1-8b"
FAST_MODEL = "llama3.1-8b"
GROQ_MODEL = "llama-3.3-70b-versatile"


# ============================================================
# 3. مُحقق المخرجات (مانع الهلوسة)
# ============================================================

class OutputValidator:
    """فلتر لمنع الهلوسة وضمان التزام النموذج بالبيانات"""

    HALLUCINATION_PHRASES = [
        "من المعروف أن",
        "تاريخياً",
        "في آخر 10 مباريات",
        "حسب الإحصائيات الأخيرة",
        "وفقاً للتقارير",
        "يُشير السجل إلى",
        "في الموسم الماضي",
        "معدل تسجيل",
        "نسبة استحواذ",
        "according to",
        "historically",
    ]

    @staticmethod
    def validate_expert_output(output, provided_data):
        if "❌" in output or "⏰" in output or "⚠️" in output:
            return output, []
        issues = []

        # كشف ادعاءات بلا مصدر
        for phrase in OutputValidator.HALLUCINATION_PHRASES:
            if phrase in output and phrase not in provided_data:
                issues.append(f"ادعاء بلا مصدر: '{phrase}'")

        # كشف أرقام مخترعة
        nums_out = set(re.findall(r'\b\d{2,}\b', output))
        nums_data = set(re.findall(r'\b\d{2,}\b', provided_data))
        invented = nums_out - nums_data
        if len(invented) > 3:
            issues.append(f"أرقام مشبوهة: {list(invented)[:5]}")

        if issues:
            disclaimer = (
                "\n\n---\n⚠️ **تنبيه جودة:** التحليل مبني فقط "
                "على البيانات المتاحة. أي رقم إضافي هو تقدير."
            )
            return output + disclaimer, issues
        return output, []

    @staticmethod
    def validate_prediction(prediction_text, probs, h_xg, a_xg):
        issues = []
        score_match = re.search(r'(\d+)\s*[-–]\s*(\d+)', prediction_text)
        if score_match:
            pred_h = int(score_match.group(1))
            pred_a = int(score_match.group(2))
            if abs(pred_h - h_xg) > 2.5:
                issues.append(
                    f"⚠️ تناقض: توقع {pred_h} أهداف لكن xG={h_xg:.2f}"
                )
            if abs(pred_a - a_xg) > 2.5:
                issues.append(
                    f"⚠️ تناقض: توقع {pred_a} أهداف لكن xG={a_xg:.2f}"
                )
        if pred_h > pred_a and probs[2] < 0.25:
            issues.append("⚠️ توقع فوز الأرض لكن احتمالها < 25%")
        if pred_a > pred_h and probs[0] < 0.25:
            issues.append("⚠️ توقع فوز الضيف لكن احتماله < 25%")

        if issues:
            prediction_text += "\n\n🔍 **فحص التناسق:**\n"
            prediction_text += "\n".join(issues)
        return prediction_text, issues


# ============================================================
# 4. القواعد المضادة للهلوسة
# ============================================================

ANTI_HALLUCINATION_RULES = """
⛔ قواعد إلزامية:
1. استخدم فقط الأرقام المذكورة في البيانات أدناه
2. لا تخترع إحصائيات أو نتائج سابقة من عندك
3. لا تذكر معلومات عن مواسم سابقة
4. إذا لم تكن متأكداً قل "بناءً على البيانات المتاحة"
5. يُمنع استخدام: "من المعروف" أو "تاريخياً" أو "في آخر"
6. أجب بالعربية فقط
"""


# ============================================================
# 5. فئة مجلس الخبراء (مُحسّنة)
# ============================================================

class MultiAgentBoard:
    def __init__(self, confidence_threshold=15):
        self.cerebras_key = get_secret("CEREBRAS_API_KEY")
        self.groq_key = get_secret("GROQ_API_KEY")
        self.confidence_threshold = confidence_threshold
        self.validator = OutputValidator()
        self.groq_client = None
        if self.groq_key:
            try:
                self.groq_client = Groq(api_key=self.groq_key)
            except Exception as e:
                st.warning(f"⚠️ فشل تهيئة عميل Groq: {e}")

    # ─────────────────────────────────────────────
    # استدعاء Cerebras
    # ─────────────────────────────────────────────
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
            "temperature": 0.0,          # ✅ صفر = لا هلوسة
            "max_tokens": 500,            # ✅ أقصر = أقل هلوسة
            "top_p": 0.9
        }

        models_to_try = [model_id, FALLBACK_MODEL]
        if model_id == FALLBACK_MODEL:
            models_to_try = [FALLBACK_MODEL]

        for current_model in models_to_try:
            try:
                payload = {**base_payload, "model": current_model}
                timeout = 45 if current_model != FALLBACK_MODEL else 20
                r = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                    stream=False
                )
                if r.status_code == 200:
                    data = r.json()
                    if 'choices' in data and len(data['choices']) > 0:
                        return data['choices'][0]['message']['content']
                elif r.status_code == 429:
                    continue
            except (requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError):
                continue
            except Exception as e:
                st.warning(f"⚠️ خطأ مع {current_model}: {e}")
                continue

        return "❌ تعذر جلب التحليل. جميع النماذج مشغولة حالياً."

    # ─────────────────────────────────────────────
    # بناء السياق (مع حراسة ضد الهلوسة)
    # ─────────────────────────────────────────────
    @staticmethod
    def _build_context(home_team, away_team, h_xg, a_xg, probs, odds_data):
        if probs[2] > probs[0]:
            h2h_note = f"{home_team} يملك الأفضلية الإحصائية."
        elif probs[0] > probs[2]:
            h2h_note = f"{away_team} يملك الأفضلية الإحصائية."
        else:
            h2h_note = "لا توجد أفضلية واضحة."

        odds_text = "غير متوفرة حالياً"
        if odds_data:
            odds_text = (
                f"فوز {home_team}: {odds_data.get('home', 'N/A')} | "
                f"تعادل: {odds_data.get('draw', 'N/A')} | "
                f"فوز {away_team}: {odds_data.get('away', 'N/A')}"
            )

        context = (
            f"═══ بيانات المباراة (المصدر الوحيد) ═══\n"
            f"المباراة: {home_team} (أرض) ضد {away_team} (ضيف)\n"
            f"─────────────────────────\n"
            f"• فرصة فوز {home_team}: {probs[2]*100:.1f}%\n"
            f"• فرصة التعادل: {probs[1]*100:.1f}%\n"
            f"• فرصة فوز {away_team}: {probs[0]*100:.1f}%\n"
            f"• الأهداف المتوقعة: {home_team} ({h_xg:.2f}) | "
            f"{away_team} ({a_xg:.2f})\n"
            f"• مجموع الأهداف المتوقع: {h_xg+a_xg:.2f}\n"
            f"• كوتا السوق: {odds_text}\n"
            f"• الأفضلية: {h2h_note}\n"
            f"═══ نهاية البيانات ═══\n"
            f"⛔ أي معلومة غير مذكورة أعلاه تُعتبر غير متاحة."
        )
        return context

    # ─────────────────────────────────────────────
    # تعريف الخبراء (برومبتات مُحصّنة)
    # ─────────────────────────────────────────────
    @staticmethod
    def _define_experts(context):
        return [
            {
                "system": (
                    "أنت محلل بيانات رياضي صارم.\n"
                    f"{ANTI_HALLUCINATION_RULES}\n"
                    "المطلوب:\n"
                    "📊 تحليل الاحتمالات: [حلل النسب المُعطاة]\n"
                    "⚽ تحليل الأهداف: [حلل xG المُعطى]\n"
                    "📈 الخلاصة: [استنتاج من الأرقام فقط]"
                ),
                "user": context,
                "model": FAST_MODEL
            },
            {
                "system": (
                    "أنت محلل تكتيكي ونفسي لكرة القدم.\n"
                    f"{ANTI_HALLUCINATION_RULES}\n"
                    "قاعدة إضافية: يُمنع كتابة أي رقم أو نسبة مئوية.\n"
                    "المطلوب:\n"
                    "🎯 السيناريو التكتيكي: [بناءً على فارق الأهداف المتوقعة]\n"
                    "🧠 العامل النفسي: [بناءً على الأفضلية المذكورة]\n"
                    "📋 الخلاصة: [توقع نوعي فقط]"
                ),
                "user": context,
                "model": PRIMARY_MODEL
            },
            {
                "system": (
                    "أنت خبير تقييم مخاطر مالية رياضية.\n"
                    f"{ANTI_HALLUCINATION_RULES}\n"
                    "المطلوب:\n"
                    "💰 تحليل القيمة: [قارن الكوتا بالاحتمالات]\n"
                    "⚖️ مستوى المخاطرة: [منخفض/متوسط/عالٍ]\n"
                    "🎯 التوصية: [توصية واحدة واضحة]"
                ),
                "user": context,
                "model": FAST_MODEL
            }
        ]

    # ─────────────────────────────────────────────
    # ⭐ الدالة الرئيسية - نفس التوقيع القديم ⭐
    # ─────────────────────────────────────────────
    def run_board_meeting(self, home_team_eng, away_team_eng,
                          h_xg, a_xg, probs, odds_data):
        """نفس التوقيع القديم بالضبط - 6 معاملات"""
        home_team = translate_team(home_team_eng)
        away_team = translate_team(away_team_eng)

        context = self._build_context(
            home_team, away_team, h_xg, a_xg, probs, odds_data
        )
        experts = self._define_experts(context)
        error_fallback = "⚠️ تعذر الحصول على تحليل هذا الخبير."

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(
                    self.ask_cerebras_expert,
                    e["system"],
                    e["user"],
                    e["model"]
                ): i for i, e in enumerate(experts)
            }
            results = [error_fallback] * 3
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    raw = future.result(timeout=60)
                    # ✅ تحقق من المخرجات
                    validated, issues = self.validator.validate_expert_output(
                        raw, context
                    )
                    results[idx] = validated
                except concurrent.futures.TimeoutError:
                    results[idx] = "⏰ انتهت مهلة الخبير."
                except Exception as e:
                    results[idx] = f"❌ خطأ: {str(e)[:100]}"

        stat, tactic, finance = results

        # مناظرة مُقيّدة بالبيانات
        debate_prompt = (
            f"بناءً على الأرقام التالية فقط:\n{context}\n\n"
            f"أدر مناظرة فنية قصيرة بالعربية بين:\n"
            f"- الإحصائي: {stat}\n"
            f"- التكتيكي: {tactic}\n"
            f"- المالي: {finance}\n\n"
            f"⛔ لا تضف أي أرقام أو معلومات من خارج "
            f"البيانات المذكورة أعلاه.\n"
            f"ركز على نقاط الاتفاق والاختلاف فقط."
        )
        debate_text = self.ask_cerebras_expert(
            "أنت مخرج استوديو تحليلي. لا تخترع معلومات.",
            debate_prompt,
            PRIMARY_MODEL
        )

        decision = self._get_manager_decision(
            home_team, away_team, debate_text, context, probs, h_xg, a_xg
        )

        return stat, tactic, finance, debate_text, decision

    # ─────────────────────────────────────────────
    # قرار المدير النهائي (مع تحقق)
    # ─────────────────────────────────────────────
    def _get_manager_decision(self, home_team, away_team, debate_text,
                              context, probs, h_xg, a_xg):
        if not self.groq_client:
            return ("⚠️ مفتاح Groq مفقود. أضفه في الإعدادات "
                    "لتفعيل قرار المدير النهائي.")

        manager_prompt = (
            f"بصفتك مدير غرفة العمليات، حلل ما يلي "
            f"لمباراة {home_team} ضد {away_team}:\n\n"
            f"📊 البيانات الأصلية:\n{context}\n\n"
            f"📋 المناظرة:\n{debate_text}\n\n"
            f"{ANTI_HALLUCINATION_RULES}\n\n"
            f"أصدر قرارك بهذا التنسيق بالعربية:\n"
            f"النتيجة المتوقعة: [النتيجة]\n"
            f"التوقع المزدوج: [التوقع]\n"
            f"نسبة الثقة: [النسبة]% (بين 40-85% فقط)\n"
            f"الخيار الآمن: [الخيار المالي]\n"
            f"الخلاصة: [سطرين من البيانات فقط]"
        )

        try:
            response = self.groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": manager_prompt}],
                temperature=0.0,
                max_tokens=400
            )
            raw_decision = response.choices[0].message.content
            # ✅ تحقق نهائي من التناسق
            validated, _ = self.validator.validate_prediction(
                raw_decision, probs, h_xg, a_xg
            )
            return validated
        except Exception as e:
            return f"❌ فشل المدير: {str(e)[:150]}"
