import requests
import os
import json
import re
import streamlit as st
import concurrent.futures
from groq import Groq


# ============================================================
# 1. إدارة قاموس التعريب والأسماء
# ============================================================

def load_team_dictionary(filepath="teams_dictionary.json"):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


ARABIC_TEAM_NAMES = load_team_dictionary()


def translate_team(english_name):
    clean = english_name.strip().lower()
    for key, val in ARABIC_TEAM_NAMES.items():
        if key.strip().lower() == clean:
            return val
    return english_name


def get_secret(key_name):
    try:
        if hasattr(st, 'secrets') and key_name in st.secrets:
            return st.secrets[key_name]
    except:
        pass
    return os.getenv(key_name)


# ============================================================
# 2. إعدادات النماذج
# ============================================================

PRIMARY_MODEL = "llama3.1-70b"
FALLBACK_MODEL = "llama3.1-8b"
FAST_MODEL = "llama3.1-8b"
GROQ_MODEL = "llama-3.3-70b-versatile"


# ============================================================
# 3. مُحقق المخرجات (Anti-Hallucination Guard)
# ============================================================

class OutputValidator:

    @staticmethod
    def validate_expert_output(output, provided_data):

        issues = []

        numbers_in_output = re.findall(r'\d+\.?\d*', output)
        numbers_in_data = re.findall(r'\d+\.?\d*', provided_data)
        data_numbers_set = set(numbers_in_data)

        invented_numbers = []
        for num in numbers_in_output:
            if num not in data_numbers_set and float(num) > 10:
                invented_numbers.append(num)

        if len(invented_numbers) > 3:
            issues.append(f"⚠️ أرقام مشبوهة: {invented_numbers[:5]}")

        hallucination_phrases = [
            "من المعروف أن",
            "تاريخياً",
            "في آخر 10 مباريات",
            "حسب الإحصائيات الأخيرة",
            "وفقاً للتقارير",
            "يُشير السجل إلى",
            "في الموسم الماضي",
            "معدل تسجيل",
            "نسبة استحواذ",
        ]

        for phrase in hallucination_phrases:
            if phrase in output and phrase not in provided_data:
                issues.append(f"⚠️ ادعاء بلا مصدر: '{phrase}'")

        if issues:
            disclaimer = (
                "\n\n---\n"
                "⚠️ **تنبيه الجودة:** التحليل أعلاه مبني فقط "
                "على البيانات المتاحة. أي رقم غير مذكور في "
                "الإحصائيات المُدخلة هو تقدير وليس حقيقة."
            )
            return output + disclaimer, issues

        return output, []

    @staticmethod
    def validate_prediction(prediction_text, probs, h_xg, a_xg):

        score_match = re.search(r'(\d+)\s*[-–]\s*(\d+)', prediction_text)
        if not score_match:
            return prediction_text, ["⚠️ لم يتم العثور على نتيجة واضحة"]

        pred_home = int(score_match.group(1))
        pred_away = int(score_match.group(2))

        issues = []

        if abs(pred_home - h_xg) > 2.5:
            issues.append(f"⚠️ تناقض: توقع {pred_home} أهداف للأرض لكن xG = {h_xg:.2f}")

        if abs(pred_away - a_xg) > 2.5:
            issues.append(f"⚠️ تناقض: توقع {pred_away} أهداف للضيف لكن xG = {a_xg:.2f}")

        if pred_home > pred_away and probs[2] < 0.25:
            issues.append("⚠️ توقع فوز الأرض لكن احتمالها أقل من 25%")

        if pred_away > pred_home and probs[0] < 0.25:
            issues.append("⚠️ توقع فوز الضيف لكن احتماله أقل من 25%")

        if issues:
            prediction_text += "\n\n🔍 **فحص التناسق:**\n" + "\n".join(issues)

        return prediction_text, issues


# ============================================================
# 4. بناء البرومبتات
# ============================================================

class PromptFactory:

    ANTI_HALLUCINATION_RULES = """
    ⛔ قواعد صارمة:
    1. استخدم فقط البيانات المُعطاة
    2. لا تخترع إحصائيات
    3. أجب بالعربية فقط
    """

    @staticmethod
    def build_statistician_prompt(data_context):
        return {
            "system": "أنت محلل إحصائي رياضي.",
            "user": f"البيانات:\n{data_context}"
        }


# ============================================================
# 5. اللوحة الرئيسية
# ============================================================

class MultiAgentBoard:

    def __init__(self, confidence_threshold=15):
        self.cerebras_key = get_secret("CEREBRAS_API_KEY")
        self.groq_key = get_secret("GROQ_API_KEY")
        self.validator = OutputValidator()
        self.groq_client = None

        if self.groq_key:
            self.groq_client = Groq(api_key=self.groq_key)

    def ask_cerebras_expert(self, system_prompt, user_prompt, model_id):

        url = "https://api.cerebras.ai/v1/chat/completions"

        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0
        }

        r = requests.post(url, json=payload)
        return r.json()['choices'][0]['message']['content']
