import os
import streamlit as st
from groq import Groq

class OracleLLM:
    def __init__(self, provider="groq"):
        self.provider = provider
        if provider == "groq":
            # محاولة قراءة المفتاح من خزنة Streamlit بأمان
            try:
                key = st.secrets["GROQ_API_KEY"]
            except Exception:
                # كبديل: محاولة قراءته من بيئة التشغيل
                key = os.getenv("GROQ_API_KEY")
                
            if not key or key == "ضع_مفتاح_groq_هنا": 
                raise ValueError("مفتاح Groq غير موجود في إعدادات Secrets!")
            
            self.client = Groq(api_key=key)
            self.model_name = "llama-3.3-70b-versatile"


    def get_double_chance(self, h_team, a_team, probs):
        prompt = f'''
        أنت خبير تحليل بيانات رياضية.
        المباراة: {h_team} ضد {a_team}.
        احتمالات فوز الأرض: {probs[2]*100:.1f}%, تعادل: {probs[1]*100:.1f}%, فوز الضيف: {probs[0]*100:.1f}%
        اختر التوقع المزدوج الأكثر أماناً.
        أجب فقط بواحدة من هذه العبارات ولا شيء غيرها:
        (الأرض أو تعادل)
        (الضيف أو تعادل)
        (أرض أو ضيف)
        '''
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=15
            )
            ans = response.choices[0].message.content.strip()
            if "الأرض أو تعادل" in ans: return "الأرض أو تعادل"
            elif "الضيف أو تعادل" in ans: return "الضيف أو تعادل"
            else: return "أرض أو ضيف"
        except Exception as e:
            # هنا التعديل الأهم: إرجاع رسالة الخطأ الأصلية من سيرفر Groq
            return f"تفاصيل الخطأ: {str(e)}"
