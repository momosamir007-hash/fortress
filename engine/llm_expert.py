import os
import streamlit as st
from groq import Groq

class OracleLLM:
    def __init__(self, provider="groq"):
        self.provider = provider
        if provider == "groq":
            # محاولة جلب المفتاح من أسرار Streamlit السحابية أولاً
            try:
                key = st.secrets["GROQ_API_KEY"]
            except Exception:
                # إذا لم يجده، يحاول جلبه من البيئة المحلية
                key = os.getenv("GROQ_API_KEY")
                
            if not key or key == "ضع_مفتاح_groq_هنا": 
                raise ValueError("لم يتم العثور على مفتاح Groq في إعدادات Secrets")
            
            self.client = Groq(api_key=key)
            # استخدام موديل Llama 3 السريع
            self.model_name = "llama3-8b-8192"

    def get_double_chance(self, h_team, a_team, probs):
        prompt = f'''
        أنت خبير تحليل بيانات رياضية.
        المباراة: {h_team} (الأرض) ضد {a_team} (الضيف).
        احتمالات الإحصاء: 
        أرض: {probs[2]*100:.1f}% | تعادل: {probs[1]*100:.1f}% | ضيف: {probs[0]*100:.1f}%
        
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
            # إرجاع رسالة الخطأ الحقيقية بدلاً من رسالة مبهمة
            return f"خطأ التفاصيل: {str(e)}"
