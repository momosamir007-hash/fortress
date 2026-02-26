import os
import streamlit as st
from groq import Groq

class OracleLLM:
    def __init__(self, provider="groq"):
        self.provider = provider
        if provider == "groq":
            try:
                key = st.secrets["GROQ_API_KEY"]
            except Exception:
                key = os.getenv("GROQ_API_KEY")
                
            if not key or key == "ضع_مفتاح_groq_هنا": 
                raise ValueError("مفتاح Groq غير موجود!")
            
            self.client = Groq(api_key=key)
            self.model_name = "llama-3.3-70b-versatile"

    def get_double_chance(self, h_team, a_team, probs):
        prompt = f'''
        المباراة: {h_team} ضد {a_team}.
        احتمالات فوز الأرض: {probs[2]*100:.1f}%, تعادل: {probs[1]*100:.1f}%, فوز الضيف: {probs[0]*100:.1f}%
        اختر التوقع المزدوج الأكثر أماناً. أجب فقط بواحدة من: (الأرض أو تعادل), (الضيف أو تعادل), (أرض أو ضيف).
        '''
        try:
            response = self.client.chat.completions.create(
                model=self.model_name, messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=15
            )
            ans = response.choices[0].message.content.strip()
            if "الأرض أو تعادل" in ans: return "الأرض أو تعادل"
            elif "الضيف أو تعادل" in ans: return "الضيف أو تعادل"
            else: return "أرض أو ضيف"
        except Exception as e:
            return f"تفاصيل الخطأ: {str(e)}"

    def get_exact_score(self, h_team, a_team, h_xg, a_xg, probs):
        prompt = f'''
        أنت محلل بيانات رياضية خبير.
        المباراة: {h_team} (الأرض) ضد {a_team} (الضيف).
        الإحصائيات من خوارزمية XGBoost:
        - احتمالات النتيجة: فوز الأرض {probs[2]*100:.1f}%, تعادل {probs[1]*100:.1f}%, فوز الضيف {probs[0]*100:.1f}%
        - الأهداف المتوقعة (xG): {h_team} [{h_xg:.2f}] هدف، و {a_team} [{a_xg:.2f}] هدف.
        
        بناءً على هذه المعطيات وخبرتك الكروية، ما هي النتيجة النهائية الأكثر واقعية؟
        أجب فقط بالنتيجة بصيغة أرقام هكذا: X-Y (حيث X أهداف الأرض و Y أهداف الضيف). لا تكتب أي حرف آخر.
        '''
        try:
            response = self.client.chat.completions.create(
                model=self.model_name, messages=[{"role": "user", "content": prompt}],
                temperature=0.2, max_tokens=10
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return "تعذر التوقع"
