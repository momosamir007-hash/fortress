import os
import re
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
            
            # حماية ضد هلوسة النموذج عبر البحث ضمن النص
            if "الأرض أو تعادل" in ans: return "الأرض أو تعادل"
            elif "الضيف أو تعادل" in ans: return "الضيف أو تعادل"
            elif "أرض أو ضيف" in ans: return "أرض أو ضيف"
            else: return "الخيار غير واضح"
            
        except Exception as e:
            return f"تعذر الاتصال بالخبير: {str(e)}"

    def get_exact_score(self, h_team, a_team, h_xg, a_xg, probs):
        # 1. حماية ضد أخطاء المصفوفات (ضمان تحويل القيم إلى float نقي)
        try:
            h_xg_val = float(h_xg[0]) if isinstance(h_xg, (list, tuple)) or type(h_xg).__name__ == 'ndarray' else float(h_xg)
            a_xg_val = float(a_xg[0]) if isinstance(a_xg, (list, tuple)) or type(a_xg).__name__ == 'ndarray' else float(a_xg)
        except:
            h_xg_val, a_xg_val = float(h_xg), float(a_xg)

        prompt = f'''
        أنت محلل بيانات رياضية خبير.
        المباراة: {h_team} (الأرض) ضد {a_team} (الضيف).
        الإحصائيات من خوارزمية XGBoost:
        - احتمالات النتيجة: فوز الأرض {probs[2]*100:.1f}%, تعادل {probs[1]*100:.1f}%, فوز الضيف {probs[0]*100:.1f}%
        - الأهداف المتوقعة (xG): {h_team} [{h_xg_val:.2f}] هدف، و {a_team} [{a_xg_val:.2f}] هدف.
        
        بناءً على هذه المعطيات وخبرتك الكروية، ما هي النتيجة النهائية الأكثر واقعية؟
        أجب فقط بالنتيجة بصيغة أرقام هكذا: X-Y (حيث X أهداف الأرض و Y أهداف الضيف). لا تكتب أي حرف آخر.
        '''
        try:
            response = self.client.chat.completions.create(
                model=self.model_name, messages=[{"role": "user", "content": prompt}],
                temperature=0.2, max_tokens=20
            )
            raw_ans = response.choices[0].message.content.strip()
            
            # 2. استخدام التعبير النمطي (Regex) لاستخراج النتيجة فقط كحماية ضد ثرثرة الذكاء الاصطناعي
            match = re.search(r'\d+\s*-\s*\d+', raw_ans)
            if match:
                # إرجاع النتيجة وإزالة أي مسافات زائدة بين الأرقام
                return match.group(0).replace(" ", "")
            else:
                return raw_ans # إرجاع النص كما هو إذا فشل الاستخراج
                
        except Exception:
            return "تعذر التوقع"
