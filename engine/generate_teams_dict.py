import json
import os
import sqlite3 # أو أي مكتبة تستخدمها لقاعدة بياناتك (pandas, psycopg2...)
from groq import Groq

# 1. إعداد مفتاح API (تأكد من وضعه في متغيرات البيئة أو استبداله مباشرة هنا للتجربة)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "ضع_مفتاح_جروق_هنا")
client = Groq(api_key=GROQ_API_KEY)

def get_english_teams_from_db():
    """
    هذه الدالة تسحب أسماء الفرق من قاعدة بياناتك.
    قم بتعديلها لتناسب هيكل بياناتك (سواء كانت SQLite، أو CSV عبر Pandas، أو API خارجي).
    """
    teams_list = []
    
    # مثال 1: سحب من قائمة تجريبية (قم بحذفها لاحقاً)
    teams_list = [
        "Bournemouth", "Sunderland AFC", "Arsenal", "Aston Villa", 
        "Brighton & Hove Albion", "Wolverhampton Wanderers", "Crystal Palace"
    ]
    
    # مثال 2: سحب من قاعدة بيانات SQLite (قم بإلغاء التعليق إذا كنت تستخدم قاعدة بيانات)
    # try:
    #     conn = sqlite3.connect('database.db')
    #     cursor = conn.cursor()
    #     cursor.execute("SELECT DISTINCT home_team FROM matches UNION SELECT DISTINCT away_team FROM matches")
    #     teams_list = [row[0] for row in cursor.fetchall()]
    #     conn.close()
    # except Exception as e:
    #     print(f"خطأ في الاتصال بقاعدة البيانات: {e}")

    return teams_list

def translate_teams_with_ai(teams_list):
    """استخدام Groq لتعريب الأسماء بذكاء رياضي للحفاظ على النطق الصحيح"""
    print(f"🔄 جاري ترجمة {len(teams_list)} فريق باستخدام الذكاء الاصطناعي...")
    
    # تحويل القائمة إلى نص لإرساله للنموذج
    teams_text = "\n".join(teams_list)
    
    system_prompt = """
    أنت خبير في كرة القدم ومعلق رياضي عربي. 
    سأعطيك قائمة بأسماء أندية كرة قدم باللغة الإنجليزية.
    مهمتك هي إرجاع قاموس JSON صالح (Valid JSON) يحتوي على الاسم الإنجليزي كمفتاح، والاسم المعرب الشائع استخدامه في القنوات الرياضية العربية كقيمة.
    مثال: {"Arsenal": "أرسنال", "Aston Villa": "أستون فيلا", "Sunderland AFC": "سندرلاند"}
    لا تكتب أي نص إضافي، فقط أرجع كود JSON.
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile", # نستخدم الموديل الأقوى لضمان دقة JSON
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"ترجم هذه الفرق:\n{teams_text}"}
            ],
            temperature=0.1,
            response_format={"type": "json_object"} # إجبار النموذج على إرجاع JSON فقط
        )
        
        # استخراج النص وتحويله إلى قاموس بايثون
        translated_json_str = response.choices[0].message.content
        translated_dict = json.loads(translated_json_str)
        return translated_dict
        
    except Exception as e:
        print(f"❌ حدث خطأ أثناء الترجمة: {e}")
        return {}

def save_to_json(data_dict, filename="teams_dictionary.json"):
    """حفظ القاموس في ملف JSON بتنسيق يدعم اللغة العربية"""
    if not data_dict:
        print("⚠️ لا توجد بيانات لحفظها.")
        return
        
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            # ensure_ascii=False ضرورية جداً لحفظ الحروف العربية بشكلها الطبيعي وليس كرموز
            json.dump(data_dict, f, ensure_ascii=False, indent=4)
        print(f"✅ تم حفظ القاموس بنجاح في ملف: {filename}")
    except Exception as e:
        print(f"❌ خطأ أثناء الحفظ: {e}")

if __name__ == "__main__":
    # 1. جلب الفرق الإنجليزية
    english_teams = get_english_teams_from_db()
    
    if english_teams:
        # 2. الترجمة عبر الذكاء الاصطناعي
        arabic_dict = translate_teams_with_ai(english_teams)
        
        # 3. الحفظ في ملف
        save_to_json(arabic_dict)
    else:
        print("⚠️ لم يتم العثور على فرق لترجمتها.")
