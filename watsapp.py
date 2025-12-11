import cv2            # مكتبة OpenCV: لقراءة ومعالجة الصور (Image Processing).
import pytesseract    # واجهة Python لـ OCR: لاستخراج النص من الصورة.
import pandas as pd   # مكتبة Pandas: لتنظيم البيانات المستخلصة في جداول (DataFrames).
import os             # للتعامل مع نظام الملفات (قراءة مجلد الصور).
import re             # التعبيرات النمطية (RegEx): لتنظيف النصوص من الرموز والضوضاء.
from fuzzywuzzy import fuzz    # خوارزمية قياس التشابه (Levenshtein Distance) - جزء من AI.
from fuzzywuzzy import process # أداة المطابقة الغامضة (Fuzzy Matching): لتحديد أقرب خيار صحيح.

# ---------------- إعدادات المشروع (المتغيرات الأساسية) ----------------

# قائمة بخيارات التصويت الصحيحة الموجودة في الصور.
VOTE_OPTIONS = ["zero","one", "two", "three","four"]

# اسم المجلد الذي يحتوي على لقطات الشاشة.
IMAGE_FOLDER = 'images'

# -------------------------------------------------------------------

def correct_vote_option(extracted_text, valid_options, threshold=75):
    """
    دالة نموذج الذكاء الاصطناعي (AI Model): تُصحح النص المشوّه إلى أقرب خيار تصويت صحيح
    باستخدام المطابقة الغامضة (Fuzzy Matching).
    """
    if not extracted_text:
        return None
    
    # تبحث عن أفضل تطابق بين النص المُستخرج وقائمة الخيارات الصالحة.
    best_match = process.extractOne(extracted_text, valid_options)
    
    # إذا كانت درجة التشابه أعلى من العتبة (75%)، يتم اعتماد النص الصحيح (قرار آلي).
    if best_match and best_match[1] >= threshold:
        return best_match[0]
    else:
        # وإلا، يتم إرجاع النص كما هو.
        return extracted_text


def is_timestamp(text):
    """
    دالة تصفية: تتجاهل الأسطر التي تحتوي على توقيت أو تاريخ (ليست أسماء مصوتين).
    """
    text = text.lower()
    # أنماط شائعة للوقت والتاريخ في واتساب
    if any(x in text for x in ['am', 'pm', 'yesterday', 'today', 'at']):
        return True
    # البحث عن تنسيق الساعة (رقمين:رقمين)
    if re.search(r'\d{1,2}:\d{2}', text):
        return True
    return False

def process_images():
    all_data = [] 
    seen_names = set() 
    current_vote_category = None

    # التحقق من وجود مجلد الصور
    if not os.path.exists(IMAGE_FOLDER):
        print(f"❌ خطأ: لم يتم العثور على مجلد الصور '{IMAGE_FOLDER}'. يرجى وضعه في نفس مكان ملف البايثون.")
        return
        
    images = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not images:
        print(f"❌ لم يتم العثور على أي صور في مجلد '{IMAGE_FOLDER}'. يرجى وضع الصور وتشغيل البرنامج مرة أخرى.")
        return

    print(f"✅ تم العثور على {len(images)} صورة. جاري معالجة البيانات...")

    for img_name in images:
        img_path = os.path.join(IMAGE_FOLDER, img_name)
        
        # 1. معالجة الصورة الرقمية (Image Pre-processing) - مفاهيم Image Processing
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # التحويل إلى Grayscale (المحاضرة 2).
        
        # تطبيق العتبة (Thresholding) لإنشاء صورة ثنائية (Binary Image) لفصل النص عن الخلفية (المحاضرة 4).
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV) 

        # 2. استخراج النصوص (OCR) باستخدام Tesseract
        try:
            # استخلاص النص من الصورة الثنائية
            text = pytesseract.image_to_string(thresh, lang='eng') 
        except pytesseract.TesseractNotFoundError:
            print("\n🚨🚨 خطأ: لم يتمكن بايثون من العثور على برنامج Tesseract.")
            return

        lines = text.split('\n')

        # 3. تحليل السطور وتصنيف المصوتين
        for line in lines:
            clean_line = line.strip()
            
            if len(clean_line) < 2:
                continue

            # التنظيف المبكر: إزالة الرموز الشائعة قبل محاولة التصنيف
            clean_line = re.sub(r'[()\-\[\]]', '',clean_line).strip()

            # ---------------- مرحلة اتخاذ القرار والتصنيف (AI Classification) ----------------
is_header = False
            
            # محاولة تصحيح السطر إلى أقرب خيار تصويت صحيح (قرار نموذج AI)
            corrected_option = correct_vote_option(clean_line, VOTE_OPTIONS, threshold=75)
            
            # إذا كان التصحيح قوياً (فوق 75%): يتم اعتباره عنوان تصويت جديد (بداية قائمة جديدة).
            if corrected_option in VOTE_OPTIONS:
                current_vote_category = corrected_option
                is_header = True
            
            # إذا كان التصحيح ضعيفاً (55% - 75%): يُعتبر عنواناً جديداً أيضاً لفرض التغيير (حل لمشكلة القراءة الضعيفة).
            elif process.extractOne(clean_line, VOTE_OPTIONS)[1] > 55:
                current_vote_category = process.extractOne(clean_line, VOTE_OPTIONS)[0]
                is_header = True

            if is_header:
                continue
            # ---------------- نهاية مرحلة التصنيف ----------------

            # 4. تصفية الأسماء وتخزينها
            if current_vote_category and not is_timestamp(clean_line):
                
                # تصفية كلمات النظام (مثل 'vote', 'you')
                if any(word in clean_line.lower() for word in ['vote', 'member', 'read', 'you']): 
                    continue
                
                # فلتر تنظيف الأسماء القوي: إزالة الرموز غير الأبجدية والرقمية (RegEx Filter).
                cleaned_name = re.sub(r'[^a-zA-Z0-9\s]+', '', clean_line).strip() 

                # معالجة أخطاء الـ OCR الشائعة: حذف الحروف العشوائية المضافة في بداية الاسم (مثل 'ah' أو 'Ee').
                name_parts = cleaned_name.split()
                if name_parts and len(name_parts[0]) <= 2 and name_parts[0].lower() not in ['al', 'ibn', 'ab']:
                    cleaned_name = " ".join(name_parts[1:])
                
                # شروط تجاهل الضوضاء والأسماء القصيرة جداً
                if len(cleaned_name) < 3: 
                    continue
                if sum(c.isalpha() for c in cleaned_name) < 3:
                    continue
                
                # تخزين البيانات في قائمة (مع تجنب تكرار الأسماء)
                unique_key = f"{cleaned_name}_{current_vote_category}"
                
                if unique_key not in seen_names:
                    all_data.append({
                        "Name": cleaned_name, 
                        "Vote": current_vote_category
                    })
                    seen_names.add(unique_key)

    # 5. الحفظ والإخراج
    if all_data:
        df = pd.DataFrame(all_data)
        output_file = "voting_results6.xlsx"
        df.to_excel(output_file, index=False)
        print(f"\n✨ اكتمل العمل! تم حفظ الملف في: '{output_file}'")
        print("\nنموذج للنتائج:")
        print(df.head())
    else:
        print("❌ لم يتم العثور على أي بيانات أسماء قابلة للاستخراج. يرجى مراجعة الصور.")

# تشغيل الدالة الرئيسية للمشروع
process_images()
