# 1. استيراد المكتبات اللازمة
from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import json

# 2. إنشاء تطبيق Flask
app = Flask(__name__)

# --- متغيرات عالمية لتحميل النماذج مرة واحدة فقط ---
model = None
model_columns = None
neighborhoods = []

# 3. دالة لتحميل القطع الأثرية (Artifacts)
def load_artifacts():
    global model, model_columns, neighborhoods
    try:
        model = joblib.load('random_forest_house_pricer.joblib')
        model_columns = joblib.load('model_columns.joblib')
        with open('categorical_values.json', 'r') as f:
            categorical_values = json.load(f)
        
        neighborhoods = categorical_values.get('Neighborhood', [])
        
        print("✅✅✅ Model and artifacts loaded successfully! ✅✅✅")

    except FileNotFoundError as e:
        print(f"❌❌❌ Error loading artifacts: {e} ❌❌❌")
        # تأكد من أن الملفات موجودة في نفس المجلد

# 4. تحديد المسار الرئيسي للتطبيق (الصفحة الرئيسية)
@app.route('/')
def home():
    # هذه هي الدالة التي تسببت في الخطأ سابقًا
    # الآن هي ترسل قاموسًا فارغًا لـ form_data
    print("--- Accessing Home Page ---")
    return render_template('index.html', 
                           neighborhoods=neighborhoods, 
                           form_data={}) # هذا هو السطر الحاسم

# 5. تحديد مسار التنبؤ
@app.route('/predict', methods=['POST'])
def predict():
    print("--- Received a prediction request ---")
    if not model or model_columns is None:
        return render_template('index.html', 
                               prediction_text='Error: Model is not loaded.', 
                               neighborhoods=neighborhoods,
                               form_data={})

    try:
        # أ: الحصول على البيانات من النموذج
        user_input = request.form.to_dict()
        print(f"User input received: {user_input}")

        # ب: إنشاء DataFrame فارغ بنفس أعمدة النموذج
        query_df = pd.DataFrame(columns=model_columns, index=[0])
        query_df.fillna(0, inplace=True)

        # ج: ملء الـ DataFrame بمدخلات المستخدم
        for key, value in user_input.items():
            if key == 'Neighborhood':
                column_name = f'Neighborhood_{value}'
                if column_name in query_df.columns:
                    query_df.loc[0, column_name] = 1
            elif key in query_df.columns:
                try:
                    query_df.loc[0, key] = float(value)
                except (ValueError, TypeError):
                    query_df.loc[0, key] = 0
        
        # د: عمل التنبؤ
        prediction = model.predict(query_df)
        print(f"Prediction successful: {prediction[0]}")
        
        # هـ: إرجاع النتيجة إلى نفس الصفحة
        return render_template('index.html', 
                               prediction_text=f'Predicted House Price: ${prediction[0]:,.2f}',
                               neighborhoods=neighborhoods,
                               form_data=user_input)

    except Exception as e:
        print(f"!!! An error occurred during prediction: {e} !!!")
        return render_template('index.html', 
                               prediction_text=f'Error during prediction: {e}', 
                               neighborhoods=neighborhoods,
                               form_data=request.form.to_dict())

# 6. تشغيل التطبيق
if __name__ == '__main__':
    load_artifacts() # تحميل النماذج عند بدء التشغيل
    app.run(port=5000, debug=True)
