from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)  # ✅ Corrected __name_

# ✅ Load the ML model and encoder
model = pickle.load(open('model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # ✅ Collect form data
            temp = float(request.form['temp'])
            rain = float(request.form['rain'])
            snow = float(request.form['snow'])
            clouds = int(request.form['clouds'])
            weather_main = request.form['weather_main']
            weather_description = request.form['weather_description']

            # ✅ Handle unseen labels safely
            if weather_main in encoder.classes_:
                weather_main_enc = encoder.transform([weather_main])[0]
            else:
                weather_main_enc = -1  # Or choose a default value

            if weather_description in encoder.classes_:
                weather_desc_enc = encoder.transform([weather_description])[0]
            else:
                weather_desc_enc = -1  # Or choose a default value

            # ✅ Create feature array
            features = np.array([[temp, rain, snow, clouds]])

            # ✅ Make prediction
            prediction = model.predict(features)
            output = round(prediction[0], 2)

            return render_template('result.html', prediction_text=f'Estimated Traffic Volume: {output}')

        except Exception as e:
            # ✅ Error handling
            return f"Error occurred: {str(e)}"

# ✅ Correct main function
if __name__ == "__main__":
    app.run(debug=True)