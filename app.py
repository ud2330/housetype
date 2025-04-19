from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load model and encoders
model = joblib.load("model/house_model.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")
target_encoder = joblib.load("model/target_encoder.pkl")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        BHK = int(request.form['BHK'])
        style = request.form['style']
        stories = int(request.form['stories'])
        location = request.form['location']
        budget = int(request.form['budget'])

        if not style or not location:
            raise ValueError("Missing input: 'style' and 'location' are required.")

        # Construct input DataFrame
        input_data = pd.DataFrame([[BHK, style, stories, location, budget]],
                                  columns=['BHK', 'Architectural Style', 'Stories', 'Location Type', 'Budget'])

        # Encode categorical features
        for col in ['Architectural Style', 'Location Type']:
            le = label_encoders.get(col)
            if le:
                value = input_data.at[0, col]
                if value not in le.classes_:
                    raise ValueError(f"Invalid {col}: '{value}' not recognized.")
                input_data[col] = le.transform([value])
            else:
                raise ValueError(f"No encoder found for column: {col}")

        # Predict
        prediction = model.predict(input_data)
        house_type = target_encoder.inverse_transform(prediction)[0]

        # Build explanation points
        explanation_points = [
            f"{BHK}-BHK suggests a moderate to large house suitable for families.",
            f"The '{style}' style is commonly seen in {house_type} homes.",
            f"{stories} stories help define the structure, which suits a {house_type}.",
            f"'{location}' locations often accommodate such house types.",
            f"A budget of ₹{budget:,} supports the possibility of owning a {house_type}."
        ]

        price_range = f"{int(budget * 0.9):,} - {int(budget * 1.1):,} INR"

        return render_template(
            'result.html',
            prediction=house_type,
            price=price_range,
            explanation_points=explanation_points
        )

    except ValueError as ve:
        return render_template('error.html', error_message=str(ve))

    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

if __name__ == '__main__':
    # For Render.com or cloud platforms
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
