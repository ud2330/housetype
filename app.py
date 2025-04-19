from flask import Flask, render_template, request
import joblib  # Switching from pickle to joblib for better compatibility with large models
import pandas as pd

app = Flask(__name__)

# Load model and encoders
model = joblib.load("model/house_model.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")
target_encoder = joblib.load("model/target_encoder.pkl")

@app.route('/')
def home():
    return render_template('home.html')  # Optional landing page

@app.route('/index')
def index():
    return render_template('index.html')  # Input form page

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        BHK = int(request.form['BHK'])
        style = request.form['style']
        stories = int(request.form['stories'])
        location = request.form['location']
        budget = int(request.form['budget'])

        # Ensure all input data is valid
        if not style or not location:
            raise ValueError("Please provide valid 'style' and 'location' values.")

        # Create DataFrame from inputs
        input_data = pd.DataFrame([[BHK, style, stories, location, budget]],
                                  columns=['BHK', 'Architectural Style', 'Stories', 'Location Type', 'Budget'])

        # Apply label encoders for categorical data
        for col in ['Architectural Style', 'Location Type']:
            if col in label_encoders:
                le = label_encoders[col]
                # Check if the label exists in the encoder
                if style not in le.classes_ or location not in le.classes_:
                    raise ValueError(f"Invalid {col} value: {style if col == 'Architectural Style' else location}")
                input_data[col] = le.transform(input_data[col])
            else:
                raise ValueError(f"Unknown label for column: {col}")

        # Predict house type
        prediction = model.predict(input_data)
        house_type = target_encoder.inverse_transform(prediction)[0]

        # Explanation points
        explanation_points = [
            f"{BHK}-BHK suggests a moderate to large house suitable for families.",
            f"The '{style}' style is commonly seen in {house_type} homes.",
            f"{stories} stories help define the structure, which suits a {house_type}.",
            f"'{location}' locations often accommodate such house types.",
            f"A budget of ₹{budget:,} supports the possibility of owning a {house_type}."
        ]

        # Estimate price range
        price_range = f"{int(budget * 0.9):,} - {int(budget * 1.1):,}"
        display_price = f"{price_range} INR"

        # Return the result page
        return render_template(
            'result.html',
            prediction=house_type,
            price=display_price,
            explanation_points=explanation_points
        )

    except ValueError as e:
        # If a value error is raised (invalid input), show an error page
        return render_template('error.html', error_message=str(e))

    except Exception as e:
        # If any other error occurs, show a general error message
        return f"An error occurred during prediction: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
