import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

# Load trained model and encoder
model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect input values
    input_values = [x for x in request.form.values()]
    
    # Feature names (MUST match training)
    feature_names = ["hour_of_day", "day_of_week", "month", "weather", "temp", "rain", "snow"]

    # Create DataFrame
    input_df = pd.DataFrame([input_values], columns=feature_names)

    # Convert numeric columns to correct type
    for col in ["hour_of_day", "day_of_week", "month", "temp", "rain", "snow"]:
        input_df[col] = pd.to_numeric(input_df[col])

    # One-hot encode categorical
    encoded = encoder.transform(input_df[["weather"]])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(["weather"]))

    # Drop 'weather' and merge encoded columns
    input_df = input_df.drop(columns=["weather"]).reset_index(drop=True)
    final_input = pd.concat([input_df, encoded_df], axis=1)

    # Ensure correct feature order
    final_input = final_input.reindex(columns=model.feature_names_in_, fill_value=0)

    # Predict
    prediction = model.predict(final_input)[0]

    return render_template('index.html', result=f"Estimated Traffic Volume: {int(prediction)}")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
