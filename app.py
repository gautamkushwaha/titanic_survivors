from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the model and scaler
with open('model.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    scaler = data['scaler']

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    Pclass = int(request.form['Pclass'])
    Sex = int(request.form['Sex'])
    Age = float(request.form['Age'])
    Fare = float(request.form['Fare'])

    # Prepare data for prediction
    input_data = np.array([[Pclass, Sex, Age, Fare]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)

    # Return result
    result = "Survived" if prediction[0] == 1 else "Did not survive"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
