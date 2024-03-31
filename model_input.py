from flask import Flask, request, render_template
import your_prediction_module  # Import your AI model for predicting controversiality

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input text from the form
        text = request.form['text']
        # Make prediction using your AI model
        prediction = your_prediction_module.predict_controversiality(text)
        # Return the prediction result
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
