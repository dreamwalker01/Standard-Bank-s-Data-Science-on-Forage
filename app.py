# REFERRED FROM KRISH NAIK SIR'S GITHUB CODE

from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        int_features = [float(x) for x in request.form.values()]  # Convert to float first
        int_features = [int(x) for x in int_features]  # Then convert to int
        
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)

        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text='Loan Status {}'.format(output))
    
    except ValueError as e:
        return render_template('index.html', prediction_text='Error: {}'.format(str(e)))

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.get_json(force=True)
        int_features = [float(val) for val in data.values()]
        int_features = [int(val) for val in int_features]

        prediction = model.predict([np.array(int_features)])
        output = prediction[0]
        
        return jsonify(output)
    
    except ValueError as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
