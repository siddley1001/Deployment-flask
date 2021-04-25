import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('xgbc_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """For rendering results on HTML GUI"""
    inputs = []
    for x in request.form.values():
        if x == 'Y':
            x = 1
        elif x == 'N':
            x = 0
        else:
            x = int(x)
        inputs.append(x)

    final_inputs = [np.array(inputs)]
    prediction = model.predict(final_inputs)

    output = prediction[0]

    if output == 1:
        return render_template('index.html', prediction_text='Individual WILL have a stroke'
                                                             '\n Note that the following model predicts strokes with '
                                                             '96% '
                                                             'accuracy')
    else:
        return render_template('index.html', prediction_text='Individual will NOT have a stroke'
                                                             '\n Note that the following model predicts strokes with '
                                                             '96% '
                                                             'accuracy')

# @app.route('/predict_api', methods=['POST'])
# def predict_api():
#     """For direct API calls through request"""
#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])
#
#     output = prediction[0]
#     return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
