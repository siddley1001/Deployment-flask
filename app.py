import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from xgboost import XGBClassifier, Booster, DMatrix

bst = Booster()

app = Flask(__name__)
model = XGBClassifier()
dirr = '/Users/vanamsid/Deployment-flask/'
model.load_model(dirr + 'xgbc_model.json')



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """For rendering results on HTML GUI"""
    inputs = [x for x in request.form.values()]

    transformer = {'Y': 1,
                   'N': 0}
    inputs = [transformer.get(key,0) for key in inputs]

    inputs = np.array(inputs).reshape((1,-1))

    output = model.predict(inputs)

    if output == 1:
        return render_template('index.html', prediction_text='Individual WILL have a stroke'
                                                             '\n Note that the following model predicts strokes with '
                                                             '96% '
                                                             'accuracy')
    elif output == 0:
        return render_template('index.html', prediction_text='Individual will NOT have a stroke'
                                                             '\n Note that the following model predicts strokes with '
                                                             '96% '
                                                             'accuracy')

# @app.route('/predict_api', methods=['POST'])
# def predict_api():                                                                                           Stroke Prediction with XGBoost
#     """For direct API calls through request"""
#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])
#
#     output = prediction[0]
#     return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
