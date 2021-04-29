import numpy as np
import os
from flask import Flask, request, render_template
from xgboost import XGBClassifier, Booster

bst = Booster()

app = Flask(__name__)
model = XGBClassifier()
dirr = '/Users/vanamsid/Deployment-flask/'
model.load_model(dirr + 'xgbc_model.json')

port = int(os.environ.get('PORT', 5000))


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

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port, debug=True)
