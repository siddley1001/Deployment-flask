import numpy as np
from flask import Flask, request, render_template
from xgboost import XGBClassifier, Booster

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
    inputs = []
    for x in request.form.values():
        if x == 'Y':
            x = 1
            # inputs.append(x)
        elif x == 'N':
            x = 0
            # inputs.append(x)
        else:
            x = float(x)
            # inputs.append(x)
    inputs = [x for x in request.form.values()]
    inputs = np.array(inputs)
    # final_inputs = [np.array(inputs)]
    output = model.predict(inputs)

    # output = np.array(output)
    # output = np.array(float(output))

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
# def predict_api():
#     """For direct API calls through request"""
#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])
#
#     output = prediction[0]
#     return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
