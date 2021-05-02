## ML-Model-Flask-Deployment
This is a simple project to predict the probability of a stroke using the XGBoost Classifier Model. 
The model is deployed in production with Heroku using the Flask API

### Prerequisites
The primary packages are xgboost, pandas, Flask, and gunicorn - the rest are specified in the requirements.txt file

### Project Structure
This project has 3 major parts :
1. ```model.py``` - This contains code for our Machine Learning classification model to predict the probability of a stroke using the 'healthcare-dataset.csv' file.
2. ```app.py``` - This contains Flask APIs that receives employee details through GUI or API calls, summarizes the predicted probability based on our model and returns Yes/No they will are susceptible to a stroke.
3. ```templates``` - This folder contains the HTML template to allow user to enter an individual's health and personal details.

*NOTE the app will not track any user data.

You can visit the deployed Flask app on: https://xgboost-strokepredict.herokuapp.com/
