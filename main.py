import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from flask import Flask, render_template, request
import pickle  # Initialize the flask App
from sklearn.linear_model import LinearRegression
import Gunicorn


'''app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))'''

df = pd.read_csv("FuelConsumption.csv")
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
x = cdf.iloc[:, :3]
y = cdf.iloc[:, -1]
regressor = LinearRegression()
regressor.fit(x, y)
pickle.dump(regressor, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2.6, 8, 10.1]]))


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    #For rendering results on HTML GUI
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text='CO2    Emission of the vehicle is :{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)