import os
import numpy as np
from flask import Flask,render_template,request
import pickle
import requests
import json

dirname = os.path.dirname(__file__)
model_name = os.path.join(dirname, 'lr_model.pkl')
# template_name = os.path.join(dirname, 'index.html')

app = Flask(__name__)
model = pickle.load(open(model_name, 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict_mean_value',methods=['POST'])
def predict_mean_value():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Predicted Complex mean value $ {}'.format(output))

@app.route('/predict_from_ml',methods=['POST'])
def predict_from_ml():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features).tolist()]

    url = "http://ee627d30-5244-4116-83e4-ff8ccd8e8841.westeurope.azurecontainer.io/score"
    data = {"data": final_features}
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    r = requests.post(url, data=json.dumps(data), headers=headers)
    data = r.json()
    prediction = data['predict']
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_ml='Predicted Complex mean value $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)