import pandas as pd
import pickle
from flask import Flask, request, render_template

application = Flask(__name__)
app = application

# Import Lasso model and StandardScaler pickles
lasso_model = pickle.load(open('D:/Vardhan/ML/MLU/LINREGPROJECT/models/lasso.pkl', 'rb'))
standard_scaler = pickle.load(open('D:/Vardhan/ML/MLU/LINREGPROJECT/models/scaler.pkl', 'rb'))
expected_columns = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'ISI', 'Classes', 'Region']

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_data():
    if request.method == 'POST':
        try:
            # Retrieve input values from the form
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))

            # Prepare input data
            input_data = [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]
            new_data = pd.DataFrame(input_data, columns=expected_columns)

            # Debugging: Print input data
            print("Input DataFrame:")
            print(new_data)

            # Scale input data
            new_data_scaled = standard_scaler.transform(new_data)

            # Debugging: Print scaled data
            print("Scaled Data:", type(new_data_scaled), new_data_scaled.shape)

            # Predict using the Lasso model
            result = lasso_model.predict(new_data_scaled)
            print(result)
            # Render result on home.html
            return render_template('home.html', result=result[0])

        except Exception as e:
            print(f"Error during prediction: {e}")
            return render_template('home.html', result="Error occurred during prediction.")
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
