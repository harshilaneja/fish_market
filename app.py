from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load your dataset
df = pd.read_csv("Fish.csv")

# Preprocessing - assuming you've already done this step
df['Species'] = df['Species'].map({'Bream': 1, 'Roach': 2, 'Whitefish': 3, 'Parkki Species': 4, 'Perch': 5, 'Pike': 6, 'Smelt': 7})
df.dropna(subset=['Species'], inplace=True)
# Define features and target
x = df[['Length1', 'Length2', 'Length3', 'Weight', 'Width','Height']]
y = df['Species']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Create and train the model
model = LinearRegression()
model.fit(x_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from the form
        length1 = float(request.form['length1'])
        length2 = float(request.form['length2'])
        length3 = float(request.form['length3'])
        weight = float(request.form['weight'])
        width = float(request.form['width'])
        Height = float(request.form['Height'])

        # Make prediction using the model
        prediction = model.predict([[length1, length2, length3, weight, width,Height]])

        # Reverse map prediction to get the species name
        species_mapping = {1: 'Bream', 2: 'Roach', 3: 'Whitefish', 4: 'Parkki Species', 5: 'Perch', 6: 'Pike', 7: 'Smelt'}
        predicted_species = species_mapping[int(prediction[0])]
        predicted_species = species_mapping.get(int(prediction[0]), 'Unknown Species')

        return render_template('result.html', prediction=predicted_species)
if __name__ == '__main__':
    app.run(debug=True)
