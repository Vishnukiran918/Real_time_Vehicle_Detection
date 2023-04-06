from flask import Flask, render_template, request
from tensorflow import keras
from PIL import Image
import numpy as np


app = Flask(__name__)

model = keras.models.load_model('C:\\Users\\CH Vishnukiran\\Desktop\\Real_time_Vehicle_Detection\\mit\\app\\models\\vehicle_classification_model.h5')

classes = {0: 'bus',
           1: 'car',
           2: 'motorcycle',
           3: 'small lorry',
           4: 'van'}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        img = Image.open(file).resize((150, 150))

        img = np.array(img)
        img = np.expand_dims(img, axis=0)
        result = model.predict(img)
        prediction = classes[np.argmax(result)]
        print(f"Predicted vehicle class: {prediction}")
        return render_template('result.html', prediction=prediction)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
