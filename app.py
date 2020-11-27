#Important Modules
from flask import Flask,render_template, url_for ,flash , redirect
#from forms import RegistrationForm, LoginForm
#from sklearn.externals import joblib
import joblib
from flask import request
import numpy as np
import tensorflow
#from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D
#from flask_sqlalchemy import SQLAlchemy
#from model_class import DiabetesCheck, CancerCheck

#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D
#from tensorflow.keras.layers import GlobalMaxPooling2D, Activation
#from tensorflow.keras.layers.normalization import BatchNormalization
#from tensorflow.keras.layers.merge import Concatenate
#from tensorflow.keras.models import Model

import os
from flask import send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

#from this import SQLAlchemy
app=Flask(__name__,template_folder='template')


app.config["SERVER_NAME"]='192.168.0.106:1000'

app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'


dir_path = os.path.dirname(os.path.realpath(__file__))

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'


from tensorflow.keras.models import load_model
model55 = load_model('model111.h5')

#FOR THE FIRST MODEL

# call model to predict an image
def api(full_path):
    data = image.load_img(full_path, target_size=(50, 50, 3))
    data = np.expand_dims(data, axis=0)
    data = data * 1.0 / 255

    #with graph.as_default():
    predicted = model55.predict(data)
    return predicted
#FOR THE SECOND MODEL

# home page

#@app.route('/')
#def home():
 #  return render_template('index.html')


# procesing uploaded file and predict it
@app.route('/upload', methods=['POST','GET'])
def upload_file():

    if request.method == 'GET':
        return render_template('index.html')
    else:
        try:
            file = request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)

            indices = {0: 'PARASITIC', 1: 'Uninfected', 2: 'Invasive carcinomar', 3: 'Normal'}
            result = api(full_name)
            print(result)

            predicted_class = np.asscalar(np.argmax(result, axis=1))
            accuracy = round(result[0][predicted_class] * 100, 2)
            label = indices[predicted_class]
            return render_template('predict.html', image_file_name = file.filename, label = label, accuracy = accuracy)
        except:
            flash("Please select the image first !!", "danger")      
            return redirect(url_for("Malaria"))




@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)



@app.route("/")

@app.route("/home")
def home():
    return render_template("home.html")
 


@app.route("/about")
def about():
    return render_template("about.html")






@app.route("/liver")
def liver():
   
    return render_template("liver.html")
@app.route("/Malaria")
def Malaria():
    return render_template("malaria.html")






def ValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    
    if(size==10):#liver
        loaded_model = joblib.load("model4555")
        result = loaded_model.predict(to_predict)
   
    return result[0]

@app.route('/result',methods = ["POST"])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
    
           
        if(len(to_predict_list)==10):
            result = ValuePredictor(to_predict_list,10)
    if(int(result)==1):
        prediction='Sorry ! Suffering'
    else:
        prediction='Congrats ! you are Healthy' 
    return(render_template("result.html", prediction=prediction))


if __name__ == "__main__":
    app.run(debug=True)
