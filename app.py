from flask import Flask
from flask import request
from joblib import load

app = Flask(__name__)
model_path = "svm_gamma=0.001_C=0.2.joblib"
model = load(model_path)

@app.route("/")
def ping():
     return {"server":"up. image different"}


@app.route("/predict", methods=['POST'])
def predict_digit():
    image1 = request.json['image1']
    image2=request.json['image2']
    class1 = model.predict([image1])
    class2=model.predict([image2])
    if class1==class2:
        return {"result":"same_number"}
    else:
        return {"result":"different_number"}

if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0')