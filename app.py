from flask import Flask, request

app = Flask(__name__)
# model = load_model("./model/mymodel_15")

@app.route('/model', methods=['POST'])
def model():
    data = request.get_json()
    content = data['content']
    line_number = data['line_number']
    # Perform your model computation using the content and line number
    # result = predict(model,content,line_number)
    result = [2]
    return {'result': result}


if __name__ == '__main__':
    app.run()
