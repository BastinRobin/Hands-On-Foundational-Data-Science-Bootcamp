from flask import Flask, request, render_template, jsonify
import numpy as np
from joblib import dump, load



app = Flask(__name__)

@app.route('/',methods=['POST','GET'])
def home():
	return render_template('index.html')



@app.route('/models/diabetic/result', methods=['GET'])
def result():

	payload = request.args.to_dict()

	to_predict = list(payload.values())

	# temp = []
	# for i in to_predict:
	# 	temp.append(float(i))

	to_predict = [float(i) for i in to_predict]


	lr = load('logit_trained.joblib')

	output = lr.predict([to_predict])

	if output[0] == 0:
		result = 'Non diabetic'
		status = output.tolist()
	else:
		result = 'diabetic'
		status = output.tolist()

	response = {

		"result" : result,
		"status" : status
	}

	return render_template('result.html', data=response)


@app.route('/models/diabetic', methods=['GET'])
def predict_diabetic_api():

	payload = request.args.to_dict()

	to_predict = list(payload.values())

	# temp = []
	# for i in to_predict:
	# 	temp.append(float(i))

	to_predict = [float(i) for i in to_predict]


	lr = load('logit_trained.joblib')

	output = lr.predict([to_predict])

	if output[0] == 0:
		result = 'Non diabetic'
		status = output.tolist()
	else:
		result = 'diabetic'
		status = output.tolist()

	response = {

		"result" : result,
		"status" : status
	}

	return response


@app.route('/about', methods=['GET'])
def about_page():
	return """
		<h3>About Us</h3>
		<p>Welcome to the prediction page</p>
	"""



if __name__ == '__main__':
	app.debug = True
	app.run()