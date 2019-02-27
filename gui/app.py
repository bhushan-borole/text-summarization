from flask import render_template, Flask, request
import flask
from summarizer import Extractive_Summarizer

app = Flask(__name__)

@app.route('/')
def index():
	print('index')
	return render_template('index.html')


@app.route('/summarize', methods=['GET', 'POST'])
def summarize():
	print('summar')
	
	if request.method == 'POST':
		text = request.form['text']
		summarizer = Extractive_Summarizer()
		summary = summarizer.summarize(text)
		return render_template("index.html", summary=summary)
	else:
		return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
