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
		num_sent = None
		if request.form['num_sentences']:
			num_sent = request.form['num_sentences']
		summarizer = Extractive_Summarizer()
		summary = summarizer.summarize(text, num_sent=num_sent)
		return render_template("index.html", summary=summary)
	else:
		return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
