from flask import render_template, Flask, request, flash
import flask
from luhn_summarizer import luhn_summarize
from text_rank import textrank_summarize

app = Flask(__name__)
type_of_summaries = ['luhn', 'textrank']

@app.route('/')
def index():
    print('index')
    return render_template('index.html')


@app.route('/summarize', methods=['GET', 'POST'])
def summarize():
    # print('summar')
    if request.method == 'POST':
        text = request.form['text']
        num_sent = None
        if request.form['num_sentences']:
            num_sent = request.form['num_sentences']

        if request.form['type_of_summ'] in type_of_summaries:
            type_of_summary = request.form['type_of_summ']
            if type_of_summary == 'luhn':
                summary = luhn_summarize(text, num_sent=num_sent)
                return render_template("index.html", summary=summary)

            elif type_of_summary == 'textrank':
                summary = textrank_summarize(text, num_sent=num_sent)
                return render_template("index.html", summary=summary)
        else:
            summary = luhn_summarize(text, num_sent=num_sent)
            return render_template("index.html", summary=summary)
    else:
    	return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
