# importing required libraries
import re
import nltk
from nltk import sent_tokenize, word_tokenize
from gensim.summarization import summarize
stopwords = nltk.corpus.stopwords.words('english')


class Luhn_Summarizer():
    def __init__(self):
        pass

    def clean_and_tokenize(self, article_text):
        # Removing Square Brackets and Extra Spaces
        article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
        article_text = re.sub(r'\s+', ' ', article_text)

        #sentence tokenization
        sentences = nltk.sent_tokenize(article_text)

        # Removing special characters and digits
        formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )
        formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)
        return sentences, formatted_article_text

    def find_word_frequencies(self, cleaned_text):
        word_frequencies = {}
        for word in nltk.word_tokenize(cleaned_text):
            if word not in stopwords:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1

        maximum_frequncy = max(word_frequencies.values())

        for word in word_frequencies.keys():
            word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

        return word_frequencies

    def calculate_sentence_scores(self, sentences, word_frequencies):
        sentence_scores = {}
        for sent in sentences:
            for word in nltk.word_tokenize(sent.lower()):
                if word in word_frequencies.keys():
                    if len(sent.split(' ')) < 30:
                        if sent not in sentence_scores.keys():
                            sentence_scores[sent] = word_frequencies[word]
                        else:
                            sentence_scores[sent] += word_frequencies[word]
        return sentence_scores

    def summarize(self, article, num_sent=None):
        sentences, cleaned_text = self.clean_and_tokenize(article)
        original_length = len(sentences)
        word_freq = self.find_word_frequencies(cleaned_text)
        sentence_scores = sorted(self.calculate_sentence_scores(sentences, word_freq), reverse=True)
        if num_sent:
            if int(num_sent) < original_length:
                return ' '.join(sentence_scores[:int(num_sent)])
        if original_length > 5:
            return ' '.join(sentence_scores[:5])
        else:
            return 'Too short to summarize'


def luhn_summarize(text, num_sent=None):
    summarizer = Luhn_Summarizer()
    summary = summarizer.summarize(text, num_sent=num_sent)
    return summary


if __name__ == '__main__':
    text = '''
    Diabetes is a chronic disease or group of metabolic disease where a person suffers from an extended level of blood glucose in the body, 
    which is either the insulin production is inadequate, or because the body’s cells do not respond properly to insulin. 
    The constant hyperglycemia of diabetes is related to long-haul harm, brokenness, and failure of various organs, 
    particularly the eyes, kidneys, nerves, heart, and veins. The objective of this research is to make use of significant features, 
    design a prediction algorithm using Machine learning and find the optimal classifier to give the closest result 
    comparing to clinical outcomes. The proposed method aims to focus on selecting the attributes that ail in early detection 
    of Diabetes Miletus using Predictive analysis. The result shows the decision tree algorithm and the Random forest has the 
    highest specificity of 98.20% and 98.00%, respectively holds best for the analysis of diabetic data. Naïve Bayesian outcome 
    states the best accuracy of 82.30%. The research also generalizes the selection of optimal features from dataset to improve 
    the classification accuracy.
    '''
    summary = luhn_summarize(text)
    print(summary)

        





