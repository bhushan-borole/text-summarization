import numpy as np
from nltk.corpus import brown, stopwords
from nltk.cluster.util import cosine_distance
from operator import itemgetter
import nltk
import re
stopwords = stopwords.words('english')

class TextRank():
    def __init__(self):
        pass

    def clean_and_tokenize(self, article_text):
        # Removing Square Brackets and Extra Spaces
        article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
        article_text = re.sub(r'\s+', ' ', article_text)

        #sentence tokenization
        sentences = nltk.sent_tokenize(article_text)

        sentences = [nltk.word_tokenize(sent) for sent in sentences]

        # Removing special characters and digits
        #formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )
        #formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)
        return sentences

    def pagerank(self, A, eps=0.0001, d=0.85):
        P = np.ones(len(A)) / len(A)
        while True:
            new_P = np.ones(len(A)) * (1 - d) / len(A) + d * A.T.dot(P)
            delta = abs(new_P - P).sum()
            if delta <= eps:
                return new_P
            P = new_P


    def sentence_similarity(self, sent1, sent2, stopwords=None):
        if stopwords is None:
            stopwords = []
            
        sent1 = [w.lower() for w in sent1]
        sent2 = [w.lower() for w in sent2]
        
        all_words = list(set(sent1 + sent2))

        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)

        # build the vector for the first sentence
        for w in sent1:
            if w in stopwords:
                continue
            vector1[all_words.index(w)] += 1

        # build the vector for the second sentence
        for w in sent2:
            if w in stopwords:
                continue
            vector2[all_words.index(w)] += 1

        return 1 - cosine_distance(vector1, vector2)


    def build_similarity_matrix(self, sentences, stopwords=None):
        # Create an empty similarity matrix
        S = np.zeros((len(sentences), len(sentences)))

        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 == idx2:
                    continue
                S[idx1][idx2] = self.sentence_similarity(sentences[idx1], sentences[idx2], stopwords)

        # normalize the matrix row-wise
        for idx in range(len(S)):
            S[idx] /= S[idx].sum()

        return S

    def textrank(self, sentences, n_top=2, stopwords=None):
        """
        sentences = a list of sentences [[w11, w12, ...], [w21, w22, ...], ...]
        n_top = how may sentences the summary should contain
        stopwords = a list of stopwords
        """
        S = self.build_similarity_matrix(sentences, stopwords) 
        sentence_ranks = self.pagerank(S)
        
        # Sort the sentence ranks
        ranked_sentence_indexes = [item[0] for item in sorted(enumerate(sentence_ranks), key=lambda item: -item[1])]
        #print(ranked_sentence_indexes)
        print(n_top)
        selected_sentences = sorted(ranked_sentence_indexes[:n_top])
        print(selected_sentences)
        summary = itemgetter(*selected_sentences)(sentences)
        return summary


def textrank_summarize(text, num_sent=None):
    text_rank = TextRank()
    sentences = text_rank.clean_and_tokenize(text)
    original_length = len(sentences)

    if num_sent:
        if int(num_sent) < original_length:
            summary = text_rank.textrank(sentences, n_top=int(num_sent))
        elif original_length > 5:
            summary = text_rank.textrank(sentences, n_top=5)
        else:
            return 'Too short to summarize'
    summary = ' '.join([' '.join(summ) for summ in summary])
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
    summary = textrank_summarize(text)
    print(summary)

