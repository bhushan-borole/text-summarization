import os
import re
import nltk
import numpy as np
from numpy.random import seed
seed(1)
import sys, gensim, logging, codecs, gzip
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize as wt
from nltk.tokenize import sent_tokenize as st
from gensim.models import Word2Vec
from numpy import argmax
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import logging
from os.path import exists
import pickle
stopwords = stopwords.words('english')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


PATH = "C:\\Users\\Bhushan Borole\\Desktop\\Coding\\Projects\\Text-Summarization\\dataset\\cnn"
gensim_model_name = 'w2v'


class LSTM():
	def __init__(self, training_data):
		self.training_data = training_data
		self.batch_size = 32
		self.epochs = 20
		self.hidden_units = 300
		self.learning_rate = 0.005
		self.clip_norm = 2.0
		self.encoder_shape = np.shape(training_data['article'][0])
		self.decoder_shape = np.shape(training_data['summaries'][0])

	def encode_decoder(self, data):
		print('Encoder_Decoder LSTM...')

		"""__encoder___"""
		encoder_inputs = Input(shape=self.encoder_shape)

		encoder_LSTM = LSTM(self.hidden_units,dropout_U=0.2, 
							dropout_W=0.2, return_sequences=True,
							return_state=True)
		encoder_LSTM_rev = LSTM(self.hidden_units,return_state=True,
								return_sequences=True, dropout_U=0.05,
								dropout_W=0.05, go_backwards=True)

		encoder_outputs, state_h, state_c = encoder_LSTM(encoder_inputs)
		encoder_outputsR, state_hR, state_cR = encoder_LSTM_rev(encoder_inputs)

		state_hfinal=Add()([state_h,state_hR])
		state_cfinal=Add()([state_c,state_cR])
		encoder_outputs_final = Add()([encoder_outputs,encoder_outputsR])

		encoder_states = [state_hfinal,state_cfinal]

		"""____decoder___"""
		decoder_inputs = Input(shape=(None,self.decoder_shape[1]))
		decoder_LSTM = LSTM(self.hidden_units, return_sequences=True,
							dropout_U=0.2, dropout_W=0.2, return_state=True)
		decoder_outputs, _, _ = decoder_LSTM(self.decoder_inputs, initial_state=encoder_states)

		#Pull out XGBoost, (I mean attention)
		attention = TimeDistributed(Dense(1, activation = 'tanh'))(encoder_outputs_final)
		attention = Flatten()(attention)
		attention = Multiply()([decoder_outputs, attention])
		attention = Activation('softmax')(attention)
		attention = Permute([2, 1])(attention)

		decoder_dense = Dense(self.decoder_shape[1],activation='softmax')
		decoder_outputs = decoder_dense(attention)

		model= Model(inputs=[encoder_inputs,decoder_inputs], outputs=decoder_outputs)
		print('-------------Model Summary------------')
		print(model.summary())
		print('-'*40)

		rmsprop = RMSprop(lr=self.learning_rate, clipnorm=self.clip_norm)
		model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])

		x_train, x_test, y_train, y_test = tts(data["article"],data["summaries"], test_size=0.20)
		history= model.fit(x=[x_train,y_train],
		      y=y_train,
		      batch_size=self.batch_size,
		      epochs=self.epochs,
		      verbose=1,
		      validation_data=([x_test,y_test], y_test))
		print('-------------Model Summary------------')
		print(model.summary())
		print('-'*40)

		"""_________________inference mode__________________"""
		encoder_model_inf = Model(encoder_inputs,encoder_states)

		decoder_state_input_H = Input(shape=(self.encoder_shape[0],))
		decoder_state_input_C = Input(shape=(self.encoder_shape[0],)) 
		decoder_state_inputs = [decoder_state_input_H, decoder_state_input_C]
		decoder_outputs, decoder_state_h, decoder_state_c = decoder_LSTM(decoder_inputs,
		                                                             initial_state=decoder_state_inputs)
		decoder_states = [decoder_state_h, decoder_state_c]
		decoder_outputs = decoder_dense(decoder_outputs)

		decoder_model_inf= Model([decoder_inputs]+decoder_state_inputs,
		                 [decoder_outputs]+decoder_states)

		scores = model.evaluate([x_test,y_test],y_test, verbose=1)


		print('LSTM test scores:', scores)
		print('-------------Model Summary------------')
		print(model.summary())
		print('-'*40)
		return model,encoder_model_inf,decoder_model_inf,history

class Word2vec():
	def __init__(self, data):
		self.data = data
		pass

	def create_corpus(self):
		corpus = []
		all_sentences = []
		for k in self.data:
			for p in self.data[k]:
				corpus.append(st(p))
		for sent in range(len(corpus)):
			for k in corpus[sent]:
				all_sentences.append(k)
		for m in range(len(all_sentences)):
			all_sentences[m] = wt(all_sentences[m])

		all_words=[]
		for sent in all_sentences:
			hold=[]
			for word in sent:
				hold.append(word.lower())
			all_words.append(hold)
		return all_words

	def save_gensim_model(self, model):
		with open(gensim_model_name, 'wb') as model_file:
			pickle.dump(model, model_file)

	def update_model(self, model, corpus):
		logger.info('Updating model')
		model.train(corpus, total_examples=len(corpus), epochs=25)
		self.save_gensim_model(model)
		return model

	def word2vec_model(self, corpus):
		emb_size = 300
		model_type = {"skip_gram":1,"CBOW":0}
		window = 10
		workers = 6
		min_count = 1
		batch_words = 20
		epochs = 25
		#include bigrams
		#bigramer = gs.models.Phrases(corpus)

		model = Word2Vec(corpus,size=emb_size,sg=model_type["skip_gram"],
		                     compute_loss=True,window=window,min_count=min_count,workers=workers,
		                     batch_words=batch_words)
		self.save_gensim_model(model)
		model.train(corpus,total_examples=len(corpus),epochs=epochs)
		
		logger.info('Model deployed')
		return model
	'''
	def update(self, model, corpus, mincount=3):
		"""
		Add new words from new data to the existing model's vocabulary,
		generate for them random vectors in syn0 matrix.
		For existing words, increase their counts by their frequency in the new data.
		Generate new negative sampling matrix (syn1neg).
		Then, train the existing model with the new data.
		"""
		added_count = 0

		logging.info("Extracting vocabulary from new data...")
		newmodel = gensim.models.Word2Vec(min_count=mincount, sample=0, hs=0)
		newmodel.build_vocab(corpus)

		logging.info("Merging vocabulary from new data...")
		sampleint = model.wv.vocab[model.index2word[0]].sample_int
		words = 0
		newvectors = []
		newwords = []
		for word in newmodel.vocab:
			words += 1
		if word not in model.vocab:
			v = gensim.models.word2vec.Vocab()
			v.index = len(model.vocab)
			model.wv.vocab[word] = v
			model.wv.vocab[word].count = newmodel.vocab[word].count
			model.wv.vocab[word].sample_int = sampleint
			model.index2word.append(word)

			random_vector = model.seeded_vector(model.index2word[v.index] + str(model.seed))
			newvectors.append(random_vector)

			added_count += 1
			newwords.append(word)
		else:
			model.vocab[word].count += newmodel.vocab[word].count
		if words % 1000 == 0:
			logging.info("Words processed: %s" % words)

		logging.info("added %d words into model from new data" % (added_count))

		logging.info("Adding new vectors...")
		alist = [row for row in model.syn0]
		for el in newvectors:
			alist.append(el)
		model.syn0 = array(alist)

		logging.info("Generating negative sampling matrix...")
		model.syn1neg = zeros((len(model.vocab), model.layer1_size), dtype=REAL)
		model.make_cum_table()

		model.neg_labels = zeros(model.negative + 1)
		model.neg_labels[0] = 1.

		model.syn0_lockf = ones(len(model.vocab), dtype=REAL)

		logging.info("Training with new data...")
		model.train(corpus, total_examples=len(corpus))

		return model
	'''

	def encode(self, corpus):
		all_words = []
		one_hot = {}
		for sent in corpus:
			for word in wt(' '.join(sent)):
				all_words.append(word.lower())
		#print(len(set(all_words)), "unique words in corpus")
		logger.info(str(len(all_words)) + 'unique words in corpus')
		#maxcorp=int(input("Enter desired number of vocabulary: "))
		maxcorp = int(len(set(all_words)) / 1.1)
		wordcount = Counter(all_words).most_common(maxcorp)
		all_words = []

		for p in wordcount:
			all_words.append(p[0])  
		    
		all_words = list(set(all_words))

		#print(len(all_words), "unique words in corpus after max corpus cut")
		#logger.info(str(len(all_words)) + 'unique words in corpus after max corpus cut')
		#integer encode
		#label_encoder = LabelEncoder()
		#integer_encoded = label_encoder.fit_transform(all_words)
		#one hot
		label_encoder = LabelEncoder()
		integer_encoded = label_encoder.fit_transform(all_words)
		onehot_encoder = OneHotEncoder(sparse=False)
		#integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
		onehot_encoded = onehot_encoder.fit_transform(np.array(all_words).reshape(-1, 1))
		for i in range(len(onehot_encoded)):
			word = label_encoder.inverse_transform([argmax(onehot_encoded[i, :])])[0].strip()
			one_hot[word] = onehot_encoded[i]
		#print(len(one_hot.keys()))
		return one_hot

	def word_vec_matrix(self, model, one_hot):
		training_data = {"article":[],"summaries":[]}
		i=1
		for k in range(len(self.data["articles"])):
			art=[]
			summ=[]
			for word in wt(self.data["articles"][k].lower()):
				try:
				    art.append(model.wv.word_vec(word))
				except Exception as e:
				    print(e)

			for word in wt(self.data["summaries"][k].lower()):
				try:
					summ.append(one_hot[word])
					#summ.append(model.wv.word_vec(word))
				except Exception as e:
					print(e)

			training_data["article"].append(art) 
			training_data["summaries"].append(summ)
			if i%100==0:
				logger.info("progress: " + str(((i*100)/len(self.data["articles"]))))
			i+=1
		
		print('\007')
		return training_data



class LoadDataset():
	def __init__(self, path):
		self.path = path
		self.dataset_categories = ['training', 'validation', 'test']
		self.data = {
				'articles': [],
				'summaries': []
		}

	def parse_text(self, dir, category, filename):
		with open(dir+'\\'+category+'\\'+filename, 'r', encoding="Latin-1") as f:
			#print("{}: {} read successfully".format(category, filename))
			text = f.read()
		text = self.clean_text(text)
		return text

	def clean_text(self, text):
		text = re.sub(r'\[[0-9]*\]', ' ', text)
		text = re.sub('[^a-zA-Z]', ' ', text)
		text = re.sub(r'\s+', ' ', text)
		text = re.sub(r"what's","what is ",text)
		text = re.sub(r"it's","it is ",text)
		text = re.sub(r"\'ve"," have ",text)
		text = re.sub(r"i'm","i am ",text)
		text = re.sub(r"\'re"," are ",text)
		text = re.sub(r"n't"," not ",text)
		text = re.sub(r"\'d"," would ",text)
		text = re.sub(r"\'s","s",text)
		text = re.sub(r"\'ll"," will ",text)
		text = re.sub(r"can't"," cannot ",text)
		text = re.sub(r" e g "," eg ",text)
		text = re.sub(r"e-mail","email",text)
		text = re.sub(r"9\\/11"," 911 ",text)
		text = re.sub(r" u.s"," american ",text)
		text = re.sub(r" u.n"," united nations ",text)
		text = re.sub(r"\n"," ",text)
		text = re.sub(r":"," ",text)
		text = re.sub(r"-"," ",text)
		text = re.sub(r"\_"," ",text)
		text = re.sub(r"\d+"," ",text)
		text = re.sub(r"[$#@%&,\"'*!~?%{}()]"," ",text)
		return text

	def get_file_names(self, dir, category):
		files = os.listdir(dir + '\\' + category)
		print(len(files))
		return files

	def printArticlesum(self, k):
	    print("---------------------original sentence-----------------------")
	    print("-------------------------------------------------------------")
	    print(self.data["articles"][k])
	    print("----------------------Summary sentence-----------------------")
	    print("-------------------------------------------------------------")
	    print(self.data["summaries"][k])

	def load_dataset(self):
		file_names = self.get_file_names(self.path, self.dataset_categories[0])
		for i in range(len(file_names[:1000])):
			if i%2 == 0:
				self.data['articles'].append(self.parse_text(self.path, self.dataset_categories[0], file_names[i]))
			else:
				self.data['summaries'].append(self.parse_text(self.path, self.dataset_categories[0], file_names[i]))
		
		#self.printArticlesum(1)
		logger.info('data loaded from {} to {}'.format(x, y))
		logger.info('Length of data: {}'.format(len(self.data['articles']) + len(self.data['articles'])))
		return self.data


if __name__ == '__main__':
	data = LoadDataset(PATH).load_dataset()
	w2v = Word2vec(data)
	corpus = w2v.create_corpus()
	one_hot = w2v.encode(corpus)
	if exists(gensim_model_name):
		with open(gensim_model_name, 'rb') as model_file:
			model = pickle.load(model_file)
	else:
		model = w2v.word2vec_model(corpus)
	# model = w2v.update(model, corpus)
	training_data = w2v.word_vec_matrix(model, one_hot)
	#print(model['quick'])
	print('The vocabulary size of the model is: ', len(model.wv.vocab))
	print("summary length: ",len(training_data["summaries"][0]))
	print("article length: ",len(training_data["article"][0]))
	

