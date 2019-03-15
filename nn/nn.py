import os
import re


PATH = "C:\\Users\\Bhushan Borole\\Desktop\\Coding\\Projects\\Text-Summarization\\dataset\\cnn"

class WordTovec():
	def __init__(self, data):
		self.data = data

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
		return text.lower()

	def clean_text(self, text):
		text = re.sub(r'\[[0-9]*\]', ' ', text)
		text = re.sub(r'\s+', ' ', text)
		text = re.sub('[^a-zA-Z]', ' ', text)
		text = re.sub(r'\s+', ' ', text)
		text=re.sub(r"what's","what is ",text)
		text=re.sub(r"it's","it is ",text)
		text=re.sub(r"\'ve"," have ",text)
		text=re.sub(r"i'm","i am ",text)
		text=re.sub(r"\'re"," are ",text)
		text=re.sub(r"n't"," not ",text)
		text=re.sub(r"\'d"," would ",text)
		text=re.sub(r"\'s","s",text)
		text=re.sub(r"\'ll"," will ",text)
		text=re.sub(r"can't"," cannot ",text)
		text=re.sub(r" e g "," eg ",text)
		text=re.sub(r"e-mail","email",text)
		text=re.sub(r"9\\/11"," 911 ",text)
		text=re.sub(r" u.s"," american ",text)
		text=re.sub(r" u.n"," united nations ",text)
		text=re.sub(r"\n"," ",text)
		text=re.sub(r":"," ",text)
		text=re.sub(r"-"," ",text)
		text=re.sub(r"\_"," ",text)
		text=re.sub(r"\d+"," ",text)
		text=re.sub(r"[$#@%&*!~?%{}()]"," ",text)

		return text

	def get_file_names(self, dir, category):
		"""dataname refers to either training, test or validation"""
		#filenames = []
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
		for i in range(len(file_names)):
			#print(i)
			if i%2 == 0:
				self.data['articles'].append(self.clean_text(self.parse_text(self.path, self.dataset_categories[0], file_names[i])))
			else:
				self.data['summaries'].append(self.clean_text(self.parse_text(self.path, self.dataset_categories[0], file_names[i])))
		
		#self.printArticlesum(1)
		return self.data


if __name__ == '__main__':
	data = LoadDataset(PATH).load_dataset()
	

