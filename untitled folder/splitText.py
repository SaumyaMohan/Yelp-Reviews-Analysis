#this program was used to split the list of business reviews into 
#two seperate train/text files with a ratio of 4:1
import sklearn
from sklearn.model_selection import train_test_split

with open('FilteredFile.txt', 'r') as file:
	x = file.readlines()
	train, test = train_test_split(x, train_size = 0.8)
	file.close()

with open('train.txt', 'w') as f:
	for l in train:
		f.write(l)
f.close()
with open('test.txt', 'w') as g:
	for l in test:
		g.write(l)
g.close() 
