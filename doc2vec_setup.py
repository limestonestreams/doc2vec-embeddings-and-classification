# from https://github.com/linanqiu/word2vec-sentiments/blob/master/word2vec-sentiment.ipynb
# create 4 documents:
# 1. half of the irrelevant sentences in one training document
# 2. half of the relevant sentences in another training document
# 3. the other half of the irrelevant sentences in a test document
# 4. the other half of the relevant sentences in a test document
# every document should be in all lowercase, without punctuation, and with each sentence on one line

# reading excel file
import pandas
label_data = pandas.read_excel (r'C:\Users\annac\OneDrive\Documents\graduate\thesis\data\R code\label_data.xlsx', sheet_name = 'Sheet1')

# turning into pandas dataframe
foo = pandas.DataFrame(label_data)

# adding training and test set index
train = foo.sample(frac = 0.8, random_state = 135)
test = foo.drop(train.index)

# creating table for .translate() to remove punctuation
import string
table = str.maketrans('', '', string.punctuation)

# writing to text file test
#irrel_train = open(r"C:\Users\annac\OneDrive\Documents\graduate\thesis\data\python code\doc2vec1.txt","w+")

#for w in train['text'].where(train['class'] == 'irrelevant').dropna():
#    irrel_train.write(str(w).translate(table).lower())
#    irrel_train.write('\n')

#irrel_train.close()

# writing to text file 1 irrelevant training set
irrel_train = open(r"C:\Users\annac\OneDrive\Documents\graduate\thesis\data\python code\train-irrel.txt","w+")

for w in train['text'].where(train['class'] == 'irrelevant').dropna():
    irrel_train.write(str(w).translate(table).lower())
    irrel_train.write('\n')

irrel_train.close()

# writing to text file 2 relevant training set
rel_train = open(r"C:\Users\annac\OneDrive\Documents\graduate\thesis\data\python code\train-rel.txt","w+")

train2 = train[(train['class'] == 'legitimizing') | (train['class'] == 'against') | (train['class'] == 'for') | (train['class'] == 'humanizing') | (train['class'] == 'delegitimizing')]

for w in train2['text'].dropna():
    rel_train.write(str(w).translate(table).lower())
    rel_train.write('\n')

rel_train.close()

# writing to text file 3 irrelevant test set
irrel_test = open(r"C:\Users\annac\OneDrive\Documents\graduate\thesis\data\python code\test-irrel.txt","w+")

for w in test['text'].where(test['class'] == 'irrelevant').dropna():
    irrel_test.write(str(w).translate(table).lower())
    irrel_test.write('\n')

irrel_test.close()

# writing to text file 4 relevant test set
rel_test = open(r"C:\Users\annac\OneDrive\Documents\graduate\thesis\data\python code\test-rel.txt","w+")

test4 = test[(test['class'] == 'legitimizing') | (test['class'] == 'against') | (test['class'] == 'for') | (test['class'] == 'humanizing') | (test['class'] == 'delegitimizing')]

for w in test4['text'].dropna():
    rel_test.write(str(w).translate(table).lower())
    rel_test.write('\n')

rel_test.close()
