# from https://github.com/linanqiu/word2vec-sentiments/blob/master/word2vec-sentiment.ipynb
# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

# numpy
import numpy

# random
import random

# The default constructor for the default LabeledLineSentence class in Doc2Vec can do that for a single text file, but can't do that for multiple files. In classification tasks however, we usually deal with multiple documents (test, training, positive, negative etc). Ain't that annoying?

# So we write our own LabeledLineSentence class. The constructor takes in a dictionary that defines the files to read and the label prefixes sentences from that document should take on. Then, Doc2Vec can either read the collection directly via the iterator, or we can access the array directly. We also need a function to return a permutated version of the array of LabeledSentences. We'll see why later on.

class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
                with utils.smart_open(source) as fin:
                    for item_no, line in enumerate(fin):
                        self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffled = list(self.sentences)
        random.shuffle(shuffled)
        return shuffled

# Now we can feed the data files to LabeledLineSentence. As we mentioned earlier, LabeledLineSentence simply takes a dictionary with keys as the file names and values the special prefixes for sentences from that document. The prefixes need to be unique, so that there is no ambiguitiy for sentences from different documents.

# The prefixes will have a counter appended to them to label individual sentences in the documetns.

sources = {'test-irrel.txt':'TEST_IRREL', 'test-rel.txt':'TEST_REL', 'train-irrel.txt':'TRAIN_IRREL', 'train-rel.txt':'TRAIN_REL'}

sentences = LabeledLineSentence(sources)

# Model and vocabulary:

model = Doc2Vec(min_count = 1, window = 10, vector_size = 100, sample = 1e-4, negative = 5, workers = 7, epochs = 50)

model.build_vocab(sentences.to_array())

# We train the model for 5 epochs, on the permutated sentences, using the perm method in the class written above

model.train(sentences.sentences_perm(), epochs = model.epochs, total_examples = model.corpus_count)

model.save('doc2vec_model1')

#print(model.most_similar('integrity'))
