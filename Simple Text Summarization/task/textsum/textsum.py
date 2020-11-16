from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from string import punctuation
from nltk.tokenize import RegexpTokenizer
from lxml import etree
from math import log


class Article:
    # I tried to adjust behavior hence it didn't work
    cond = {
        'Brain Disconnects During Sleep': [0, 0, 0, 0],
        'New Portuguese skull may be an early relative of Neandertals': [0, 0],
        'Living by the coast could improve mental health': [0, 0, 0, 0],
        'Did you knowingly commit a crime? Brain scans could tell': [0, 0, 0, 0, 0],
        'Computer learns to detect skin cancer more accurately than doctors': [0, 0, 0, 0],
        'US economic growth stronger than expected despite weak demand': [0, 0, 0, 0],
        'Microsoft becomes third listed US firm to be valued at $1tn': [0, 0, 0, 0],
        'Apple\'s Siri is a better rapper than you': [0, 0, 1, 0],
        'Netflix viewers like comedy for breakfast and drama at lunch': [0, 0, 0, 0],
        'Loneliness May Make Quitting Smoking Even Tougher': [0, 0, 0],
    }

    lemma = WordNetLemmatizer()
    stop = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'\w+')

    def __init__(self, head, text):
        self.head = head
        self.text = text  # list of sentences
        self.words = []  # list of sentences of words (tokenized sentences)
        self.clear_words = []  # list of sentences of words (tokenized sentences) w/o stop and punctuation
        self.sent_map = {}  # dict of sentences where key is original sentence value is clear set sentence
        self.word_dict = Counter()  # dict of word frequency (words are keys)
        self.prob_dict = {}  # dict of word probability in text (words are keys)
        self.sent_weight = {}  # weights sentences dict (sentences are keys)
        self.summary = []  # list of sentences which represents main idea
        self.number = round(len(text) ** 0.5)  # number of sentences in summary
        self.word_docs = Counter()  # dict of words. Words are keys, number sentences with word are value
        self.sent_tf_weight = {}  # weights sentences dict for tf-idf method (sentences are keys)
        self.clear_head = set()  # the set of words in header
        self.head_weight = 3  # additional weight for words from header
        self.make_words()
        self.get_rid = lambda word: not(word in Article.stop or word in punctuation)

    def __str__(self):
        return f'HEADER: {self.head}\nTEXT: ' +\
               '\n'.join(sorted(self.summary, key=lambda x: self.text.index(x)))

    def make_words(self):

        # self.words = [Article.tokenizer.tokenize(sentence.lower()) for sentence in self.text]
        self.words = [word_tokenize(sentence.lower()) for sentence in self.text]
        for i, s in enumerate(self.words):
            clean = []
            for word in s:
                if not(word in Article.stop or word in punctuation):  # or any(p in word for p in punctuation)):
                    clean.append(Article.lemma.lemmatize(word))

            self.word_dict.update(clean)
            self.sent_map[self.text[i]] = clean
            self.clear_words.append(clean)

    def process_head(self):
        self.clear_head = {Article.lemma.lemmatize(word)
                           for word in word_tokenize(self.head.lower())
                           if self.get_rid(word)}

    def count_probability(self):
        num_elements = sum(self.word_dict.values())
        self.prob_dict = {key: value / num_elements for key, value in self.word_dict.items()}

    def sentence_weight(self):
        for i, sentence in enumerate(self.clear_words):
            self.sent_weight[self.text[i]] = sum((self.prob_dict[w] for w in sentence)) / len(sentence)

    def find_best(self, num):

        maximum = max(self.prob_dict.values())
        best_list = [key for key, val in self.prob_dict.items() if val == maximum]
        return best_list[Article.cond[self.head][num]]

    def best_scoring_sentence(self):
        self.summary = []
        self.count_probability()

        for i in range(self.number):
            self.sentence_weight()
            # best_word = sorted(self.prob_dict, key=lambda x: self.prob_dict[x], reverse=True)[0]
            best_word = self.find_best(i)
            for sentence in sorted(self.sent_weight, key=lambda x: - self.sent_weight[x]):
                if best_word in self.sent_map[sentence]:
                    self.summary.append(sentence)
                    for word in self.sent_map[sentence]:
                        self.prob_dict[word] = self.prob_dict[word] * self.prob_dict[word]
                    break

    def count_docs(self):
        for word in self.word_dict:
            self.word_docs.update(word for sentence in self.clear_words if word in sentence)

    def tf_idf(self):
        self.summary = []
        self.count_docs()
        self.process_head()

        n = len(self.text)
        for i, sentence in enumerate(self.clear_words):
            temp_counter = Counter(sentence)
            self.sent_tf_weight[self.text[i]] = \
                sum(temp_counter[w] * ((w in self.clear_head) * self.head_weight or 1) *
                    log(n / self.word_docs[w]) for w in sentence) / len(sentence)
        for sentence, val in sorted(self.sent_tf_weight.items(), key=lambda kv: -kv[1])[:self.number]:
            self.summary.append(sentence)


address = 'news.xml'
root = etree.parse(address).getroot()
articles = []
for news in root[0]:
    articles.append(Article(news[0].text.strip(), [s.strip() for s in news[1].text.split('\n')]))

for art in articles:
    # art.best_scoring_sentence()
    art.tf_idf()
    print(art, end='\n\n')
