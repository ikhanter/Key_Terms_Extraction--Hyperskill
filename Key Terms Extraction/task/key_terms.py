from sklearn.feature_extraction.text import TfidfVectorizer
import string
import numpy as np
import nltk
from lxml import etree


xml_file = 'news.xml'
root = etree.parse(xml_file).getroot()
vectorizer = TfidfVectorizer(input='content', lowercase=True, ngram_range=(1,1), min_df=0.1, max_df=0.6)
all_texts = []
articles_dict = {}
for news in root[0]:
    articles_dict[news[0].text] = news[1].text
    text = articles_dict[news[0].text]
    tokenized_text = nltk.tokenize.word_tokenize(text.lower())
    lemmatized_text = []
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    text_in_one_string = ''
    for word in tokenized_text:
        lemmatized_text.append(lemmatizer.lemmatize(word))
    edited_text = []
    for word in lemmatized_text:
        if word in string.punctuation or word in nltk.corpus.stopwords.words('english'):
            pass
        elif nltk.pos_tag([word])[0][1] != 'NN':
            pass
        else:
            edited_text.append(word)
            text_in_one_string += word + ' '
    articles_dict[news[0].text] = text_in_one_string
    all_texts.append(text_in_one_string)
vectorizer.fit(all_texts)
for title, text in articles_dict.items():
    vector = vectorizer.transform([text])
    terms = vectorizer.get_feature_names_out()
    vector_array = vector.toarray()[0]
    max_ind = np.argsort(-1*vector_array)[:10]
    pairs = sorted(list(zip(vector_array[max_ind], terms[max_ind])), key=lambda x: (x[0], x[1]), reverse=True)
    print(title + ':')
    for i in range(5):
        print(pairs[i][1], end=' ')
    print()
    print()
