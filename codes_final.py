import pandas as pd

# make all the comments string and make it, erase rows that comments are null value

def ingest_train():
    data = pd.read_csv('reviews_tex.csv')
    data = data[data.comments.isnull() == False]

    data = data[data['comments'].isnull() == False]
    data['comments'] = data['comments'].map(str)
    data.reset_index(inplace=True)
    data.drop('index', axis=1, inplace=True)
    return data

df_2818=ingest_train()

df_2818=df_2818[["comments"]]

# return the wordnet object value corresponding to the POS tag
from nltk.corpus import wordnet

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

#preprocessing~

import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer

def clean_text(text):
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return(text)

# clean text data
df_2818["df_2818_clean"] = df_2818["comments"].apply(lambda x: clean_text(x))

# add sentiment anaylsis columns
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#use vader to use lexicon of words
sid = SentimentIntensityAnalyzer()
df_2818["sentiments"] = df_2818["comments"].apply(lambda x: sid.polarity_scores(x))
df_2818 = pd.concat([df_2818.drop(['sentiments'], axis=1), df_2818['sentiments'].apply(pd.Series)], axis=1)

# add number of characters column
df_2818["nb_chars"] = df_2818["comments"].apply(lambda x: len(x))

# add number of words column
df_2818["nb_words"] = df_2818["comments"].apply(lambda x: len(x.split(" ")))

# create doc2vec vector columns
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(df_2818["df_2818_clean"].apply(lambda x: x.split(" ")))]

# train a Doc2Vec model with our text data
model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

# transform each document into a vector data
doc2vec_df = df_2818["df_2818_clean"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
df_2818 = pd.concat([df_2818, doc2vec_df], axis=1)

# add tf-idfs columns
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df = 10)
tfidf_result = tfidf.fit_transform(df_2818["df_2818_clean"]).toarray()
tfidf_df = pd.DataFrame(tfidf_result, columns = tfidf.get_feature_names())
tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
tfidf_df.index = df_2818.index
df_2818 = pd.concat([df_2818, tfidf_df], axis=1)

#wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color = 'white',
        max_words = 200,
        max_font_size = 40,
        scale = 3,
        random_state = 42
    ).generate(str(data))

    fig = plt.figure(1, figsize = (20, 20))
    plt.axis('off')
    if title:
        fig.suptitle(title, fontsize = 20)
        fig.subplots_adjust(top = 2.3)

    plt.imshow(wordcloud)
    plt.show()

# print wordcloud
show_wordcloud(df_2818["comments"])

# highest positive sentiment reviews (with more than 5 words)
pd.set_option('display.max_colwidth', -1)
df_2818[df_2818["nb_words"] >= 5].sort_values("pos", ascending = False)[["comments", "pos"]].head(10)

# lowest negative sentiment reviews (with more than 5 words)

pd.set_option('display.max_colwidth', -1)
df_2818[df_2818["nb_words"] >= 5].sort_values("neg", ascending = False)[["comments", "neg"]].head(10)
