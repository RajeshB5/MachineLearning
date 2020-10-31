import string
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer

text = open("Read.txt", encoding="utf-8").read()
lower_text = text.lower()
clean_text = lower_text.translate(str.maketrans('', '', string.punctuation))
tokenized_text = word_tokenize(clean_text, "english")

final_words = []
for word in tokenized_text:
    if word not in stopwords.words("english"):
        final_words.append(word)

lemma_words = []
for word in final_words:
    word = WordNetLemmatizer().lemmatize(word)
    lemma_words.append(word)

emotion_list = []
with open('emotions.txt', 'r') as file:
    for line in file:
        clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
        word, emotion = clear_line.split(':')

        if word in lemma_words:
            emotion_list.append(emotion)

w = Counter(emotion_list)

def sentiment_analyse(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    if score['neg'] > score['pos']:
        print("Negative Sentiment")
        img = mpimg.imread('DataSets/sad.jpg')
    elif score['neg'] < score['pos']:
        print("Positive Sentiment")
        img = mpimg.imread('DataSets/happy.jpg')
    else:
        print("Neutral Sentiment")
        img = mpimg.imread('DataSets/neutral.jpg')
    return img

img = sentiment_analyse(clean_text)
plt.imshow(img)
fig, ax1 = plt.subplots()

ax1.bar(w.keys(), w.values())
fig.autofmt_xdate()
plt.savefig('graph.png')
plt.show()
