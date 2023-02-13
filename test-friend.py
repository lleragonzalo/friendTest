#name = input("Name: ")
#print(f"Hello, {name}")

"""n = int(input("Number: "))

if n > 0:
    print("n es positivo")
elif n < 0:
    print("ne es negativo")
else:
    print("n es cero")"""

"""import math
print(math.pi)



import datetime

now = datetime.datetime.now()
now.year

print(now)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score



# Carga tus datos de sentimiento con texto asociado
text_data = [("great feeling", "positive"),
             ("not a good feeling", "negative"),
             ("feeling awesome", "positive"),
             ("terrible feeling", "negative")]

# Separar los datos en dos listas: una para el texto y otra para las etiquetas de sentimiento
texts, labels = zip(*text_data)

# Convertir el texto en una representación numérica utilizando un vectorizador
vectorizer = CountVectorizer()
texts = vectorizer.fit_transform(texts)

# Convertir las etiquetas de sentimiento en números
labels = np.array([1 if label == "positive" else 0 for label in labels])

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)

# Entrenar un modelo de Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

def analyze_sentiment(review):
    sid = SentimentIntensityAnalyzer()
    sentiment = sid.polarity_scores(review)
    return sentiment

reviews = [
    "The movie was fantastic! I loved it.",
    "It was a terrible movie and I hated it.",
    "It was just okay, not really my thing.",
    "I was pleasantly surprised by how much I liked the movie.",
    "It was a boring movie, I fell asleep."
]

for review in reviews:
    sentiment = analyze_sentiment(review)
    if sentiment['compound'] >= 0.05:
        print("Positive")
    elif sentiment['compound'] <= -0.05:
        print("Negative")
    else:
        print("Neutral")

import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

short_pos = open("positive.txt","r").read()
short_neg = open("negative.txt","r").read()

# move this up here
all_words = []
documents = []

#  j is adject, r is adverb, and v is verb
#allowed_word_types = ["J","R","V"]
allowed_word_types = ["J"]

for p in short_pos.split('\n'):
    documents.append( (p, "pos") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for p in short_neg.split('\n'):
    documents.append( (p, "neg") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())



save_documents = open("pickled_algos/documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()


all_words = nltk.FreqDist(all_words)


word_features = list(all_words.keys())[:5000]


save_word_features = open("pickled_algos/word_features5k.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()


def find_features(document):
    words = word_tokenize(document)"""


"""def decide(issue):
    print("Usted está enfrentando el siguiente problema: ", issue)
    print("1. Ver los hechos disponibles")
    print("2. Consultar con un amigo o experto")
    print("3. Considerar sus valores y prioridades")
    print("4. Tomar una decisión informada")

    choice = int(input("Seleccione una opción (1-4): "))

    if choice == 1:
        print("Recopile toda la información relevante sobre el tema.")
    elif choice == 2:
        print("Hable con alguien en quien confíe para obtener una perspectiva diferente.")
    elif choice == 3:
        print("Piense en sus valores y prioridades, y cómo se relacionan con el problema.")
    elif choice == 4:
        print(
            "Tome una decisión informada y confiada basada en los hechos, las opiniones de otros y sus propios valores.")
    else:
        print("Opción inválida. Inténtalo de nuevo.")


issue = input("Introduzca el problema que desea resolver: ")
decide(issue)"""

print("Bienvenido al programa para determinar la calidad de una amistad.")

nombre = input("Por favor, introduzca el nombre de su amigo: ")

print("Hola", nombre + ",")

pregunta1 = input("¿Te apoya tu amigo incondicionalmente? (Sí/No): ")

pregunta2 = input("¿Comparte tu amigo tus intereses y pasatiempos? (Sí/No): ")

pregunta3 = input("¿Te escucha atentamente cuando tienes algo importante que decir? (Sí/No): ")

pregunta4 = input("¿Te defiende tu amigo cuando alguien te critica o te hace daño? (Sí/No): ")

pregunta5 = input("¿Te da tu amigo buenos consejos y te ayuda a resolver tus problemas? (Sí/No): ")

puntuacion = 0

if pregunta1 == "Sí":
    puntuacion += 1

if pregunta2 == "Sí":
    puntuacion += 1

if pregunta3 == "Sí":
    puntuacion += 1

if pregunta4 == "Sí":
    puntuacion += 1

if pregunta5 == "Sí":
    puntuacion += 1

if puntuacion >= 4:
    print("En base a tus respuestas, se puede decir que", nombre, "es un amigo verdadero.")
else:
    print("En base a tus respuestas, se puede decir que", nombre, "no es un amigo verdadero.")

