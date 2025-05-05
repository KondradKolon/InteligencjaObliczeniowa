import text2emotion as te
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
negative_review = """
I would like to emphasize that I am a guest who has been to the hotel several times in the past and it has always been good.
The only thing that ruined my vacation this time was the poor attitude of the reception staff on the first day I arrived and until
I changed the room later, and until I shouted at them out of anger, they did not treat me seriously. At the beginning of the stay,
there were two people in the room (my daughter and I), and later my wife joined us, and when I arrived to get a room for three,
they gave me a smaller room than I had when we were two people! This is a room that is not suitable for three people and is next
to a noisy elevator, and I explained to them that it does not make sense for a room for three people to be small and/or the same
as a room for two people. I would like to emphasize that I paid for a room for three, and until there was anger, frustration,
and anger, they did not fix the matter. The disdain of the reception staff, and this is what ruined the vacation!
A guest should not arrive and be angry! It should be noted that there was a large, bald receptionist who was particularly rude
and impudent who said "these rooms we have are not suitable for you, don't come and stay with us next time" with an annoying
smile and that is a rude and bad answer, other than that all the other staff at the hotel were fine. Also, the sofa that opens
for the third person in the room is not comfortable at all and needs improvement. I am writing this for only one reason so that
such unpleasantness does not happen to future guests and such an incident and attitude raises questions for me if I will come to
this hotel in the future
"""

positive_review = """
Everything is new, beautiful views of the city and very conveniently located. Close to the metro, train but you can also walk anywhere within the city center. The hotel also has a very nice bar, a rooftop bar and a lovely restaurant.
In my review I’ve added a panoramic photo of the view from my hotel room on the 15th floor."""


# Pobranie zasobów VADER (wykonaj raz)
nltk.download('vader_lexicon')

# Inicjalizacja analizatora
sia = SentimentIntensityAnalyzer()


def analyze_with_vader(text):
    scores = sia.polarity_scores(text)
    print(f"VADER Scores: {scores}")
    return scores


# pip install emoji==1.7.0
# Analiza opinii
print("--- Pozytywna opinia ---")
vader_pos = analyze_with_vader(positive_review)

print("\n--- Negatywna opinia ---")
vader_neg = analyze_with_vader(negative_review)


print("text-to-emotion")
print("--- Negatywna ---")
a = te.get_emotion(negative_review)
print(a)
print("--- pozytywna ---")
b = te.get_emotion(positive_review)
print(b)

#wyniki nie sa zgodne z oczekiwaniami
