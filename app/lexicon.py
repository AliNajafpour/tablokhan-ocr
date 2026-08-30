import csv

from .settings import LEXICON_PATH


def distance(left, right):
    if len(left) < len(right):
        return distance(right, left)
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, 1):
        current = [row]
        for column, right_char in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


class Lexicon:
    def __init__(self):
        self.phrases = []
        if LEXICON_PATH.exists():
            with LEXICON_PATH.open(encoding="utf-8-sig") as file:
                self.phrases = [
                    " ".join(row["3-gram"].split())
                    for row in csv.DictReader(file)
                    if row.get("3-gram")
                ]
        self.by_length = {}
        for phrase in self.phrases:
            self.by_length.setdefault(len(phrase), []).append(phrase)

    def correct(self, text):
        text = " ".join((text or "").split())
        if len(text) < 3:
            return text
        limit = max(1, min(5, round(len(text) * 0.22)))
        best, best_distance = text, limit + 1
        for length in range(max(1, len(text) - limit), len(text) + limit + 1):
            for candidate in self.by_length.get(length, ()):
                candidate_distance = distance(text, candidate)
                if candidate_distance < best_distance:
                    best, best_distance = candidate, candidate_distance
        return best if best_distance <= limit else text


lexicon = None


def correct_texts(texts):
    global lexicon
    if lexicon is None:
        lexicon = Lexicon()
    return [
        lexicon.correct(text) if sum("\u0600" <= char <= "\u06ff" for char in text) >= 2 else text
        for text in texts
    ]
