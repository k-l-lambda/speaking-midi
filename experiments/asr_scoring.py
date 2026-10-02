"""Frozen raw CER/WER convention used throughout the poetry experiments."""
import re


def units(text, language):
    return [x for x in text if x.isalnum()] if language == 'Chinese' else re.findall(r'[a-z0-9]+', text.lower())


def distance(a, b):
    row = list(range(len(b)+1))
    for i, x in enumerate(a):
        current = [i+1]
        for j, y in enumerate(b):
            current.append(min(current[-1]+1, row[j+1]+1, row[j]+(x != y)))
        row = current
    return row[-1]


def score(reference, hypothesis, language):
    ref = units(reference, language)
    errors = distance(ref, units(hypothesis, language))
    return dict(errors=errors, reference_units=len(ref), error_rate=errors/len(ref))
