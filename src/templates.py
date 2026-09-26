"""Cloze-style (left-to-right) templates for the 16 PopQA relations.

Each template ends right before the object, so the object is the model's
continuation. `{s}` is replaced by the subject's surface form.
"""

TEMPLATES = {
    "occupation":     "{s}'s occupation is",
    "place of birth": "{s} was born in",
    "genre":          "The genre of {s} is",
    "father":         "The father of {s} is",
    "mother":         "The mother of {s} is",
    "country":        "{s} is located in the country of",
    "producer":       "{s} was produced by",
    "director":       "{s} was directed by",
    "screenwriter":   "The screenplay of {s} was written by",
    "composer":       "The music of {s} was composed by",
    "author":         "The author of {s} is",
    "capital of":     "{s} is the capital of",
    "capital":        "The capital of {s} is",
    "color":          "The color of {s} is",
    "religion":       "The religion of {s} is",
    "sport":          "{s} plays the sport of",
}
