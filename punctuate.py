import os
from transformers import logging
from repunc import CasePuncPredictor


checkpoint_path = "models/vosk-recasepunc-en-0.22/checkpoint"

if not os.path.exists(checkpoint_path):
    print("Punctuation models: failed to find")
    exit(1)

logging.set_verbosity_error()
predictor = CasePuncPredictor(checkpoint_path, lang="en")


def punctuate_text(text):
    tokens = list(enumerate(predictor.tokenize(text)))

    results = ""
    for token, case_label, punc_label in predictor.predict(tokens, lambda x: x[1]):
        prediction = predictor.map_punc_label(
            predictor.map_case_label(token[1], case_label), punc_label)

        if token[1][0] == '\'' or (len(results) > 0 and results[-1] == '\''):
            results = results + prediction
        elif token[1][0] != '#':
            results = results + ' ' + prediction
        else:
            results = results + prediction

    return results.strip().replace(" - ", "-").replace(" .", ". ").replace(" ,", ",").replace(" !", "!").replace(" ?", "?")
