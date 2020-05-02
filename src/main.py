import pickle
import json

from src.tokenization import Tokenizer
from src.language_identification import LanguageIdentifier
from src.normalization import Normalizer
from src.translation import Translation

def normalize(tweet):
    with open('models/tokenizer_model-v1.pkl', 'rb') as f:
        tokenizer_model = pickle.load(f)

    with open('models/lang-identifier_model-v1.pkl', 'rb') as f:
        language_identifier_model = pickle.load(f)
    
    id_dict = []
    with open(f'models/all-id-words-v2.txt') as f:
        lines = f.readlines()
        for line in lines:
            id_dict.append(line.strip())
    en_dict = []
    with open(f'models/all-en-words-v1.txt') as f:
        lines = f.readlines()
        for line in lines:
            en_dict.append(line.strip())
    with open(f'models/mapping-unformal-words-en-v3.json') as f:
        en_word_mapper = json.load(f)
    with open(f'models/mapping-unformal-words-id-v3.json') as f:
        id_word_mapper = json.load(f)
    with open(f'models/mapping-unformal-words-combine-v3.json') as f:
        combine_word_mapper = json.load(f)
    with open(f'models/contracted-words-en-v1.json') as f:
        contracted_words_mapper = json.load(f)

    
    tokenizer = Tokenizer()
    tokenizer.model = tokenizer_model
    _, tokens = tokenizer.predict([tweet])
    tokens = tokens[0]

    language_identifier = LanguageIdentifier()
    language_identifier.model = language_identifier_model
    langs = language_identifier.predict([tokens])
    langs = langs[0]

    normalizer = Normalizer(id_word_mapper, en_word_mapper, id_dict, en_dict, contracted_words_mapper)

    norm_tokens = []

    for token, lang in zip(tokens, langs):
        norm_token = normalizer.normalize(token, lang)
        norm_tokens.append(norm_token)

    translation = Translation()
    res = translation.translate(norm_tokens, langs)

    return res
    