import pickle
import re

from nltk.corpus import words, wordnet
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

stemmer = StemmerFactory().create_stemmer()

bahasa_words = []
with open('models/all-id-words-v2.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        bahasa_words.append(line.strip())

with open('models/pre_language_identifications.pkl', 'rb') as f:
    pre_language_identifications = pickle.load(f)


def remove_duplication(token):
    new_token = ''
    prev_char = ''
    length = 0
    for char in token:
        if char != prev_char:
            new_token += char
            prev_char = char
            length = 1
        else:
            length += 1
            if length <= 2:
                new_token += char
    return new_token

def is_probably_phone_number(token):
    chars = re.sub('[0-9]', '', token)
    if len(set(chars)) == 1 and (chars[0].lower() == 'x'):
        return True
    return False

def is_probably_blackberry_pin(token):
    nums = re.sub('[a-zA-Z]', '', token)
    if len(token) == 8 and len(nums) >= 3:
        return True
    return False

def is_probably_emoticon(token):
    chars = re.sub('[a-zA-Z.,\']', '', token)
    return chars == token

def is_link(token):
    return token.startswith('http')

def is_word(token):
    if len(token) == 0:
        return False
    # filter not alpha numeric (this will filter out hashtag, mention and emoji too)
    if not token.isalnum():
        if not re.search('[a-zA-Z]', token):
            return False
    # filter number only token
    if token.isdigit():
        return False
    # filter phone number and blackberry pin
    if is_probably_phone_number(token) or is_probably_blackberry_pin(token):
        return False
    if is_link(token):
        return False
    # filter hashtag
    elif token[0] == '#':
        return False
    # filter mention
    elif token[0] == '@':
        return False
    elif is_link(token):
        return False
    elif is_probably_emoticon(token):
        return False
    return True

def is_bahasa(token):
    stemmed_token = stemmer.stem(token)
    if stemmed_token in bahasa_words:
        return True
    return False

def is_english(token):
    if token in words.words():
        return True
    if len(wordnet.synsets(token)) > 0:
        return True
    return False

def guess_lang(token):
    if is_bahasa(token):
        return 'id'
    if is_english(token):
        return 'en'
    if token in pre_language_identifications:
        return pre_language_identifications[token]
    return 'un'

def lcs(X , Y): 
    m = len(X) 
    n = len(Y) 
  
    DP = [[None]*(n+1) for i in range(m+1)] 
  
    for i in range(m+1): 
        for j in range(n+1): 
            if i == 0 or j == 0 : 
                DP[i][j] = 0
            elif X[i-1] == Y[j-1]: 
                DP[i][j] = DP[i-1][j-1]+1
            else: 
                DP[i][j] = max(DP[i-1][j] , DP[i][j-1])
    
    cur = DP[m][n]
    #backtrack
    lcs_ = [""] * (cur+1) 
    lcs_[cur] = "" 
  
    i = m 
    j = n 
    while i > 0 and j > 0: 
        if X[i-1] == Y[j-1]: 
            lcs_[cur-1] = X[i-1] 
            i-=1
            j-=1
            cur-=1
        elif DP[i-1][j] > DP[i][j-1]: 
            i-=1
        else: 
            j-=1
  
    return DP[m][n], lcs_

# code mixed index
def calculate_CMI(token_en, token_id, token_un):
    total_token = token_en + token_id + token_un
    if total_token == token_un:
        return 0
    return 100.0 * (1 - (max(token_en, token_id)/(total_token - token_un)))


def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-8s %s" % (weight, label, attr))
