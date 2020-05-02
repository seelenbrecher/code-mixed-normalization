from nltk.stem import PorterStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from helper_function import is_word, remove_duplication


class Normalizer:
    id_mapper = None
    en_mapper = None
    id_words = None
    en_words = None

    def __init__(self, id_mapper, en_mapper, id_words, \
                 en_words, contracted_words_mapper):
        self.id_mapper = id_mapper
        self.en_mapper = en_mapper
        self.id_words = id_words
        self.en_words = en_words
        self.contracted_words_mapper = contracted_words_mapper
        self.en_stemmer = PorterStemmer()
        self.id_stemmer = StemmerFactory().create_stemmer()
    
    def _lookup_id(self, token):
        stemmed_token = self.id_stemmer.stem(token)
        if self.id_words and (stemmed_token in \
                              self.id_words or token in self.id_words):
            return token
        if self.id_mapper and token in self.id_mapper:
            return self.id_mapper[token]
        return None
    
    def _lookup_en(self, token):
        stemmed_token = self.en_stemmer.stem(token)
        if self.en_words and (stemmed_token in \
                                self.en_words or token in self.en_words):
            return token
        if self.en_mapper and token in self.en_mapper:
            return self.en_mapper[token]
        return None
    
    def _normalize_id(self, token):
        # mengatasi kasus nomor 6
        if '2' in token and token.index('2') != 0:
            unnormalized_singulars = token.split('2')
            norm_sing_1 = self._lookup_id(unnormalized_singulars[0])
            norm_sing_2 = self._lookup_id(unnormalized_singulars[1])
            if not norm_sing_1:
                norm_sing_1 = unnormalized_singulars[0]
            if not norm_sing_2:
                norm_sing_2 = unnormalized_singulars[1]
            res = f'{norm_sing_1}-{norm_sing_1}'
            if norm_sing_2:
                res += norm_sing_2
            return res

        return self._lookup_id(token)
    
    def _normalize_en(self, token):
        # Handle case #9, affix nge-
        if token.startswith('nge'):
            token = token[3:]
            return self._lookup_en(token)
        
        # Handle case #10, affix -nya
        elif token.endswith('nya'):
            token = token[:-3]
            norm_token = self._lookup_en(token)
            if norm_token is not None:
                return f'the {norm_token}'
            else:
                return f'the {token}'

        token = self._lookup_en(token)

        # Handle case #8, contracted words
        if token in self.contracted_words_mapper:
            token = self.contracted_words_mapper[token]
        
        return token
    
    def _normalize(self, word, lang):
        res = None
        if lang == 'id':
            res = self._normalize_id(word)
        if lang == 'en':
            res = self._normalize_en(word)
        if res:
            return res
        return None

    def normalize(self, token, lang='un'):
        if not is_word(token):
            return token

        token = remove_duplication(token).lower()

        # Handle multiple words
        if ' ' in token:
            unnormalized_singulars = token.split(' ')
            normalized_singulars = []
            for sing in unnormalized_singulars:
                norm = self.normalize(sing, lang)
                normalized_singulars.append(norm if norm else sing)
            
            if normalized_singulars[0] == normalized_singulars[1]:
                return '-'.join(normalized_singulars)
            else:
                return ' '.join(normalized_singulars)

        possible_words = [token]
        possible_words.extend(self._generate_all_words(token))

        for word in possible_words:
            res = self._normalize(word, lang)
            if res:
                return res
        
        return token

    
    def _generate_all_words(self, token):
        single_token = ''
        ids = []
        for idx, char in enumerate(token):
            if idx > 0 and token[idx-1] == char:
                ids.append(len(single_token) - 1)
            else:
                single_token += char
        
        possible_words = []
        max_iter = 1 << len(ids)
        len_ids = len(ids)

        for i in range(0, max_iter):
            bin_ids = '{:08b}'.format(i).lstrip('0')

            appeared_id = []

            for idx, bin_id in enumerate(bin_ids):
                if bin_id == '1':
                    appeared_id.append(ids[idx])
            
            word = ''

            for idx, char in enumerate(single_token):
                word += char
                if idx in appeared_id:
                    word += char
            
            possible_words.append(word)
        
        return possible_words
