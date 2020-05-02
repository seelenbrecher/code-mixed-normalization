# !pip install translate
from translate import Translator

class Translation:
    translator_id_en = None
    translator_en_id = None

    map_id_en = dict()
    map_en_id = dict()

    SENTENCE_SEPARATORS = ['.', '!', '?']

    def __init__(self):
        secret = '3afa669d2445493695e941fcf4d66dea'
        self.translator_id_en = Translator(
            provider='microsoft', from_lang='id',
            to_lang='en', secret_access_key=secret
        )
        self.translator_en_id = Translator(
            provider='microsoft', from_lang='en',
            to_lang='id', secret_access_key=secret
        )
    
    def _translate_id_en(self, token):
        if token in self.map_id_en:
            return self.map_id_en[token]
        try:
            res = self.translator_id_en.translate(token)
        except:
            res = token
        
        self.map_id_en[token] = res
        return res

    def _translate_en_id(self, token):
        if token in self.map_en_id:
            return self.map_en_id[token]
        try:
            res = self.translator_en_id.translate(token)
        except:
            res = token
        
        self.map_en_id[token] = res
        return res

    def _translate_sentence(self, tokens, langs):
        lang_id = langs.count('id')
        lang_en = langs.count('en')

        if lang_id >= lang_en:
            matrix_lang = 'id'
            embedded_lang = 'en'
        else:
            matrix_lang = 'en'
            embedded_lang = 'id'
        
        new_tokens = []
        for token, lang in zip(tokens, langs):
            if lang == matrix_lang:
                new_tokens.append(token)
            else:
                if lang == 'id':
                    # jika tidak bisa melakukan translasi, kembalikan token awal
                    new_tokens.append(self._translate_id_en(token))
                elif lang == 'en':
                    # jika tidak bisa melakukan translasi, kembalikan token awal
                    new_tokens.append(self._translate_en_id(token))
                else:
                    new_tokens.append(token)
        
        sentence = ' '.join(token for token in new_tokens)

        if matrix_lang == 'en':
            sentence = self._translate_en_id(sentence)
        
        return sentence
                

    def translate(self, contexts, langs):
        results = ''
        sentence_tokens = []
        sentence_langs = []
        for context, lang in zip(contexts, langs):
            # sudah satu kalimat
            if context in self.SENTENCE_SEPARATORS:
                translated_sentences = self._translate_sentence(
                    sentence_tokens, sentence_langs
                )
                if results != '':
                    results += ' '
                results += f'{translated_sentences}{context}'
                sentence_tokens = []
                sentence_langs = []
            else:
                sentence_tokens.append(context)
                sentence_langs.append(lang)
        if len(sentence_tokens) > 0:
            if results != '':
                results += ' '
            translated_sentences = self._translate_sentence(
                sentence_tokens, sentence_langs
            )
            results += f'{translated_sentences}'

        return results.lower()