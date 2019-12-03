# Pretrained Model

Available Pretrained-Model:

| Type          | Model         |
| ------------- |:-------------:|
| Word Embedding | 1.6M-english-tweet-word-vectors |
| Word Embedding | 2.6M-combine-tweet-word-vectors      | 
| Word Embedding | 900k-english-tweet-word-vectors      |

Link to downlaod : https://drive.google.com/drive/folders/1fLq6jKXtOsu4dan1_cwzFyTJnTQ0mO4V?usp=sharing

Each model consists of more than a part. To use it
* First download all parts for each models.
* Put all the parts in the same directory.
* Then run this code
```
from gensim.models import Word2Vec
word2vec_model = Word2Vec.load('<path>/2.6M-combine-tweet-word-vectors')
```