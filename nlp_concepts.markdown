# Basic Natural Language Processing (NLP) Concepts

Below is an explanation of key NLP concepts: Tokenization, Lemmatization, Stemming, Part-of-Speech (POS) Tagging, Corpus, and Stopwords Removal. Each section includes a Python code example using popular libraries like `nltk` and `spacy`.

---

## 1. Tokenization
Tokenization is the process of breaking down text into smaller units called "tokens," which can be words, sentences, or characters. It’s the first step in text preprocessing.

- **Example**:  
  Text: "I love reading books"  
  Tokens: ["I", "love", "reading", "books"]

- **Python Code**:
```python
import nltk
nltk.download('punkt')  # Download the tokenizer models

text = "I love reading books."
tokens = nltk.word_tokenize(text)
print("Word Tokens:", tokens)

sentence_tokens = nltk.sent_tokenize(text)
print("Sentence Tokens:", sentence_tokens)
```

**Output**:
```
Word Tokens: ['I', 'love', 'reading', 'books', '.']
Sentence Tokens: ['I love reading books.']
```

---

## 2. Lemmatization
Lemmatization reduces words to their base or dictionary form (lemma) while considering the word's meaning and context.

- **Example**:  
  Words: "running", "ran", "runs"  
  Lemma: "run"

- **Python Code**:
```python
import spacy

nlp = spacy.load("en_core_web_sm")  # Load English model
text = "I am running and ran yesterday."
doc = nlp(text)

lemmas = [token.lemma_ for token in doc]
print("Lemmas:", lemmas)
```

**Output**:
```
Lemmas: ['I', 'be', 'run', 'and', 'run', 'yesterday', '.']
```

**Note**: You need to install `spacy` and download the model (`python -m spacy download en_core_web_sm`).

---

## 3. Stemming
Stemming reduces words to their root form by removing prefixes or suffixes, often using simple rules, without considering context.

- **Example**:  
  Words: "running", "runner", "ran"  
  Stem: "run"

- **Python Code**:
```python
from nltk.stem import PorterStemmer
nltk.download('punkt')

stemmer = PorterStemmer()
words = ["running", "runner", "ran"]
stems = [stemmer.stem(word) for word in words]
print("Stems:", stems)
```

**Output**:
```
Stems: ['run', 'runner', 'ran']
```

---

## 4. Part-of-Speech Tagging (POS)
POS tagging assigns a grammatical category (e.g., noun, verb, adjective) to each word in a sentence based on its context.

- **Example**:  
  Sentence: "The cat runs fast"  
  Tags: The (DET), cat (NOUN), runs (VERB), fast (ADV)

- **Python Code**:
```python
import nltk
nltk.download('averaged_perceptron_tagger')

text = "The cat runs fast"
tokens = nltk.word_tokenize(text)
pos_tags = nltk.pos_tag(tokens)
print("POS Tags:", pos_tags)
```

**Output**:
```
POS Tags: [('The', 'DT'), ('cat', 'NN'), ('runs', 'VBZ'), ('fast', 'RB')]
```

---

## 5. Corpus
A corpus is a large, structured collection of texts used for linguistic analysis or training machine learning models.

- **Example**:  
  A corpus could be a collection of news articles or tweets.

- **Python Code**:
```python
import nltk
nltk.download('gutenberg')

from nltk.corpus import gutenberg
corpus = gutenberg.words('austen-emma.txt')  # Access Jane Austen's Emma
print("First 10 words in the corpus:", corpus[:10])
```

**Output**:
```
First 10 words in the corpus: ['[', 'Emma', 'by', 'Jane', 'Austen', '1816', ']', 'CHAPTER', 'I', 'Emma']
```

---

## 6. Stopwords Removal
Stopwords are common words (e.g., "the", "is", "and") that are often removed because they carry little meaning in text analysis.

- **Example**:  
  Text: "I am going to the school"  
  After Stopwords Removal: ["going", "school"]

- **Python Code**:
```python
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
text = "I am going to the school"
tokens = nltk.word_tokenize(text)
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
print("Filtered Tokens:", filtered_tokens)
```

**Output**:
```
Filtered Tokens: ['going', 'school']
```

---

## Notes
- These processes are often combined in NLP pipelines to preprocess text for tasks like sentiment analysis or text classification.
- Ensure you have `nltk` and `spacy` installed (`pip install nltk spacy`).
- For `spacy`, download the English model as mentioned above.
- The examples use English text, but similar processes can be adapted for other languages with appropriate libraries or models.