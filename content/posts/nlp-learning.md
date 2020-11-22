---
title: "Nlp Learning with tutorial and code (part 1)"
date: 2020-11-05T23:18:50+01:00
draft: false
authors: ["luanpham"]
tags:
    - NLP
    - Deep Learning
    - Machine Learning
categories:
    - Tutorial
---

We follow a NLP tutorial with code today ([this link](https://blog.insightdatascience.com/how-to-solve-90-of-nlp-problems-a-step-by-step-guide-fda605278e4e))

## Getting Data

We're using "Disasters in Social Media" dataset which consists  of tweet archive where 

> Over 10,000 tweets culled with a variety of 
> searches like “ablaze”, “quarantine”, and “pandemonium”, then 
> noted whether the tweet referred to a disaster event (as opposed 
> to a joke with the word or a movie review or something 
> non-disastrous).

The objective is to distinguish between tweets that signal disasters and "irrelevant" tweets.

## Process Data
#### 1. Show Data

A sample of the data:

```csv 
        text	choose_one	class_label
0	Just happened a terrible car crash	Relevant	1
1	Our Deeds are the Reason of this #earthquake M...	Relevant	1
2	Heard about #earthquake is different cities, s...	Relevant	1
3	there is a forest fire at spot pond, geese are...	Relevant	1
4	Forest fire near La Ronge Sask. Canada	Relevant	1
5	All residents asked to 'shelter in place' are ...	Relevant	1
6	13,000 people receive #wildfires evacuation or...	Relevant	1
7	Just got sent this photo from Ruby #Alaska as ...	Relevant	1
8	#RockyFire Update => California Hwy. 20 closed...	Relevant	1
9	Apocalypse lighting. #Spokane #wildfires	Relevant	1
```

#### 2. Clean Data

Firstly, we to clean some data and replace all the extranious tokens with empty string `""`

```python
# questions is the dataset loaded with pandas
def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    return df

questions = standardize_text(questions, "text")

questions.to_csv("clean_data.csv")
questions.head()
```

#### 3. Tokenize Data

```python
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

clean_questions["tokens"] = clean_questions["text"].apply(tokenizer.tokenize)
```

Here we tokenizer by words separated by space `r'\w+'`


## Actual Machine Learning :D

#### Bag of Words Count

The simplest approach we can start with is to use a bag of words model, and apply a logistic regression on top. A bag of words just associates an index to each word in our vocabulary, and embeds each sentence as a list of 0s, with a 1 at each index corresponding to a word present in the sentence.

```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def cv(data):
    count_vectorizer = CountVectorizer()
    emb = count_vectorizer.fit_transform(data)
    return emb, count_vectorizer

list_corpus = clean_questions["text"].tolist()
list_labels = clean_questions["class_label"].tolist()

X_train, X_test, y_train, y_test = train_test_split(list_corpus, 
                            list_labels, test_size=0.2, random_state=40)

X_train_counts, count_vectorizer = cv(X_train)
X_test_counts = count_vectorizer.transform(X_test)
```

`X_train_counts` is the new embeding with each word embeded in a vector of 15928 length (more info at this [link](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)). This approach totally ignores the order of words in our sentences.

```python
X_train_counts.get_shape()
(8687, 15928)
```

#### Visualize the embedding

Normally we need to reduce the dimension of the embedding to graph  

{{< img_resize URL="/images/nlp_embed_visualize.png" style="width: 550px;" caption="Bag of Words visualization" >}}

As shown in the image, there is no distinguishing features between the Irrelevant and Disaster class because the vector from Bag of Word algorithm was initialized randomly. 

#### Linear Classifier

Use a simple linear classifier as a baseline:

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg', 
                         multi_class='multinomial', n_jobs=-1, random_state=40)
clf.fit(X_train_counts, y_train)

y_predicted_counts = clf.predict(X_test_counts)
```

And use `sklearn.metrics` matric to measure the peformance: 

```python
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

def get_metrics(y_test, y_predicted):  
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted, pos_label=None,
                                    average='weighted')             
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None,
                              average='weighted')
    
    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')
    
    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1
```

We got accuracy of `0.761` and f1 score of `0.759` for a simple linear classifier. Not bad. This model is very simple to train and we can easily extract the most important features. 

#### Confusing matrix

One of the most useful tool is the confusing matrix where we can have in-depth vision of where the model got wrong.

{{< img_resize URL="/images/confusion_matrix.png" style="width: 550px;">}}

As we can see, the model predicts 25% false negative, which mean it predicts "Irrelevant" when in fact it was a "Disaster". False positive is not desirable in this problem because it gives the authority less time to prepare when in fact there is a disaster coming in. 

Some extra visualizations in the notebook, most notably most important words that the classifier uses to determine the class.

#### TFIDF Bag of Words

We use a TF-IDF score (Term Frequency, Inverse Document Frequency) on top of our Bag of Words model. TF-IDF weighs words by how rare they are in our dataset, discounting words that are too frequent and just add to the noise.

```python
def tfidf(data):
    tfidf_vectorizer = TfidfVectorizer()

    train = tfidf_vectorizer.fit_transform(data)

    return train, tfidf_vectorizer

X_train_tfidf, tfidf_vectorizer = tfidf(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
```

{{< img_resize URL="/images/tfidf.png" style="width: 600px;" caption="IF-IDF visualization" >}}

Here the result is better. By leaving out many common words, it is 
easier to separate the 2 groups. Now our linear classifier should be 
able to do a better job. New accuracy is 76%, which is not a big 
improvement from previous method.

{{< img_resize URL="/images/confusion_matrix_TFIDF.png" style="width: 550px;">}}

## Word2Vec

Due to small number of tweets, it is unlikely that our model will pick up  the semantic meaning of words. It means that some tweets are classfied differently even when they contain very similiar words. To rectify this problem, we use a Word2Vec to capture the closeness between words and use in our model.

{{< highlight python "linenos=true" >}}
def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, clean_questions, generate_missing=False):
    embeddings = clean_questions['tokens'].apply(lambda x: get_average_word2vec(x, vectors, 
                                                                                generate_missing=generate_missing))
    return list(embeddings)

embeddings = get_word2vec_embeddings(word2vec, clean_questions)
X_train_word2vec, X_test_word2vec, y_train_word2vec, y_test_word2vec = train_test_split(embeddings, list_labels, 
                                                                                        test_size=0.2, random_state=40)
{{< / highlight >}}

As we see in line 10, we do average of all word2vec vectors of each word. Just this simple calculation give us a quite good representation the tweets, even though order is not taken into consideration. We graph the new average word2vec as below

{{< img_resize URL="/images/word2vec.png" style="width: 600px;" caption="Average word2vec" >}}

And the confusion matrix, new method has almost identical accuracy but is still a sligh improvement over all previous methods. 

{{< img_resize URL="/images/word2vec_confusionmatrix.png" style="width: 600px;" caption="Average word2vec" >}}

## CNN for text classification

Convolutional Neural Network (CNN) model is better known in image classification but it can be used in text classification. Unlike the previous models where order is not taken into consideration, CNN can distinguish between "Paul eats plants" and "Plants eat Paul"

See more in part 2...
