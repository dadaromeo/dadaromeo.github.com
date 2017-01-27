.. title: The Voice Afrique Tweets Mining Part 3
.. slug: the-voice-afrique-tweets-mining-part-3
.. date: 2016-11-08 14:10:58 UTC+01:00
.. tags: text mining, topic modeling, text processing, gensim
.. category: 
.. link: 
.. description: 
.. type: text

Topic modeling
--------------
`Previously <https://dadaromeo.github.io/the-voice-tweets-mining-part-2>`_, we 
explored a model that exploits the links between the entities to help us find the 
key players in the data. Here, we will focus on the tweet’s text to better 
understand what the users are talking about. We move away from the network model 
we’ve used previously and discuss other methods for text analysis. We first 
explore `topic modeling <https://en.wikipedia.org/wiki/Topic_model>`_, an 
approach that finds natural topics within the text. We then move on to 
`sentiment analysis <https://dadaromeo.github.io/the-voice-afrique-tweets-mining-part-4>`_, 
the practice of associating a document with a sentiment score

.. TEASER_END

Finding topics
~~~~~~~~~~~~~~
The data we collected from Twitter is a relatively small sample, but attempting 
to read each individual tweet is a hopeless cause. A more reachable goal is to get 
a high-level understanding of what users are talking about. One way to do this is 
by understanding the topics the users are discussing in their tweets. In this 
section we discuss the automatic discovery of topics in the text through *topic modeling* 
with `Latent Dirichlet allocation <https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation>`_ 
(LDA), a popular topic modeling algorithm.

Every topic in LDA is a collection of words. Each topic contains all of the words 
in the corpus with a probability of the word belonging to that topic. So, while all 
of the words in the topic are the same, the weight they are given differs between topics.

LDA finds the most probable words for a topic, associating each topic with a theme is left to the user.

LDA with Gensim
~~~~~~~~~~~~~~~
To perfom the LDA computation in Python, we will use the :code:`gensim` library 
(`topic modeling for human <https://radimrehurek.com/gensim/>`_). As we can see, 
most of the work is done for us, the real effort is in the preprocessing of the 
documents to get the documents ready. The preprocessing we will perfom are:

-   **Lowercasing** - Strip casing of all words in the document 
    (i.e: :code:`"@thevoiceafrique #TheVoiceAfrique est SUPERB! :) https://t.co/2ty"` 
    becomes :code:`"#thevoiceafrique est superb! :) https://t.co/2ty"`)


-   **Tokenizing** - Convert the string to a list of tokens based on whitespace. 
    This process also removes punctuation marks from the text. This becomes the list 
    :code:`["@thevoiceafrique", "#thevoiceafrique", "est" "superb", ":)", "https://t.co/2ty"]`


-   **Stopword Removal** - Remove *stopwords*, words so common that their presence 
    does not tell us anything about the dataset. This also removes smileys, emoticons, 
    mentions hashtags and links: :code:`["@thevoiceafrique", "#thevoiceafrique", "superb"]`

.. code-block:: python
    
    import re
    import string
    import numpy as np
    import emoji
    from twitter.parse_tweet import Emoticons
    from pymongo import MongoClient
    from nltk.corpus import stopwords
    from nltk.tokenize import TweetTokenizer
    from gensim.models import LdaModel
    from gensim.corpora import TextCorpus
    
    np.random.seed(42)
    
    host = "localhost"
    port = 27017
    
    db = MongoClient(host, port).search

The :code:`stopwords-fr.txt` file is downloaded `here <https://github.com/stopwords-iso/stopwords-fr>`_.

.. code-block:: python
    
    stop_words = set()
    stop_words.update(list(string.punctuation))
    stop_words.update(stopwords.words("french"))
    stop_words.update(Emoticons.POSITIVE)
    stop_words.update(Emoticons.NEGATIVE)
    stop_words.update(["’", "…", "ca", "°", "çà", "»", "«", "•", "the",
                       "voice", "afrique", "voix", "–", "::", "“", "₩", "🤣"])
    
    with open("data/stopwords-fr.txt") as f:
        stop_words.update(map(str.strip, f.readlines()))
    
    tokenize = TweetTokenizer().tokenize

Little helpers

.. code-block:: python
    
    def parse(text):
        
        text = text.strip()
        text = text.strip("...")
        found = emoji.demojize(text).split(" ")
        text = " ".join([t for t in found if not("_" in t)])
        text = " ".join(re.split(r"\w*\d+\w*", text)).strip()
        tokens = tokenize(text)
        
        for token in tokens:
            cond = (token.startswith(("#", "@", "http", "www")) or
                    "." in token or
                    "'" in token
                    )
                
            if not(cond):
                yield token
    
    def preprocess(text):
        text = text.lower()
        for token in parse(text):
            if not(token in stop_words):
                yield token
    
    class Corpus(TextCorpus):
        
        def __len__(self):
            return len(self.input)
        
        def get_texts(self):
            for tweet in self.input:
                tweet = preprocess(tweet)
                yield list(tweet)

Load the tweets.

.. code-block:: python
    
    tweets = [tweet["text"] for tweet in db.thevoice.find() if not("retweeted_status" in tweet.keys())]

Enrich the stopwords set.

.. code-block:: python
    
    regexp = emoji.get_emoji_regexp().findall
    
    for tweet in tweets:
        stop_words.update(regexp(tweet))

Build the corpus.

.. code-block:: python
    
    corpus = Corpus(tweets)
    
    print("Number of documents: {}\nNumber of tokens: {}".format(len(corpus), len(corpus.dictionary)))

Build the model.

.. code-block:: python
    
    lda = LdaModel(corpus, num_topics=5, id2word=corpus.dictionary)

A helper for printing the topics

.. code-block:: python
    
    def show_topics(n=5, n_words=10, fmt="simple"):
        """Show `n` randomly selected topics and thier
        top words.
        """
        from tabulate import tabulate
        
        topics = {}
        ids = np.arange(lda.num_topics)
        ids = np.random.choice(ids, n, replace=False)
        for i in ids:
            topic = lda.show_topic(i, n_words)
            words,prop = zip(*topic)
            topics[i+1] = list(words)
        
        tabular = tabulate(topics, headers="keys", tablefmt=fmt)
        
        print(tabular)

Show the topics

.. code-block:: python
    
    show_topics()

+---------+---------+----------+---------+--------+
|1        |   2     |    3     |     4   |      5 |
+=========+=========+==========+=========+========+
|singuila |   gars  |  chante  |  asalfo |   fire |
+---------+---------+----------+---------+--------+
|coachs   |   lokua |    nadia |shayden  | famille|
+---------+---------+----------+---------+--------+
|chante   |charlotte| pub      | singuila|  faut  |
+---------+---------+----------+---------+--------+
|lol      |  go     | chanson  |  grâce  |   vrai |
+---------+---------+----------+---------+--------+
|congolais|  soir   |   grace  |   deh   |retourne|
+---------+---------+----------+---------+--------+
|asalfo   |asalfo   | choix    | belle   |  faire |
+---------+---------+----------+---------+--------+
|charlotte| super   |candidats | talent  |  pro   |
+---------+---------+----------+---------+--------+
|talent   | déjà    | belle    | soir    |  coach |
+---------+---------+----------+---------+--------+
|albert   | ndem    | heroine  |  ans    | nadia  |
+---------+---------+----------+---------+--------+
|frère    | chante  | soirée   | soeur   |  gars  |
+---------+---------+----------+---------+--------+

The table above show the distribution of words within the different topics. From 
that, we can see that viewers are talking about the different candidates and 
coaches. In the `next <https://dadaromeo.github.io/the-voice-afrique-tweets-mining-part-3>`_ 
post, we will use *Sentiment Analysis* to see if we see what sentiment is the most 
present in the data.

Thanks for following.