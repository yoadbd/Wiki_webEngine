# Wiki_webEngine
a webengine of 7m docs of wikipedia

first of all we understood the structure of the project; the big parts we need to combine together. Then we scanned the practical assignments we did and tried to understand which of those scripts can be relevant to our own project, in addition to understand the code deeply. then we built a small corpus out of the train queries we got 1180~ docs.
all the evaluation and testing we did on the small corpus for then and on, and then validated on the big corpus.
Our first big idea was to make an index of tfidf instead of tf at index_body, it will save us calculation time at the search() function. 
we created it the same way we did in assignment3, but changed to calculation to tfidf

search_anchor():
before rendering the information we changed it to the project's requirements, other is the same as assignment3

Search_body():
used cosine_sim and normalizing by max_term

Search():
our implementation was based on the idea of learning rate.
At first, we gave the body a lot of weight, around 70% of probability, and the titles and anchor got the rest. Every time an answer is appended if the probability of it is chosen, the algorithm reduces the probability of it picking next time and raising the other’s probabilities, up to a limit ( lower bound )

