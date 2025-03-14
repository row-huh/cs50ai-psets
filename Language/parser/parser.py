import nltk
import sys


TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""


NONTERMINALS = """
S -> NP VP | NP VP Conj NP VP | NP VP Conj VP
NP -> N | Det N | NP P NP | P NP | Det AP N 
VP -> V | Adv VP | V Adv | VP NP | V NP Adv
AP -> Adj | AP Adj
PP -> P NP | PP NP
"""


grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    
    sentence = sentence.lower()

    punkt = nltk.tokenize.punkt.PunktLanguageVars()
    words = punkt.word_tokenize(s=sentence)

    print("initial list of tokenized words", words)

    new_words = []

    for word in words:
        # confirm that there is atleast one alphabet in each
        flag = 0
        for letter in word:
            if letter.isalpha():
                flag = 1
        else:
            if flag == 0:
                print(f"{word} removed")
                continue    # do not add that word to the list
        
            if not(letter.isalpha()):
                #print("old word", word)
                #print("New word", word[:word.index(letter)])
                word = word[:word.index(letter)]

        new_words.append(word.lower())

    print("list before returning", new_words)
    return new_words


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    np = []

    for subtree in tree.subtrees():
        if subtree.label() == "NP":
            if not any(descendant.label() == "NP" for descendant in subtree.subtrees() if descendant != subtree):
                np.append(subtree)


    print("NP", np)
    return np


if __name__ == "__main__":
    main()