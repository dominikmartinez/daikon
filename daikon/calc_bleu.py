#!/usr/bin/python3
#-*- coding: utf-8 -*-

# Maschinelle Übersetzung
# Übung 1 - Aufgabe 1
# Dominik Martinez

"""This module calculates the BLEU-score for a hypothesis translation.

arguments to introduce in the command line:
1st: text file with the hypothesis translation
2nd: text file with the reference translation
3rd: the max. n-gram order (1 for unigrams, 2 for unigrams and bigrams...)
"""


import sys
from nltk.tokenize import word_tokenize
from math import exp


def tokenise(input):
    """Tokenise the input file and return a list of tokens
    
    arguments:
    1st: closed file-like object containing text
    
    return value: list of token strings
    """
    with open(input, "r") as input:
        raw_text = str()
        for line in input:      # Write the file into a string
            raw_text += line
        tokenized_text = word_tokenize(raw_text)        # Use the NLTK tool
        return tokenized_text


def n_grams(tokenized_text, n):
    """Calculate the n-grams (of nth order) of a list of tokens

    arguments:
    1st: the token list to compute
    2nd: the n-gram order
    
    return value: list of n-grams (a list of lists)
    """
    n_gram_list = list()
    for i in range(0, (len(tokenized_text)+1-n)):
        n_gram = tokenized_text[i:i+n]
        n_gram_list.append(n_gram)
    return n_gram_list


def n_gram_prec(hyp, ref, n):
    """Calculate the n-gram precision (of nth order) of a list of tokens

    arguments:
    1st: token list of the hypothesis translation
    2nd: token list of the reference translation
    3rd: the n-gram order
    
    return value: the precision for the introduced n-gram order
    """
    n_grams_hyp = n_grams(hyp, n)
    n_grams_ref = n_grams(ref, n)
    correct = 0
    for n_gram in n_grams_hyp:      # Count the number of hyp n-grams in ref
        if n_gram in n_grams_ref:
            correct += 1
            n_grams_ref.remove(n_gram)      # Make sure the clipping works
    return correct / len(n_grams_hyp)       # Precision

    
def calc_bleu(hyp, ref, n = 4):
    """Calculate the BLEU-score of the hypothesis translation

    arguments:
    1st: token list of the hypothesis translation
    2nd: token list of the reference translation
    3rd: the highest n-gram order (4 per default)
    
    return value: the BLEU-score as float number
    """
    ## Set the precision start value to 1.  For each number from 1 to n, 
    ## multiply the current p value with the current n-gram precision. 
    ## At the end, take the n-root of the current precision value to 
    ## calculate the unified n-gram precision. 
    p = int(1)
    for i in range(1, n+1):
        p *= n_gram_prec(hyp, ref, i)
    p = p ** (1/n)
    
    ## Calculate the brevity penalty, which is either 1, or, if smaller, 
    ## the number e raised to the power of division of the reference length
    ## and the hypothesis length. 
    bp = min(1.0, exp(1 - (len(ref)/len(hyp))))

    ## Return the BLEU-score, which is the product of the brevity penalty
    ## and the unified n-gram precision. Round the result.
    return round(bp * p, 3)


def main():
    # Tokenise the input files
    tok_hyp = tokenise(sys.argv[1])
    tok_ref = tokenise(sys.argv[2])
    
    ## Calculate the BLEU-score. Use the 3rd argument from the command line as 
    ## the max. n-gram order. If there is no 3rd argument, use 4 per default. 
    if len(sys.argv) == 4:
        print("BLEU-score: {}".format(calc_bleu(tok_hyp, tok_ref, \
        int(sys.argv[3]))))
    elif len(sys.argv) < 4:
        print("BLEU-score: {}".format(calc_bleu(tok_hyp, tok_ref)))        

if __name__ == "__main__":
    main()