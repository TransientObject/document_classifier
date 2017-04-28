from tfidf import *
from perceptron import *
from nb import *
from intelligrep import *
import sys

if __name__=="__main__":
    if(len(sys.argv) < 2):
        print("invalid format - To run, execute classifier_main.py [a|b|c|d|e]")
        print("a - intelligrep\nb - naive bayes\nc - perceptron\nd - tfidf\ne - improved nb\n")
        exit(0)

    method = str(sys.argv[1])
    if method == 'a':
        print ("method not implemented")
    elif method == 'b':
        print("method not implemented")
    elif method == 'c':
        p = Perceptron()
        p.classify()
    elif method == 'd':
        tfidf = TfIdfClassifier()
        tfidf.classify()
    elif method == 'e':
        print("method not implemented")