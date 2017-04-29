from tfidf import *
from perceptron import *
from naivesbayes import *
from intelli_grep import *
from naivebayes_improve1 import *
from naivebayes_improve2 import *
import sys

if __name__=="__main__":
    if(len(sys.argv) < 2):
        print("invalid format - To run, execute classifier_main.py [a|b|c|d|e|f]")
        print("a - intelligrep\nb - naive bayes\nc - naive bayes improved1\nd - naive bayes improved2\ne - perceptron\nf - tfidf\ne - improved nb\n")
        exit(0)

    method = str(sys.argv[1])
    if method == 'a':
        ig = Intelli_Grep()
        ig.classify()
    elif method == 'b':
        nb = NaiveBayes()
        nb.classify()
    elif method == 'c':
        nb = NaiveBayes_Improve1()
        nb.classify()
    elif method == 'd':
        nb = NaiveBayes_Improve2()
        nb.classify()
    elif method == 'e':
        p = Perceptron()
        p.classify()
    elif method == 'f':
        tfidf = TfIdfClassifier()
        tfidf.classify()
    else:
        print("invalid entry - To run, execute classifier_main.py [a|b|c|d|e|f]")
        print("a - intelligrep\nb - naive bayes\nc - naive bayes improved1\nd - naive bayes improved2\ne - perceptron\nf - tfidf\ne - improved nb\n")