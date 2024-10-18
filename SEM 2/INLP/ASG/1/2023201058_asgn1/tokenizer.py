import re
# import sys
# import pandas as pd
# import numpy as np
# from numpy import c_, exp, log, inf, NaN, sqrt
# from tokenizer import Tokenizer
# from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt
# from scipy import linalg
# from collections import defaultdict
# from itertools import zip_longest
# import pylab
# from matplotlib import rc

class Tokenizer:
    def __init__(self):
        # Define regular expressions for different cases
        self.sentence_tokenizer = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s')
        self.word_tokenizer = re.compile(r'\w+|[^\w\s]')
        self.number_tokenizer = re.compile(r'\b\d+\b')
        self.mail_id_tokenizer = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.url_tokenizer = re.compile(r'(https?:\/\/|www\.)?\S+[a-zA-Z0-9]{2,}\.[a-zA-Z0-9]{2,}\S+\b(?!\.)?')
        self.hashtag_tokenizer = re.compile(r'#\w+')
        self.mention_tokenizer = re.compile(r'@\w+')

        # self.percentage_tokenizer = re.compile(r'\b\d+(\.\d+)?%\b')
        # self.age_tokenizer = re.compile(r'\b\d+\s?(?:years?|yrs?)\b')
        # self.time_expression_tokenizer = re.compile(r'\b(?:\d+\s?(?:hours?|mins?|minutes?|seconds?|secs?|AM|PM))\b')
        # self.time_period_tokenizer = re.compile(r'\b(?:\d+\s?(?:days?|weeks?|months?|years?))\b')

        # Define placeholders for substitution
        self.url_placeholder = '<URL>'
        self.hashtag_placeholder = '<HASHTAG>'
        self.mention_placeholder = '<MENTION>'
        self.number_placeholder = '<NUM>'
        self.mail_id_placeholder = '<MAILID>'

        # self.percentage_placeholder = '<PERCENTAGE>'
        # self.age_placeholder = '<AGE>'
        # self.time_expression_placeholder = '<TIME_EXPRESSION>'
        # self.time_period_placeholder = '<TIME_PERIOD>'

    def replace_tokens(self, text):
        # Replace tokens with appropriate placeholders
        # text = text.lower()
        text = self.url_tokenizer.sub(self.url_placeholder, text)
        text = self.mail_id_tokenizer.sub(self.mail_id_placeholder, text)
        text = self.hashtag_tokenizer.sub(self.hashtag_placeholder, text)
        text = self.mention_tokenizer.sub(self.mention_placeholder, text)

        # text = self.percentage_tokenizer.sub(self.percentage_placeholder, text)
        # text = self.age_tokenizer.sub(self.age_placeholder, text)
        # text = self.time_expression_tokenizer.sub(self.time_expression_placeholder, text)
        # text = self.time_period_tokenizer.sub(self.time_period_placeholder, text)

        text = self.number_tokenizer.sub(self.number_placeholder, text)
        return text

    def replace_symbols(self, text):
        text = re.sub(r"_(.*?)_", r"\1", text)
        text = re.sub(r'(Mr\.|Mrs\.|Ms\.)[a-zA-Z]*', '', text)
        text = re.sub(r'can\'t', r'can not', text)
        text = re.sub(r'won\'t', r'will not', text)
        text = re.sub(r'([a-zA-Z]+)n\'t', r'\1 not', text)
        text = re.sub(r'([a-zA-Z]+)\'s', r'\1 is', text)
        text = re.sub(r'([iI])\'m', r'\1 am', text)
        text = re.sub(r'([a-zA-Z]+)\'ve', r'\1 have', text)
        text = re.sub(r'([a-zA-Z]+)\'d', r'\1 had', text)
        text = re.sub(r'([a-zA-Z]+)\'ll', r'\1 will', text)
        text = re.sub(r'([a-zA-Z]+)\'re', r'\1 are', text)
        text = re.sub(r'([a-zA-Z]+)in\'', r'\1ing', text)
        # text = re.sub(r'([\*\-\#\%\!\"\$\&\'\(\)\+\,\-\/\:\;\=\?\@\[\\\]\^\_\‘\{\|\}\~])', r' \1 ', text)
        # text = re.sub(r'\s{2,}', r' ', text.strip())
        return text

    def tokenize_sentences(self, text):
        text = self.replace_tokens(text)
        text = self.replace_symbols(text)
        sentences = self.sentence_tokenizer.split(text)
        return [self.tokenize_words(sentence) for sentence in sentences]

    def tokenize_words(self, sentence):
        tokens = self.word_tokenizer.findall(sentence)
        return tokens

    @staticmethod
    def sentence_tokenizer(text):
        # Replace newline characters with spaces
        text = text.replace('\n', ' ')

        # Define the regular expression pattern for identifying sentence boundaries
        pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'

        # Split the text based on the pattern
        sentences = re.split(pattern, text)

        return sentences

def merges_in_list1(input_list):
    output_list = []

    for sublist in input_list:
        # i = 0
        # while i < len(sublist) - 1:
        #     if sublist[i] == "'" and i + 1 < len(sublist) and (sublist[i+1] == "re" or sublist[i+1] == "s" or sublist[i+1] == "t"): #isinstance(sublist[i + 1], str):
        #         merged_element = f"{sublist[i]}{sublist[i + 1]}"
        #         sublist[i:i+2] = [merged_element]
        #     else:
        #         i += 1

        i = 0
        while i < len(sublist):
            if sublist[i] == '<' and i + 2 < len(sublist) and sublist[i + 2] == '>':
                merged_element = ''.join(sublist[i:i+3])
                sublist[i:i+3] = [merged_element]
            else:
                i += 1
        # sublist.insert(0, '<SOS>')
        # if (sublist[-1] == '.'):
        #     sublist[-1] = '<EOS>'
        # else:
        #     sublist.append('<EOS>')

        output_list.append(sublist)

    return output_list

def merges_in_list2(input_list):
    output_list = []
    vocabulary = set()
    for sublist in input_list:
        subsublist = []
        for ele in sublist:
            if ele not in ['\\', '*', '-', '#', '%', '!', '"', '$', '&', '\'', '(', ')', '+', ',', '-', '.', '/', ':', ';', '=', '?', '@', '[', ']', '^', '_', '‘', '{', '|', '}', '~']:
                subsublist.append(ele.lower())
                # vocabulary.update(ele.lower())
                vocabulary.update([re.sub(r'\s', '', ele.lower())])
                # print(subsublist, vocabulary)
        # if(sublist == ['<SOS>', '<EOS>']):
        #     continue
        subsublist.insert(0, '<SOS>')
        subsublist.insert(0, '<SOS>')
        # if (sublist[-1] == '.'):
        #     sublist[-1] = '<EOS>'
        # else:
        #     sublist.append('<EOS>')
        subsublist.append('<EOS>')
        output_list.append(subsublist)
    return output_list, vocabulary

retokenized_sentences = [] #['<SOS>']

if __name__ == "__main__":
    # tokenizer = Tokenizer()
    # pre_tokenized_sentences = list()
    # corpus_path = 'PrideandPrejudice-JaneAusten.txt'  # Replace with the path to your file
    # with open(corpus_path, 'r') as fp:
    #     corpus_text = fp.read()
    # pre_tokenized_sentences = Tokenizer().tokenize_sentences(corpus_text)
    corpus_text = str(input())
    pre_tokenized_sentences = Tokenizer().tokenize_sentences(corpus_text)
    # for sentence in tokenized_sentences:
        # print(sentence)


    '''
    if __name__ == "__main__":
        corpus_path = '/content/CopyofPrideandPrejudice-JaneAusten.txt'
        with open(corpus_path, 'r') as fp:
            corpus_text = fp.readlines()
        # Example text
        # example_text = "In 'Pride and Prejudice' by Jane Austen, Elizabeth Bennett meets Mr Darcy at a ball hosted by her friend @charles_bingly. They don't dance, but Mr Darcy finds her behaviour 'tolerable, but not handsome enough to tempt him' #rude. She later visits Pemberley, Mr Darcy's estate, where she learns more about his character. Check out more information at https://janeausten.co.uk. then What about me!!!?"
        # example_text = "My email address (Mail ID) is john.doe@example.com, and I received an invoice with a total of $500. I can't believe it's already 2024! Time flies when you're having fun. #NewYear. @Alice, have you checked out the latest blog post on our website? The URL is www.example.com/latest-post. The meeting is scheduled for 3:30 PM, and we need to finalize the budget for the project. Just booked tickets for our vacation! The flight leaves at 8:45 AM from www.travelairlines.com. #Excited for the upcoming event! Don't forget to RSVP at event@example.com. The report indicates a growth rate of 15%, and we need to discuss the strategy moving forward. @Bob, can you send me the document at bob.smith@exampleco.com? The recipe for the delicious cake can be found on our blog: www.examplebakes.com/cake-recipe. Planning a weekend getaway! Any recommendations? #Travel."
        # example_text = "The sun sets in the west, casting a warm glow over the horizon. The horizon is painted with hues of orange and pink, creating a mesmerizing display. Displaying the beauty of nature, nature's wonders are truly captivating. Captivating our senses, the senses are heightened in this tranquil moment. Moments like these remind us of the simple pleasures, pleasures that bring joy to our hearts. Hearts that beat in harmony with the rhythm of the universe. The universe unfolds its secrets, secrets whispered in the gentle rustle of leaves. Leaves dance in the breeze, a dance that tells stories of ancient times. Times when the world was young and full of possibilities. Possibilities that stretch endlessly into the vast unknown. The unknown holds the promise of adventure, adventures waiting to be explored. Explore the depths of your imagination, imagination that knows no bounds. Bounds that only exist in the mind, a mind that is free to wander. Wander through the landscapes of dreams, dreams that take flight like birds in the sky. The sky is a canvas of dreams, dreams that paint a picture of endless possibilities. Possibilities that shimmer and sparkle like stars in the night sky. The night sky is a tapestry of dreams, dreams that weave together a story of infinite beauty. Beauty that transcends time and space, space where the trigrams of existence align in perfect harmony. Harmony that echoes through the trigrams, trigrams repeating in a cosmic dance. Dance to the rhythm of the trigrams, trigrams echoing through the vast expanse of the universe. The universe, a symphony of trigrams, trigrams that resonate with the heartbeat of creation. Creation unfolding in a continuous loop, a loop that echoes the trigrams of existence."
        # Create tokenizer instance
        # print(corpus_text)

        tokenizer = Tokenizer()

    for example_text in corpus_text:
        if example_text != "\n" or example_text != '':
            tok_sen = tokenizer.tokenize_sentences(example_text)
            # pt = tokenize(t)
            # temp = tok_sen.split()
            if len(tok_sen[0]) != 0:
                # print(len(tok_sen), end = ' ')
                # print(tok_sen)
                pre_tokenized_sentences += tok_sen
    '''
    # Tokenize sentences
        # tokenized_sentences.append()
    # Retokenize and output the result
    tokenized_sentences = merges_in_list1(pre_tokenized_sentences)
    print(tokenized_sentences)
    # vocabulary = set()



    retokenized_sentences, vocabulary = merges_in_list2(tokenized_sentences)
    # print(retokenized_sentences)
    # print(vocabulary)
    # Output tokenized text
    # for sentence in tokenized_sentences:
    #     print(sentence)
    # flattened_retokenized_sentences = []
    # for txt in retokenized_sentences:
    #     # if(txt == ['<SOS>', '<EOS>']):
    #     #     continue
    #     for ele in txt:
    #         if ele not in ['\\', '*', '-', '#', '%', '!', '"', '$', '&', '\'', '(', ')', '+', ',', '-', '.', '/', ':', ';', '=', '?', '@', '[', ']', '^', '_', '‘', '{', '|', '}', '~']:
    #             flattened_retokenized_sentences.append(ele)
    # flattened_retokenized_sentences.insert(0, '<SOS>')
    # flattened_retokenized_sentences.append('<EOS>')
    # flattened_retokenized_sentences.pop(2) # comment this
    # print(flattened_retokenized_sentences)

    # vocab = set()
    # for txt in flattened_retokenized_sentences:
    #     vocab.update([re.sub(r'\s', '', txt)])
    # sorted_vocab_set = list(sorted(vocab))
    # print(sorted_vocab_set)

    # from collections import defaultdict
    # vocab = defaultdict(int)
    # for txt in flattened_retokenized_sentences:
    #     vocab[txt] += 1
    # sorted_vocab_dict_freq = dict(sorted(vocab.items(), key=lambda x: x[0]))
    # print(sorted_vocab_dict_freq)