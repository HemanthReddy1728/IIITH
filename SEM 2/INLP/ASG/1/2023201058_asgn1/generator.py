import json
import sys
from tokenizer import Tokenizer, merges_in_list1, merges_in_list2
from collections import defaultdict



if __name__ == "__main__":
    # tokenizer = Tokenizer()
    model = sys.argv[1]
    corpus_path = sys.argv[2]  # Replace with the path to your file
    k = int(sys.argv[3]) # no_of_top_candidates
    corpus_path = sys.argv[2][:-4]
    if(model == 'n'):
        # File path
        file_path_NS = f'{corpus_path}_NoSmoothingInterProbas.txt'
        # predictNextWordInSentence = 'you may copy it pride and pre'
        NoSmoothingProbas = defaultdict(int, {eval(k): v for k, v in json.load(open(file_path_NS)).items()})
        predictNextWordInSentence = str(input("input sentence: "))
        [sent_tkn], voc = merges_in_list2(merges_in_list1(Tokenizer().tokenize_sentences(predictNextWordInSentence)))
        # print(sent_tkn)
        [penultimateWord, ultimateWord] = sent_tkn[-3:-1]
        # print(penultimateWord, ultimateWord)
        possibleNextWords = defaultdict(int)
        for trigram in NoSmoothingProbas:
            if trigram[0] == penultimateWord and trigram[1] == ultimateWord:
                # print(trigram)
                possibleNextWords[trigram[2]] += NoSmoothingProbas[trigram]
        # print(possibleNextWords)

        total_sum = sum(possibleNextWords.values())
        for key in possibleNextWords:
            possibleNextWords[key] /= total_sum
        # print(possibleNextWords)
        print(dict(sorted(possibleNextWords.items(), key=lambda item: item[1], reverse=True)))

    elif(model == 'i'):
        # File path
        file_path_IP_tri = f'{corpus_path}_InterpolationProbasTrig.txt'
        # predictNextWordInSentence = 'you may copy it pride and pre'
        InterpolationProbasTrig = defaultdict(int, {eval(k): v for k, v in json.load(open(file_path_IP_tri)).items()})
        predictNextWordInSentence = str(input("input sentence: "))
        [sent_tkn], voc = merges_in_list2(merges_in_list1(Tokenizer().tokenize_sentences(predictNextWordInSentence)))
        # print(sent_tkn)
        [penultimateWord, ultimateWord] = sent_tkn[-3:-1]
        # print(penultimateWord, ultimateWord)

        possibleNextWords = defaultdict(int)
        for trigram in InterpolationProbasTrig:
            if trigram[0] == penultimateWord and trigram[1] == ultimateWord:
                # print(trigram)
                possibleNextWords[trigram[2]] += InterpolationProbasTrig[trigram]
        # print(possibleNextWords)

        total_sum = sum(possibleNextWords.values())
        for key in possibleNextWords:
            possibleNextWords[key] /= total_sum
        # print(possibleNextWords)
        print(dict(sorted(possibleNextWords.items(), key=lambda item: item[1], reverse=True)))

    elif(model == 'g'):
        # File path
        file_path_SGT = f'{corpus_path}_SimpleGoodTuringProbas.txt'
        # predictNextWordInSentence = 'you may copy it pride and pre'
        SimpleGoodTuringProbas = defaultdict(int, {eval(k): v for k, v in json.load(open(file_path_SGT)).items()})
        predictNextWordInSentence = str(input("input sentence: "))
        [sent_tkn], voc = merges_in_list2(merges_in_list1(Tokenizer().tokenize_sentences(predictNextWordInSentence)))
        # print(sent_tkn, voc)
        # lnt = len(sent_tkn)
        [penultimateWord, ultimateWord] = sent_tkn[-3:-1]
        # print(penultimateWord, ultimateWord)
        possibleNextWords = defaultdict(int)
        for trigram in SimpleGoodTuringProbas:
            if trigram[0] == penultimateWord and trigram[1] == ultimateWord:
                # print(trigram)
                possibleNextWords[trigram[2]] += SimpleGoodTuringProbas[trigram]
        # print(possibleNextWords)
        '''
        for bigram in bigram_counts:
            if bigram[0] == ultimateWord:
                # print(trigram)
                possibleNextWords[bigram[1]] += InterpolationProbasBig[bigram]
        # print(possibleNextWords)
        '''
        total_sum = sum(possibleNextWords.values())
        for key in possibleNextWords:
            possibleNextWords[key] /= total_sum
        # print(possibleNextWords)
        print(dict(sorted(possibleNextWords.items(), key=lambda item: item[1], reverse=True)))