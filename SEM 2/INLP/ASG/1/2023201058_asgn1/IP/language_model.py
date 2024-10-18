import random
import json
import os
import sys

from numpy import c_, exp, argmax, log, sqrt
from tokenizer import Tokenizer, merges_in_list1, merges_in_list2
from scipy import linalg
from collections import defaultdict

def create_n_grams(n, token_sets):

    n_grams = []
    n_gram_counts = defaultdict(int)
    if(n == 3):
        for tes in token_sets:
            for i in range(len(tes) - 2):
                n_gram = (tes[i], tes[i + 1], tes[i + 2])
                n_grams.append(n_gram)
                n_gram_counts[n_gram] += 1
    elif(n == 2):
        for tes in token_sets:
            tes = tes[1:]
            for i in range(len(tes) - 1):
                n_gram = (tes[i], tes[i + 1])
                n_grams.append(n_gram)
                n_gram_counts[n_gram] += 1
    elif(n == 1):
        # tokens = [x for x in tokens if x not in ['<SOS>', '<EOS>']]
        for tes in token_sets:
            tes = tes[1:-1]
            for i in range(len(tes)):
                n_gram = (tes[i])
                n_grams.append(n_gram)
                n_gram_counts[n_gram] += 1
    return n_grams, n_gram_counts



def simpleGoodTuringProbs(counts, confidenceLevel=1.96):
    if 0 in counts.values():
        raise ValueError('Species must not have 0 count.')
    
    totalCounts = sum(counts.values())   # N (G&S)
    # print(counts.values()) # N (G&S))
    # print("totcnt",totalCounts)
    # countsOfCounts = countOfCountsTable(counts) # r -> n (G&S)

    countsOfCounts = {}
    for c in counts.values():
        countsOfCounts[c] = countsOfCounts.get(c,0)+1
        # countsOfCounts[c] = 0
        # for species, speciesCount in counts.items():
        #     if speciesCount == c:
        #         countsOfCounts[c] += 1

    sortedCounts = sorted(countsOfCounts.keys())
    # print(totalCounts)
    # print("totcnt",totalCounts)
    # print(sum([r*n for r,n in countsOfCounts.items()]))
    assert(totalCounts == sum([r*n for r,n in countsOfCounts.items()]))


    sgtProbs = {}
    sgtProbs[('<OOV>', '<OOV>', '<OOV>')] = countsOfCounts[1] / totalCounts
    # print('p0 =', p0)

    # Z = __sgtZ(sortedCounts, countsOfCounts)
    Z = {}
    for jIdx, j in enumerate(sortedCounts):
        i = sortedCounts[jIdx-1] if jIdx != 0 else 0
        k = 2*j - i if jIdx == len(sortedCounts)-1 else sortedCounts[jIdx+1]
        Z[j] = 2*countsOfCounts[j] / float(k-i)

    # Compute a loglinear regression of Z[r] on r
    rs = list(Z.keys())
    zs = list(Z.values())

    # a, b = __loglinregression(rs, zs)
    a, b = linalg.lstsq(c_[log(rs), (1,)*len(rs)], log(zs))[0]
    # print('Regression: log(z) = %f*log(r) + %f' % (a,b))
    # if a > -1:
        # print('Warning: slope is > -1')    

    # Gale and Sampson's (1995/2001) "simple" loglinear smoothing method.
    rSmoothed = {}
    useY = False
    for r in sortedCounts:
        # y is the loglinear smoothing
        y = (r+1) * exp(a*log(r+1) + b) / exp(a*log(r) + b)

        # If we've already started using y as the estimate for r, then
        # continue doing so; also start doing so if no species was observed
        # with count r+1.
        if r+1 not in countsOfCounts:
            # if not useY:
                # print('Warning: reached unobserved count before crossing the smoothing threshold.')
            useY = True

        if useY:
            rSmoothed[r] = y
            continue
        
        # x is the empirical Turing estimate for r
        Nr = countsOfCounts[r]
        Nr1 = countsOfCounts[r+1]
        x = (r+1) * Nr1/Nr

        # t is the width of the 95% (or whatever) confidence interval of the
        # empirical Turing estimate, assuming independence.
        # t = confidenceLevel * sqrt((r+1)**2 * (Nr1 / Nr**2) * (1 + (Nr1 / Nr)))
        t = confidenceLevel * abs(x) * sqrt(1/Nr1 + 1/Nr)

        # If the difference between x and y is more than t, then the empirical
        # Turing estimate x tends to be more accurate. Otherwise, use the
        # loglinear smoothed value y.
        if abs(x - y) > t:
            rSmoothed[r] = x
        else:
            rSmoothed[r] = y
        useY = True

    # normalize and return the resulting smoothed probabilities, less the
    # estimated probability mass of unseen species.
    smoothTot = 0
    for r, rSmooth in rSmoothed.items():
        smoothTot += countsOfCounts[r] * rSmooth
    for species, spCount in counts.items():
        sgtProbs[species] = (1 - sgtProbs[('<OOV>', '<OOV>', '<OOV>')]) * (rSmoothed[spCount] / smoothTot)
    
    return sgtProbs





if __name__ == "__main__":
    # tokenizer = Tokenizer()
    model = sys.argv[1]
    corpus_path = sys.argv[2]  # Replace with the path to your file
    # if not os.path.exists(corpus_path):
    pre_tokenized_sentences = list()
    with open(corpus_path, 'r') as fp:
        corpus_text = fp.read()
    pre_tokenized_sentences = Tokenizer().tokenize_sentences(corpus_text)
    corpus_path = sys.argv[2][:-4]
    # for sentence in tokenized_sentences:
        # print(sentence)
    
    if(corpus_path == './PrideandPrejudice-JaneAusten' and model == 'g'):
        file_name1 = '2023201058_LM1_'
    elif(corpus_path == './PrideandPrejudice-JaneAusten' and model == 'i'):
        file_name1 = '2023201058_LM2_'
    elif(corpus_path == './Ulysses-JamesJoyce' and model == 'g'):
        file_name1 = '2023201058_LM3_'
    elif(corpus_path == './Ulysses-JamesJoyce' and model == 'i'):
        file_name1 = '2023201058_LM4_'
    else:
        file_name1 = '2023201058_LM5_'
    
    file_name2 = '-Perplexity.txt'

    # Tokenize sentences
        # tokenized_sentences.append()
    # Retokenize and output the result
    tokenized_sentences = merges_in_list1(pre_tokenized_sentences)
    # print(tokenized_sentences)
    # vocabulary = set()
    
    # Remove 1000 random lists
    test_tokenized_sentences = []


    
    
    ############################################################################################################
    

    
    # print(frequenciesOfFrequencies)

    if(model == 'n'):
        file_path_NS = f'{corpus_path}_NoSmoothingProbas.txt'
        
        for _ in range(1000):
            index = random.randint(0, len(tokenized_sentences) - 1)
            test_tokenized_sentences.append(tokenized_sentences.pop(index))

        # Now 'tokenized_sentences' contains the remaining lists
        train_tokenized_sentences = tokenized_sentences

        # print("Removed Lists:")
        # print(test_tokenized_sentences)

        # print("\nRemaining Lists:")
        # print(train_tokenized_sentences)


        train_retokenized_sentences, train_vocabulary = merges_in_list2(train_tokenized_sentences)
        test_retokenized_sentences, test_vocabulary = merges_in_list2(test_tokenized_sentences)
        

        
        # print(retokenized_sentences)
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
        # Example usage:
        train_token_sets = train_retokenized_sentences

        n = 1
        unigrams, unigram_counts = create_n_grams(n, train_token_sets)
        # print(unigrams)
        # print(len(unigrams))
        # print(unigram_counts)
        # print(len(unigram_counts))
            # File path
        file_path_UnigCnt = f'{corpus_path}_ns_unigram_counts.txt'

        # Write dictionary to file
        with open(file_path_UnigCnt, 'w') as file:
            json.dump({str(key): value for key, value in unigram_counts.items()}, file)

        n = 2
        bigrams, bigram_counts = create_n_grams(n, train_token_sets)
        # print(bigrams)
        # print(len(bigrams))
        # print(bigram_counts)
        # print(len(bigram_counts))
        # File path
        file_pathBigCnt = f'{corpus_path}_ns_bigram_counts.txt'

        # Write dictionary to file
        with open(file_pathBigCnt, 'w') as file:
            json.dump({str(key): value for key, value in bigram_counts.items()}, file)

        n = 3
        trigrams, trigram_counts = create_n_grams(n, train_token_sets)
        # print(trigrams)
        # print(len(trigrams))
        # print(trigram_counts)
        # print(len(trigram_counts))

        # File path
        file_path_TrigCnt = f'{corpus_path}_ns_trigram_counts.txt'

        # Write dictionary to file
        with open(file_path_TrigCnt, 'w') as file:
            json.dump({str(key): value for key, value in trigram_counts.items()}, file)

        if not os.path.exists(file_path_NS) or os.path.getsize(file_path_NS) == 0:
            # trigram_counts = json.load(open(file_path_TrigCnt), 'r')
            trigram_counts = defaultdict(int, {eval(k): v for k, v in json.load(open(file_path_TrigCnt)).items()})
            NoSmoothingProbas = defaultdict(int)
            for trigram in trigram_counts:
                if trigram[0:2] != ('<SOS>', '<SOS>'):
                # print(key[0:2])
                    NoSmoothingProbas[trigram] = trigram_counts[trigram]/bigram_counts[trigram[0:2]]
                else:
                    NoSmoothingProbas[trigram] = trigram_counts[trigram]/bigram_counts[trigram[1:3]]
            # tot_prob=sum(NoSmoothingProbas.values())
            # print(NoSmoothingProbas)
            # for key in NoSmoothingProbas:
                # NoSmoothingProbas[key]=NoSmoothingProbas[key]/tot_prob
            # print(NoSmoothingProbas)

            # File path

            # Write dictionary to file
            with open(file_path_NS, 'w') as file:
                json.dump({str(key): value for key, value in NoSmoothingProbas.items()}, file)            

            # # File path
            # file_path = "dictionary.txt"

            # # Write dictionary to file
            # with open(file_path, 'w') as file:
            #     json.dump(data, file)

            # sentence = 'you may copy it'
            # NoSmoothingProbas = json.load(open(file_path_NS), 'r')
            NoSmoothingProbas = defaultdict(int, {eval(k): v for k, v in json.load(open(file_path_NS)).items()})

            total_train_Perplexity = 0
            count_train = 0
            for sent_tkn in train_retokenized_sentences:
                count_train += 1
                # sentence = str(input("input sentence: "))
                # [sent_tkn], voc = merges_in_list2(merges_in_list1(Tokenizer().tokenize_sentences(sentence)))
                # sent_tkn = tok_list
                # print(sent_tkn)
                lnt = len(sent_tkn)
                NoSmoothingSentLogEProba = 0
                for i in range(len(sent_tkn)-2):
                    p_curr = NoSmoothingProbas.get((sent_tkn[i],sent_tkn[i+1],sent_tkn[i+2]))
                    # print(p_curr)
                    NoSmoothingSentLogEProba += log(p_curr)
                
                Probability_NS = exp(NoSmoothingSentLogEProba)
                # print("Probability_NS: ", Probability_NS)
                # NoSmoothingSentLogEProba

                Perplexity_NoSmoothing = exp(NoSmoothingSentLogEProba*-1/lnt)
                total_train_Perplexity += Perplexity_NoSmoothing
                # print("Perplexity_NS: ", Perplexity_NoSmoothing)
                
                file_path = f'{file_name1}train{file_name2}'
                with open(file_path, 'a') as file:
                    sentence = ' '.join(sent_tkn)
                    combined_text = f"{sentence}\t{Perplexity_NoSmoothing}"
                    file.write(combined_text + '\n')

            with open(f'{file_name1}train{file_name2}', 'a') as file:
                file.write(f"Average Train Perplexity: {total_train_Perplexity/count_train}\n")


            total_test_Perplexity = 0
            count_test = 0
            for sent_tkn in test_retokenized_sentences:
                count_test += 1
                # sentence = str(input("input sentence: "))
                # [sent_tkn], voc = merges_in_list2(merges_in_list1(Tokenizer().tokenize_sentences(sentence)))
                # sent_tkn = tok_list
                # print(sent_tkn)
                lnt = len(sent_tkn)
                NoSmoothingSentLogEProba = 0
                for i in range(len(sent_tkn)-2):
                    p_curr = NoSmoothingProbas.get((sent_tkn[i],sent_tkn[i+1],sent_tkn[i+2]))
                    # print(p_curr)
                    NoSmoothingSentLogEProba += log(p_curr)
                    
                Probability_NS = exp(NoSmoothingSentLogEProba)
                # print("Probability_NS: ", Probability_NS)
                # NoSmoothingSentLogEProba

                Perplexity_NoSmoothing = exp(NoSmoothingSentLogEProba*-1/lnt)
                total_test_Perplexity += Perplexity_NoSmoothing
                # print("Perplexity_NS: ", Perplexity_NoSmoothing)
                
                file_path = f'{file_name1}test{file_name2}'
                with open(file_path, 'a') as file:
                    sentence = ' '.join(sent_tkn)
                    combined_text = f"{sentence}\t{Perplexity_NoSmoothing}"
                    file.write(combined_text + '\n')

            with open(f'{file_name1}test{file_name2}', 'a') as file:
                file.write(f"Average Test Perplexity: {total_test_Perplexity/count_test}\n")

        
        """
        ############################################################################################################
        # sentence = 'you may copy it'
        # NoSmoothingProbas = json.load(open(file_path_NS), 'r')
        NoSmoothingProbas = defaultdict(int, {eval(k): v for k, v in json.load(open(file_path_NS)).items()})
        sentence = str(input("input sentence: "))
        [sent_tkn], voc = merges_in_list2(merges_in_list1(Tokenizer().tokenize_sentences(sentence)))
        # print(sent_tkn)
        lnt = len(sent_tkn)
        NoSmoothingSentLogEProba = 0
        for i in range(len(sent_tkn)-2):
            p_curr = NoSmoothingProbas.get((sent_tkn[i],sent_tkn[i+1],sent_tkn[i+2]))
            # print(p_curr)
            NoSmoothingSentLogEProba += log(p_curr)
        print("Probability_NS: ",exp(NoSmoothingSentLogEProba))
        # NoSmoothingSentLogEProba


        # return exp(p)

        Perplexity_NoSmoothing = exp(NoSmoothingSentLogEProba*-1/lnt)
        print("Perplexity_NS: ", Perplexity_NoSmoothing)
        # total_train_Perplexity += Perplexity_Interpolation
        # print("Perplexity_NS: ", Perplexity_NoSmoothing)
        """


        '''
        # predictNextWordInSentence = 'you may copy it pride and pre'
        NoSmoothingProbas = json.load(open(file_path_NS), 'r')
        predictNextWordInSentence = str(input("input sentence: "))
        [sent_tkn], voc = merges_in_list2(merges_in_list1(Tokenizer().tokenize_sentences(predictNextWordInSentence)))
        print(sent_tkn)
        [penultimateWord, ultimateWord] = sent_tkn[-2:-1]
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
        print(possibleNextWords)
        '''






    elif(model == 'i'):
        file_path_IProbability_tri = f'{corpus_path}_InterpolationProbasTrig.txt'
        if not os.path.exists(file_path_IProbability_tri) or os.path.getsize(file_path_IProbability_tri) == 0:        
            for _ in range(1000):
                index = random.randint(0, len(tokenized_sentences) - 1)
                test_tokenized_sentences.append(tokenized_sentences.pop(index))

            # Now 'tokenized_sentences' contains the remaining lists
            train_tokenized_sentences = tokenized_sentences

            # print("Removed Lists:")
            # print(test_tokenized_sentences)

            # print("\nRemaining Lists:")
            # print(train_tokenized_sentences)


            train_retokenized_sentences, train_vocabulary = merges_in_list2(train_tokenized_sentences)
            test_retokenized_sentences, test_vocabulary = merges_in_list2(test_tokenized_sentences)
            

            
            # print(retokenized_sentences)
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
            # Example usage:
            train_token_sets = train_retokenized_sentences

            n = 1
            unigrams, unigram_counts = create_n_grams(n, train_token_sets)
            # print(unigrams)
            # print(len(unigrams))
            # print(unigram_counts)
            # print(len(unigram_counts))
                # File path
            file_path_UnigCnt = f'{corpus_path}_ip_unigram_counts.txt'

            # Write dictionary to file
            with open(file_path_UnigCnt, 'w') as file:
                json.dump({str(key): value for key, value in unigram_counts.items()}, file)

            n = 2
            bigrams, bigram_counts = create_n_grams(n, train_token_sets)
            # print(bigrams)
            # print(len(bigrams))
            # print(bigram_counts)
            # print(len(bigram_counts))
            # File path
            file_pathBigCnt = f'{corpus_path}_ip_bigram_counts.txt'

            # Write dictionary to file
            with open(file_pathBigCnt, 'w') as file:
                json.dump({str(key): value for key, value in bigram_counts.items()}, file)

            n = 3
            trigrams, trigram_counts = create_n_grams(n, train_token_sets)
            # print(trigrams)
            # print(len(trigrams))
            # print(trigram_counts)
            # print(len(trigram_counts))

            # File path
            file_path_TrigCnt = f'{corpus_path}_ip_trigram_counts.txt'

            # Write dictionary to file
            with open(file_path_TrigCnt, 'w') as file:
                json.dump({str(key): value for key, value in trigram_counts.items()}, file)

        file_path_NSI = f'{corpus_path}_NoSmoothingInterProbas.txt'
        if not os.path.exists(file_path_NSI) or os.path.getsize(file_path_NSI) == 0:
            # trigram_counts = json.load(open(file_path_TrigCnt), 'r')
            # bigram_counts = json.load(open(file_pathBigCnt), 'r')
            # unigram_counts = json.load(open(f'{corpus_path}_unigram_counts.txt'), 'r')
            trigram_counts = defaultdict(int, {eval(k): v for k, v in json.load(open(file_path_TrigCnt)).items()})
            bigram_counts = defaultdict(int, {eval(k): v for k, v in json.load(open(file_pathBigCnt)).items()})
            # unigram_counts = defaultdict(int, {eval(k): v for k, v in json.load(open(f'{corpus_path}_unigram_counts.txt')).items()})
            NoSmoothingProbas = defaultdict(int)
            for trigram in trigram_counts:
                if trigram[0:2] != ('<SOS>', '<SOS>'):
                # print(key[0:2])
                    NoSmoothingProbas[trigram] = trigram_counts[trigram]/bigram_counts[trigram[0:2]]
                else:
                    NoSmoothingProbas[trigram] = trigram_counts[trigram]/bigram_counts[trigram[1:3]]
            # tot_prob=sum(NoSmoothingProbas.values())
            # print(NoSmoothingProbas)
            # for key in NoSmoothingProbas:
                # NoSmoothingProbas[key]=NoSmoothingProbas[key]/tot_prob
            # print(NoSmoothingProbas)

            # File path
            

            # Write dictionary to file
            with open(file_path_NSI, 'w') as file:
                json.dump({str(key): value for key, value in NoSmoothingProbas.items()}, file)  
            
        # # File path
        # file_path = "dictionary.txt"

        # # Write dictionary to file
        # with open(file_path, 'w') as file:
        #     json.dump(data, file)

        # sentence = 'you may copy it'
        # NoSmoothingProbas = json.load(open(file_path_NSI), 'r')    
        NoSmoothingProbas = defaultdict(int, {eval(k): v for k, v in json.load(open(file_path_NSI)).items()})    
        # def linearInterpolation(self):

        
        file_path_lamb_das_tri = f'{corpus_path}_lamb_das_tri.txt'
        if not os.path.exists(file_path_lamb_das_tri) or os.path.getsize(file_path_lamb_das_tri) == 0:
            InterpolationProbasTrig = defaultdict(int)
            lamb_das_tri = [0, 0, 0]
            for trigram in trigram_counts:
                f_abc = trigram_counts[trigram]
                f_ab = bigram_counts[(trigram[0],trigram[1])]
                f_bc = bigram_counts[(trigram[1],trigram[2])]
                f_b = unigram_counts[trigram[1]]
                f_c = unigram_counts[trigram[2]]
                # print(f_abc, f_ab, f_bc, f_b, f_c)
                func_vals = [(f_abc - 1)/(f_ab - 1) if f_ab > 1 else 0, (f_bc - 1)/(f_b - 1) if f_b > 1 else 0, (f_c - 1)/(len(unigrams) - 1)]
                lamb_das_tri[argmax(func_vals)] += f_abc




            total = sum(lamb_das_tri)
            # lamb_das = [lamb_da/np.sum(lamb_das) for lamb_da in lamb_das]
            lamb_das_tri = [lamb_da / total for lamb_da in lamb_das_tri]
            # lamb_das_tri = lamb_das_tri.tolist()
            # print(lamb_das_tri)
            
            # File path

            # Write dictionary to file
            with open(file_path_lamb_das_tri, 'w') as file:
                json.dump(lamb_das_tri, file)

        if not os.path.exists(file_path_IProbability_tri) or os.path.getsize(file_path_IProbability_tri) == 0:
            lamb_das_tri = json.load(open(file_path_lamb_das_tri))
            for trigram in trigram_counts:
                p_uni = unigram_counts.get(trigram[2]) / len(unigrams)

                f_bc = bigram_counts[(trigram[1], trigram[2])]
                f_b = unigram_counts[trigram[1]]
                p_bi = f_bc / f_b if f_b > 0 else 0

                p_tri = NoSmoothingProbas.get(trigram)

                InterpolationProbasTrig[trigram] = lamb_das_tri[0] * p_uni + lamb_das_tri[1] * p_bi + lamb_das_tri[2] * p_tri
            # print(InterpolationProbasTrig)
                
            # File path

            # Write dictionary to file
            with open(file_path_IProbability_tri, 'w') as file:
                json.dump({str(key): value for key, value in InterpolationProbasTrig.items()}, file)  
        



            # sentence = 'you may copy it hemu'

            # InterpolationProbasTrig = json.load(open(file_path_IProbability_tri), 'r')
            InterpolationProbasTrig = defaultdict(int, {eval(k): v for k, v in json.load(open(file_path_IProbability_tri)).items()})

            total_train_Perplexity = 0
            count_train = 0
            for sent_tkn in train_retokenized_sentences:
                count_train += 1        
                # sentence = str(input("input sentence: "))
                # [sent_tkn], voc = merges_in_list2(merges_in_list1(Tokenizer().tokenize_sentences(sentence)))
                # print(sent_tkn)
                lnt = len(sent_tkn)
                InterpolationSentLogEProba = 0
                for i in range(len(sent_tkn)-2):
                    p_curr = InterpolationProbasTrig.get((sent_tkn[i], sent_tkn[i+1], sent_tkn[i+2]))
                    # print((sent_tkn[0][i],sent_tkn[0][i+1],sent_tkn[0][i+2]))
                    # print(p_curr)

                    if p_curr == None :
                        # continue
                        p_curr = 0.00001

                    InterpolationSentLogEProba += log(p_curr)

                Probability_LI = exp(InterpolationSentLogEProba)
                # print("Probability_LI: ", exp(InterpolationSentLogEProba))
                # return exp(p)

                Perplexity_Interpolation = exp(InterpolationSentLogEProba*-1/lnt)
                # print("Perplexity_LI: ", Perplexity_Interpolation)
                total_train_Perplexity += Perplexity_Interpolation
                # print("Perplexity_NS: ", Perplexity_NoSmoothing)
                
                file_path = f'{file_name1}train{file_name2}'
                with open(file_path, 'a') as file:
                    sentence = ' '.join(sent_tkn)
                    combined_text = f"{sentence}\t{Perplexity_Interpolation}"
                    file.write(combined_text + '\n')

            # open(f'{file_name1}train{file_name2}', 'a').write(f"Average Train Perplexity: {total_train_Perplexity/count_train}\n")
            # with open(f'{file_name1}train{file_name2}', 'a') as file:
            #     file.write(f"Average Train Perplexity: {total_train_Perplexity/count_train}\n")
            with open(f'{file_name1}train{file_name2}', 'r+') as file:
                # Read the contents of the file
                # Move the cursor to the start of the file
                # Write the new line at the beginning of the file
                # Write back the contents after the new line
                contents = file.read()
                file.seek(0)
                file.write(f"Average Test Perplexity: {total_train_Perplexity/count_train}\n")
                file.write(contents)


            InterpolationProbasTrig = defaultdict(int, {eval(k): v for k, v in json.load(open(file_path_IProbability_tri)).items()})
            total_test_Perplexity = 0
            count_test = 0
            for sent_tkn in test_retokenized_sentences:
                count_test += 1        
                # sentence = str(input("input sentence: "))
                # [sent_tkn], voc = merges_in_list2(merges_in_list1(Tokenizer().tokenize_sentences(sentence)))
                # print(sent_tkn)
                lnt = len(sent_tkn)
                InterpolationSentLogEProba = 0
                for i in range(len(sent_tkn)-2):
                    p_curr = InterpolationProbasTrig.get((sent_tkn[i], sent_tkn[i+1], sent_tkn[i+2]))
                    # print((sent_tkn[0][i],sent_tkn[0][i+1],sent_tkn[0][i+2]))
                    # print(p_curr)

                    if p_curr == None :
                        # continue
                        p_curr = 0.00001

                    InterpolationSentLogEProba += log(p_curr)

                Probability_LI = exp(InterpolationSentLogEProba)
                # print("Probability_LI: ", exp(InterpolationSentLogEProba))
                # return exp(p)

                Perplexity_Interpolation = exp(InterpolationSentLogEProba*-1/lnt)
                # print("Perplexity_LI: ", Perplexity_Interpolation)
                total_test_Perplexity += Perplexity_Interpolation
                # print("Perplexity_NS: ", Perplexity_NoSmoothing)
                
                file_path = f'{file_name1}test{file_name2}'
                with open(file_path, 'a') as file:
                    sentence = ' '.join(sent_tkn)
                    combined_text = f"{sentence}\t{Perplexity_Interpolation}"
                    file.write(combined_text + '\n')

            # open(f'{file_name1}train{file_name2}', 'a').write(f"Average Test Perplexity: {total_test_Perplexity/count_train}\n")
            # with open(f'{file_name1}test{file_name2}', 'a') as file:
            #     file.write(f"Average Test Perplexity: {total_test_Perplexity/count_test}\n")
            with open(f'{file_name1}test{file_name2}', 'r+') as file:
                # Read the contents of the file
                # Move the cursor to the start of the file
                # Write the new line at the beginning of the file
                # Write back the contents after the new line
                contents = file.read()
                file.seek(0)
                file.write(f"Average Test Perplexity: {total_test_Perplexity/count_test}\n")
                file.write(contents)


        
        ############################################################################################################
        InterpolationProbasTrig = defaultdict(int, {eval(k): v for k, v in json.load(open(file_path_IProbability_tri)).items()})
        sentence = str(input("input sentence: "))
        [sent_tkn], voc = merges_in_list2(merges_in_list1(Tokenizer().tokenize_sentences(sentence)))
        # print(sent_tkn)
        lnt = len(sent_tkn)
        InterpolationSentLogEProba = 0
        for i in range(len(sent_tkn)-2):
            p_curr = InterpolationProbasTrig.get((sent_tkn[i], sent_tkn[i+1], sent_tkn[i+2]))
            # print((sent_tkn[0][i],sent_tkn[0][i+1],sent_tkn[0][i+2]))
            # print(p_curr)

            if p_curr == None :
                # continue
                p_curr = 0.00001

            InterpolationSentLogEProba += log(p_curr)

        # Probability_LI = exp(InterpolationSentLogEProba)
        print("Probability_LI: ", exp(InterpolationSentLogEProba))
        # return exp(p)

        Perplexity_Interpolation = exp(InterpolationSentLogEProba*-1/lnt)
        print("Perplexity_LI: ", Perplexity_Interpolation)
        # total_train_Perplexity += Perplexity_Interpolation
        # print("Perplexity_NS: ", Perplexity_NoSmoothing)
        




        '''
        # predictNextWordInSentence = 'you may copy it pride and pre'
        InterpolationProbasTrig = json.load(open(file_path_IProbability_tri), 'r')
        predictNextWordInSentence = str(input("input sentence: "))
        [sent_tkn], voc = merges_in_list2(merges_in_list1(Tokenizer().tokenize_sentences(predictNextWordInSentence)))
        print(sent_tkn)
        [penultimateWord, ultimateWord] = sent_tkn[-2:-1]
        # print(penultimateWord, ultimateWord)

        possibleNextWords = defaultdict(int)
        for trigram in InterpolationProbasTrig:
            if trigram[0] == penultimateWord and trigram[1] == ultimateWord:
                # print(trigram)
                possibleNextWords[trigram[2]] += InterpolationProbasTrig[trigram]
        # print(possibleNextWords)
        """
        for bigram in bigram_counts:
            if bigram[0] == ultimateWord:
                # print(trigram)
                possibleNextWords[bigram[1]] += InterpolationProbasBig[bigram]
        # print(possibleNextWords)
        """
        total_sum = sum(possibleNextWords.values())
        for key in possibleNextWords:
            possibleNextWords[key] /= total_sum
        print(possibleNextWords)'''





    elif(model == 'g'):
        file_path_SGT = f'{corpus_path}_SimpleGoodTuringProbas.txt'
        if not os.path.exists(file_path_SGT) or os.path.getsize(file_path_SGT) == 0:
            for _ in range(1000):
                index = random.randint(0, len(tokenized_sentences) - 1)
                test_tokenized_sentences.append(tokenized_sentences.pop(index))

            # Now 'tokenized_sentences' contains the remaining lists
            train_tokenized_sentences = tokenized_sentences

            # print("Removed Lists:")
            # print(test_tokenized_sentences)

            # print("\nRemaining Lists:")
            # print(train_tokenized_sentences)


            train_retokenized_sentences, train_vocabulary = merges_in_list2(train_tokenized_sentences)
            test_retokenized_sentences, test_vocabulary = merges_in_list2(test_tokenized_sentences)
            

            
            # print(retokenized_sentences)
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
            # Example usage:
            train_token_sets = train_retokenized_sentences

            n = 1
            unigrams, unigram_counts = create_n_grams(n, train_token_sets)
            # print(unigrams)
            # print(len(unigrams))
            # print(unigram_counts)
            # print(len(unigram_counts))
                # File path
            file_path_UnigCnt = f'{corpus_path}_sgt_unigram_counts.txt'

            # Write dictionary to file
            with open(file_path_UnigCnt, 'w') as file:
                json.dump({str(key): value for key, value in unigram_counts.items()}, file)

            n = 2
            bigrams, bigram_counts = create_n_grams(n, train_token_sets)
            # print(bigrams)
            # print(len(bigrams))
            # print(bigram_counts)
            # print(len(bigram_counts))
            # File path
            file_pathBigCnt = f'{corpus_path}_sgt_bigram_counts.txt'

            # Write dictionary to file
            with open(file_pathBigCnt, 'w') as file:
                json.dump({str(key): value for key, value in bigram_counts.items()}, file)

            n = 3
            trigrams, trigram_counts = create_n_grams(n, train_token_sets)
            # print(trigrams)
            # print(len(trigrams))
            # print(trigram_counts)
            # print(len(trigram_counts))

            # File path
            file_path_TrigCnt = f'{corpus_path}_sgt_trigram_counts.txt'

            # Write dictionary to file
            with open(file_path_TrigCnt, 'w') as file:
                json.dump({str(key): value for key, value in trigram_counts.items()}, file)
            
        if not os.path.exists(file_path_SGT) or os.path.getsize(file_path_SGT) == 0:
            # trigram_counts = json.load(open(file_path_TrigCnt), 'r')
            trigram_counts = defaultdict(int, {eval(k): v for k, v in json.load(open(file_path_TrigCnt)).items()})
            # print(trigram_counts)
            # print(trigram_counts)
            SimpleGoodTuringProbas = simpleGoodTuringProbs(trigram_counts, 1.65)
            # SimpleGoodTuringProbas[('<OOV>', '<OOV>', '<OOV>')] = p_0
            # print(SimpleGoodTuringProbas)
            # print(p_0)
            # File path

            # Write dictionary to file
            with open(file_path_SGT, 'w') as file:
                # json.dump(SimpleGoodTuringProbas, file)
                json.dump({str(key): value for key, value in SimpleGoodTuringProbas.items()}, file)


            # SimpleGoodTuringProbas = json.load(open(file_path_SGT), 'r')
            SimpleGoodTuringProbas = defaultdict(int, {eval(k): v for k, v in json.load(open(file_path_SGT)).items()})

            total_train_Perplexity = 0
            count_train = 0
            for sent_tkn in train_retokenized_sentences:
                count_train += 1   
                # sentence = str(input("input sentence: "))
                # [sent_tkn], voc = merges_in_list2(merges_in_list1(Tokenizer().tokenize_sentences(sentence)))
                # print(sent_tkn)
                lnt = len(sent_tkn)
                SimpleGoodTuringSentLogEProba = 0
                for i in range(len(sent_tkn)-2):
                    # p_curr = SimpleGoodTuringProbas.get((sent_tkn[i], sent_tkn[i+1], sent_tkn[i+2]))
                    # print((sent_tkn[0][i],sent_tkn[0][i+1],sent_tkn[0][i+2]))
                    # print(p_curr)
                    if (sent_tkn[i],sent_tkn[i+1],sent_tkn[i+2]) not in SimpleGoodTuringProbas:
                        p_curr = SimpleGoodTuringProbas[('<OOV>', '<OOV>', '<OOV>')]
                    else:
                        p_curr = SimpleGoodTuringProbas[(sent_tkn[i],sent_tkn[i+1],sent_tkn[i+2])]
                    # if p_curr == None :
                        # continue
                        # p_curr = 1 #exp(-1)


                    SimpleGoodTuringSentLogEProba += log(p_curr)

                Probability_SGT = exp(SimpleGoodTuringSentLogEProba)
                # print("Probability_SGT: ", exp(SimpleGoodTuringSentLogEProba))
                Perplexity_SGT = exp(SimpleGoodTuringSentLogEProba*-1/lnt)
                # print("Perplexity_SGT: ", Perplexity_SGT)
                # print("Probability_LI: ", exp(InterpolationSentLogEProba))
                # return exp(p)

                # Perplexity_Interpolation = exp(InterpolationSentLogEProba*-1/lnt)
                # print("Perplexity_LI: ", Perplexity_Interpolation)
                total_train_Perplexity += Perplexity_SGT
                # print("Perplexity_NS: ", Perplexity_NoSmoothing)
                
                file_path = f'{file_name1}train{file_name2}'
                with open(file_path, 'a') as file:
                    sentence = ' '.join(sent_tkn)
                    combined_text = f"{sentence}\t{Perplexity_SGT}"
                    file.write(combined_text + '\n')

            # open(f'{file_name1}train{file_name2}', 'a').write(f"Average Train Perplexity: {total_train_Perplexity/count_train}\n")
            # with open(f'{file_name1}train{file_name2}', 'a') as file:
            #     file.write(f"Average Train Perplexity: {total_train_Perplexity/count_train}\n")
            with open(f'{file_name1}train{file_name2}', 'r+') as file:
                # Read the contents of the file
                # Move the cursor to the start of the file
                # Write the new line at the beginning of the file
                # Write back the contents after the new line
                contents = file.read()
                file.seek(0)
                file.write(f"Average Test Perplexity: {total_train_Perplexity/count_train}\n")
                file.write(contents)

            # SimpleGoodTuringProbas = json.load(open(file_path_SGT), 'r')
            SimpleGoodTuringProbas = defaultdict(int, {eval(k): v for k, v in json.load(open(file_path_SGT)).items()})

            total_test_Perplexity = 0
            count_test = 0
            for sent_tkn in test_retokenized_sentences:
                count_test += 1            
                # sentence = str(input("input sentence: "))
                # [sent_tkn], voc = merges_in_list2(merges_in_list1(Tokenizer().tokenize_sentences(sentence)))
                # print(sent_tkn)
                lnt = len(sent_tkn)
                SimpleGoodTuringSentLogEProba = 0
                for i in range(len(sent_tkn)-2):
                    # p_curr = SimpleGoodTuringProbas.get((sent_tkn[i], sent_tkn[i+1], sent_tkn[i+2]))
                    # print((sent_tkn[0][i],sent_tkn[0][i+1],sent_tkn[0][i+2]))
                    # print(p_curr)
                    if (sent_tkn[i],sent_tkn[i+1],sent_tkn[i+2]) not in SimpleGoodTuringProbas:
                        p_curr = SimpleGoodTuringProbas[('<OOV>', '<OOV>', '<OOV>')]
                    else:
                        p_curr = SimpleGoodTuringProbas[(sent_tkn[i],sent_tkn[i+1],sent_tkn[i+2])]
                    # if p_curr == None :
                        # continue
                        # p_curr = 1 #exp(-1)


                    SimpleGoodTuringSentLogEProba += log(p_curr)

                Probability_SGT = exp(SimpleGoodTuringSentLogEProba)
                # print("Probability_SGT: ", exp(SimpleGoodTuringSentLogEProba))
                Perplexity_SGT = exp(SimpleGoodTuringSentLogEProba*-1/lnt)
                # print("Perplexity_SGT: ", Perplexity_SGT)
                # print("Probability_LI: ", exp(InterpolationSentLogEProba))
                # return exp(p)

                # Perplexity_Interpolation = exp(InterpolationSentLogEProba*-1/lnt)
                # print("Perplexity_LI: ", Perplexity_Interpolation)
                total_test_Perplexity += Perplexity_SGT
                # print("Perplexity_NS: ", Perplexity_NoSmoothing)
                
                file_path = f'{file_name1}test{file_name2}'
                with open(file_path, 'a') as file:
                    sentence = ' '.join(sent_tkn)
                    combined_text = f"{sentence}\t{Perplexity_SGT}"
                    file.write(combined_text + '\n')

            # open(f'{file_name1}test{file_name2}', 'a').write(f"Average Test Perplexity: {total_test_Perplexity/count_train}\n")
            # with open(f'{file_name1}test{file_name2}', 'a') as file:
            #     file.write(f"Average Test Perplexity: {total_test_Perplexity/count_test/.1}\n")
            with open(f'{file_name1}test{file_name2}', 'r+') as file:
                # Read the contents of the file
                # Move the cursor to the start of the file
                # Write the new line at the beginning of the file
                # Write back the contents after the new line
                contents = file.read()
                file.seek(0)
                file.write(f"Average Test Perplexity: {total_test_Perplexity/count_test/.1}\n")
                file.write(contents)


        
        
        ############################################################################################################
        # SimpleGoodTuringProbas = json.load(open(file_path_SGT), 'r')
        SimpleGoodTuringProbas = defaultdict(int, {eval(k): v for k, v in json.load(open(file_path_SGT)).items()})
        # sentence = 'the project berg'
        sentence = str(input("input sentence: "))
        [sent_tkn], voc = merges_in_list2(merges_in_list1(Tokenizer().tokenize_sentences(sentence)))
        # print(sent_tkn)
        lnt = len(sent_tkn)
        SimpleGoodTuringSentLogEProba = 0
        for i in range(len(sent_tkn)-2):
            # p_curr = SimpleGoodTuringProbas.get((sent_tkn[i], sent_tkn[i+1], sent_tkn[i+2]))
            # print((sent_tkn[0][i],sent_tkn[0][i+1],sent_tkn[0][i+2]))
            # print(p_curr)
            if (sent_tkn[i],sent_tkn[i+1],sent_tkn[i+2]) not in SimpleGoodTuringProbas:
                p_curr = SimpleGoodTuringProbas[('<OOV>', '<OOV>', '<OOV>')]
            else:
                p_curr = SimpleGoodTuringProbas[(sent_tkn[i],sent_tkn[i+1],sent_tkn[i+2])]
            # if p_curr == None :
                # continue
                # p_curr = 1 #exp(-1)


            SimpleGoodTuringSentLogEProba += log(p_curr)

        print("Probability_SGT: ", exp(SimpleGoodTuringSentLogEProba))
        Perplexity_SGT = exp(SimpleGoodTuringSentLogEProba*-1/lnt)
        print("Perplexity_SGT: ", Perplexity_SGT)
        





        '''
        # predictNextWordInSentence = 'the project berg'
        SimpleGoodTuringProbas = json.load(open(file_path_SGT), 'r')
        predictNextWordInSentence = str(input("input sentence: "))
        [sent_tkn], voc = merges_in_list2(merges_in_list1(Tokenizer().tokenize_sentences(predictNextWordInSentence)))
        print(sent_tkn)
        # lnt = len(sent_tkn)
        [penultimateWord, ultimateWord] = sent_tkn[-2:-1]
        # print(penultimateWord, ultimateWord)
        possibleNextWords = defaultdict(int)
        for trigram in SimpleGoodTuringProbas:
            if trigram[0] == penultimateWord and trigram[1] == ultimateWord:
                # print(trigram)
                possibleNextWords[trigram[2]] += SimpleGoodTuringProbas[trigram]
        # print(possibleNextWords)
        """
        for bigram in bigram_counts:
            if bigram[0] == ultimateWord:
                # print(trigram)
                possibleNextWords[bigram[1]] += InterpolationProbasBig[bigram]
        # print(possibleNextWords)
        """
        total_sum = sum(possibleNextWords.values())
        for key in possibleNextWords:
            possibleNextWords[key] /= total_sum
        print(possibleNextWords)
        '''

