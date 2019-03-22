
from __future__ import division
import numpy as np
import os
from collections import Counter
import nltk

nltk.download('punkt')
from nltk import word_tokenize
from nltk.util import ngrams
import sklearn
import string
from collections import deque
from itertools import islice
import collections
import math
import argparse
import time

# defining global variables
STOP_token = '_STOP_'
unknown_token = '_unknown_'
# token for ngram that appear in dev and test but not in training
UNSEEN_NGRAM = '_UNSEEN_'
# unknown_threshold = 1
# total counts of words in training data. contrain duplicated word
total_words_len = 0
ADD_K_SMOOTHING = 'add_k_smoothing'
LINER_INT = 'liner interpolation'
NO_SMOOTHING = 'no smoothing'
# training tokens
replaced_tokens_train = list()
vocabulary = set()

'''
open a text file, add STOP_token, replace punctuations
'''


def text_processing(filename):
    text = ""
    with open(filename, 'r+') as f:
        text = f.read()
        text = text.replace('\n', ' ' + STOP_token + '\n')
        for ch in '!"#$%&()*+,-./:;<=>?@[\\]^`{|}~':
            text = str.replace(text, ch, ' ')
    return text

'''tokenize processed text'''

def tokenize_unigram(text):
    return nltk.word_tokenize(text)

'''get Vocabulary from training set. including unknown
   :param: input tokenized training text
   :output: unigrams {unigrams:count}
'''

class UnigramModel:
    def unigram_Vocab(self, training_tokens, unknown_threshold):
        # this is the total length/num of tokens in training data
        global total_words_len
        global replaced_tokens_train
        total_words_len = len(training_tokens)
        # initialize word_count pairs
        unigram_vocab = {}
        unigram_vocab[unknown_token] = 0
        # initial work-count dictionary population
        for token in training_tokens:
            unigram_vocab[token] = unigram_vocab.get(token, 0) + 1
        # re-assign unknown
        unk_words = set()
        items = unigram_vocab.items()
        for word, count in items:
            # treat low freq word as unknown
            if count <= unknown_threshold:
                unk_words.add(word)
                unigram_vocab[unknown_token] += count

        unk_words.discard(STOP_token)
        unk_words.discard(unknown_token)

        for word in unk_words:
            del unigram_vocab[word]

        replaced_tokens_train = training_tokens
        for idx, token in enumerate(replaced_tokens_train):
            if token in unk_words:
                replaced_tokens_train[idx] = unknown_token
        #               modify tuple to contain unknown
        return unigram_vocab



'''
after obtaining vocabulary, then
replace unknown in training and test data 
param: input unigram tokens
'''


def replace_unknown(tokens, vocabulary):
    for idx, token in enumerate(tokens):
        if token not in vocabulary:
            #               modify tuple to contain unknown
            token_ls = list(tokens)
            token_ls[idx] = unknown_token
            tokens = tuple(token_ls)
    return tokens




'''ngram generator, n>1
   : param: input tokened texts with STOP sign, and unknown replaced
            could be either training data or test data or sentences

'''


def ngrams_gen(tokens, n):
    start_time = time.time()
    ngrams_tp = tuple()
    text = ' '.join(tokens)
    text = text.replace(STOP_token, STOP_token + '\n')
    sentences = set([w for w in text.splitlines()])
    for word in sentences:
        it = iter(word.split())
        window = deque(islice(it, n), maxlen=n)
        yield tuple(window)
        for item in it:
            window.append(item)
            yield tuple(window)

    ngrams_tp += tuple(window)
    yield ngrams_tp


''' 
    only for n=2,3 or more, generate {words:count} 
    generate ngram count with unknown

'''


def word_freq(tokens):
    start_time = time.time()
    ngram_freq = {}
    # initial work-count dict population
    for token in tokens:
        ngram_freq[token] = ngram_freq.get(token, 0) + 1
    return ngram_freq


''' calculate MLE Probablity of unigram

'''


def unigrams_prob(uni_count_dict):
    # probability dict {word:prob}
    prob_dict = uni_count_dict
    # print vocabulary
    items = prob_dict.items()
    for word, count in items:
        prob_dict[word] = float(count) / float(total_words_len)
    return prob_dict


'''
calculate MLE probability of ngram, n>=2

'''


def ngram_prob(n, tokens, unigram_count):
    start_time = time.time()
    # generate {ngrams:count} from training data
    ngram_list = list(ngrams_gen(tokens, n))

    ngram_count_pairs = word_freq(ngram_list)
    prob_dict = ngram_count_pairs
    if (n == 2):
        items = prob_dict.items()
        uni_count = unigram_count
        # current probablity and word, in case n = 2, input is bigram words:count dict
        # input {a,b}: count, continue to get {a}: count
        for words, count in items:
            # extract the first item in bigram.
            prior_word = words[0]
            # get the count from {unigram: count} generated before
            cnt_prior = uni_count[prior_word]
            # q(w/v) = c(v,w)/c(v)
            prob_dict[words] = count / cnt_prior
        # this should save as global for later use as bigram_prob_dict
        return prob_dict
    if (n > 2):
        items = prob_dict.items()
        # get {n-1gram:count} pairs
        priorgram_list = list(ngrams_gen(tokens, n - 1))
        priorgram_count_pairs = word_freq(priorgram_list)
        # -----------need to discard first few items--------
        for words, count in items:
            prior_word = words[:n - 1]
            cnt_prior = priorgram_count_pairs[prior_word]
            prob_dict[words] = count / cnt_prior
        return prob_dict


'''
    Evaluate the (negative) log probability of this word in this context.
    :param word: the word to get the probability of
    :param prob_dict: the context the word is in
'''
def logprob(word, prob_dict):
    prob_dict = prob_dict
    return -math.log(prob_dict[word], 2)


''' calculate entropy given a test/dev text

'''

def entropy(test_test, n, prob_dict, smooth_type):
    entr = 0.0
    text = test_test
    tokens = tokenize_unigram(text)
    # number of words in text
    text_len = len(tokens)
    global vocabulary

    sentences = set([s for s in text.splitlines()])
    # number of sentences
    sent_num = len(sentences)
    voc_set = set(prob_dict.keys())

    if (n == 1):

        for sent in sentences:
            sent_temp = tokenize_unigram(sent)
            for word in sent_temp:
                if word not in voc_set:
                    entr += logprob(unknown_token, prob_dict)
                else:
                    entr += logprob(word, prob_dict)
    if (n > 1):
        # ngram_prob_dict = ngram_prob(n,train_cut)
        for sent in sentences:
            # generate ngram for single sentence test data
            ngram_tmp = tuple(ngrams_gen(tokenize_unigram(sent), n))
            # iterate ngram in one sentence, skip first n-1 items
            for i in range(n - 1, len(list(ngram_tmp))):
                if ngram_tmp[i] not in voc_set:
                    if (smooth_type == NO_SMOOTHING):
                        entr += -math.log(0, 2)
                    if (smooth_type == ADD_K_SMOOTHING):
                        entr += logprob(UNSEEN_NGRAM, prob_dict)

                else:
                    entr += logprob(ngram_tmp[i], prob_dict)
    return entr / float(text_len - (n - 1) * sent_num)


''' 
perplexity for  ngram
'''


def perplexity(test_text, n, prob_dict, smooth_type):
    return math.pow(2.0, entropy(test_text, n, prob_dict, smooth_type))


'''
add k smoothing for unigram,bigram and trigram, similar to ngrams_prob()
'''


def add_k_smoothing(n, tokens, unigram_count, k, V):
    # generate {ngrams:count} from training data
    if (n == 1):
        prob_dict = unigram_count
    else:
        ngram_list = list(ngrams_gen(tokens, n))
        ngram_count_pairs = word_freq(ngram_list)
        prob_dict = ngram_count_pairs
    if (n == 1):

        items = prob_dict.items()
        for word, count in items:
            prob_dict[word] = (float(count) + k) / (float(total_words_len) + V * k)
        return prob_dict

    if (n == 2):
        items = prob_dict.items()
        # should fix this duplicated thing
        uni_count = unigram_count
        # current probablity and word, in case n = 2, input is bigram words:count dict
        # input {a,b}: count, continue to get {a}: count
        for words, count in items:
            # extract the first item in bigram.
            prior_word = words[0]
            # get the count from {unigram: count} generated before
            cnt_prior = uni_count[prior_word]
            # q(w/v) = c(v,w)/c(v)
            prob_dict[words] = (count + k) / (cnt_prior + k * V)
        # dealing with unseen ngram that might appear in test data
        prob_dict[UNSEEN_NGRAM] = 1 / V
        # this should save as global for later use as bigram_prob_dict
        return prob_dict
    if (n > 2):
        items = prob_dict.items()
        # get {n-1gram:count} pairs
        priorgram_list = list(ngrams_gen(tokens, n - 1))
        priorgram_count_pairs = word_freq(priorgram_list)

        for words, count in items:
            prior_word = words[:n - 1]
            cnt_prior = priorgram_count_pairs[prior_word]
            prob_dict[words] = (count + k) / (cnt_prior + k * V)
        prob_dict[UNSEEN_NGRAM] = 1 / V
        return prob_dict


'''
linear interpolation trigram, use ngrams_prob()
output perplexity directly
'''


def linear_interpolation(dev_text, unigrams_prob_dict, bigram_prob_dict, trigram_prob_dict,
                         la1, la2, la3):
    entr = 0.0
    perplexity = 0.0
    # global vocabulary
    text = dev_text
    tokens = tokenize_unigram(text)
    # number of words in text
    text_len = len(tokens)
    sentences = set([s for s in text.splitlines()])
    # number of sentences
    sent_num = len(sentences)
    n = 3
    new_prob_dict = {}
    unigram_p_dict = unigrams_prob_dict.copy()
    bigram_p_dict = bigram_prob_dict.copy()
    trigram_p_dict = trigram_prob_dict.copy()


    uni_keys_set = set(unigram_p_dict.keys())
    bi_keys_set = set(bigram_p_dict.keys())
    tri_keys_set = set(trigram_p_dict.keys())

    for sent in sentences:
        # generate trigram for dev/test data
        ngram_tmp = tuple(ngrams_gen(tokenize_unigram(sent), n))
        # iterate ngram in one sentence, skip first n-1 items
        for i in range(n - 1, len(list(ngram_tmp))):
            words = ngram_tmp[i]
            uniword = words[2]
            bi_word = words[1:]

            # dealing with unseen words
            if uniword not in uni_keys_set:
                unigram_p_dict[uniword] = unigram_p_dict[unknown_token]
            if bi_word not in bi_keys_set:
                bigram_p_dict[bi_word] = 0
            if ngram_tmp[i] not in tri_keys_set:
                trigram_p_dict[ngram_tmp[i]] = 0
            new_prob_dict[ngram_tmp[i]] = float(la1) * trigram_p_dict[ngram_tmp[i]] + float(la2) * bigram_p_dict[
                bi_word] + float(la3) * unigram_p_dict[uniword]
            # get entropy
            entr += logprob(ngram_tmp[i], new_prob_dict)
    # get perplexity
    return math.pow(2.0, (entr / float(text_len - (n - 1) * sent_num)))


'''
 input n, test and training, right now no smoothing 
 output perplexity
'''


def main():
    start_time = time.time()
    # args = get_args()
    # n = args.n
    """training_set = args.training_set
    dev_set = args.dev_set
    unknown_threshold = args.threshold"""

    training_set = open("brown.train.txt", "r")
    dev_set = open("brown.dev.txt", "r")
    unknown_threshold = 1

    print('your input: unknown_threshold, training, dev or test', unknown_threshold, training_set, dev_set)

    # process train and dev/test data
    train_text = text_processing("./brown.train.txt")
    dev_text = text_processing("./brown.dev.txt")
    train_token = tokenize_unigram(train_text)

    print('tokenization done!')

    start_time = time.time()
    un = UnigramModel()
    unigram_count = un.unigram_Vocab(train_token, unknown_threshold)
    # a list of vocabulary in unigrams
    vocabulary = set(unigram_count.keys())

    # generate unigram probablity dict
    uni_prob_dict = {}
    uni_prob_dict = unigram_count.copy()
    start_time = time.time()
    unigrams_prob_dict = unigrams_prob(uni_prob_dict)

    V = len(vocabulary)
    print("Vocabulary length", V)
    print('total_words_len', total_words_len)
    print("training unigram finished")
    # generate bigram probability dict
    bigram_prob_dict = ngram_prob(2, replaced_tokens_train, unigram_count)
    print(bigram_prob_dict)
    print("training bigram finished")
    # generate trigram probability dict
    trigram_prob_dict = ngram_prob(3, replaced_tokens_train, unigram_count)
    print("training trigram finished")

    # get perplextity for unigram for dev data
    print('perplexity for unsmoothed unigram:')
    print(perplexity(dev_text, 1, unigrams_prob_dict, NO_SMOOTHING))
    print('\n')

    # --------------------------uncomment this block if only wanna run on test-------
    # on dev data
    # add k smoothing
    print('perplexity for add-k-smoothing on development data:')
    k_ls = (0.0000001, 0.000001, 0.00001, 0.0001, 0.01, 0.1, 1)
    for k in k_ls:
        print('k=:', k)
        print('perplexity of add k unigram:')
        uni_addk_prob_dict = add_k_smoothing(1, replaced_tokens_train, unigram_count, k, V)
        print(perplexity(dev_text, 1, uni_addk_prob_dict, ADD_K_SMOOTHING))

        print('perplexity for add k bigram:')
        bi_addk_prob_dict = add_k_smoothing(2, replaced_tokens_train, unigram_count, k, V)
        print(perplexity(dev_text, 2, bi_addk_prob_dict, ADD_K_SMOOTHING))

        print('perplexity for add k trigram:')
        tri_addk_prob_dict = add_k_smoothing(3, replaced_tokens_train, unigram_count, k, V)
        print(perplexity(dev_text, 3, tri_addk_prob_dict, ADD_K_SMOOTHING))
        print('\n')

    print('------end add_k_smoothing---------------')
    print('\n')

    # linear interpolation
    print('------liner interpolation on development data-----')

    la_ls = [(0.001, 0.009, 0.99), (0.3, 0.3, 0.4), (0.6, 0.3, 0.1), (0.99, 0.009, 0.001)]

    for lamda in la_ls:
        print('lamda1,lamda2,lamda3:', lamda[0], lamda[1], lamda[2])
        perplexity_li = linear_interpolation(dev_text, unigrams_prob_dict, bigram_prob_dict,
                                             trigram_prob_dict, lamda[0], lamda[1], lamda[2])
        print('linear interpolation perplexity:')
        print(perplexity_li)
    print('\n')
    print("--- %s seconds for training and testing on develoment data ---" % (time.time() - start_time))
    # ------------------------end of dev data --------------------------------------------

    # ---------------------------------------------------------------------------#
    # -----------------uncomment the following chunk to run test data------------#
    # run on test data
    #
    # # add k smoothing on test
    # test_text = text_processing("./brown.test.txt")
    #
    # print('perplexity for add-k-smoothing on test data:')
    # k = 0.0000001
    # print('k=:', k)
    #
    # print('perplexity for add k trigram:')
    # tri_addk_prob_dict = add_k_smoothing(3, replaced_tokens_train, unigram_count, k, V)
    # print(perplexity(test_text, 3, tri_addk_prob_dict, ADD_K_SMOOTHING))
    # print('\n')
    #
    # print('------end add_k_smoothing---------------')
    # print('\n')
    #
    # # linear interpolation
    # print('------linear interpolation on test data-----')
    # lamda = (0.1, 0.2, 0.7)
    #
    # print('lamda1,lamda2,lamda3:', lamda[0], lamda[1], lamda[2])
    # perplexity_li = linear_interpolation(test_text, unigrams_prob_dict, bigram_prob_dict,
    #                                      trigram_prob_dict, lamda[0], lamda[1], lamda[2])
    # print('linear interpolation perplexity:')
    #
    # print(perplexity_li)
    # print('\n')


if __name__ == "__main__":
    main()
