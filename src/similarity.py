import numpy as np
from numpy import linalg as la
import datetime
import sys
from config import cfg
from util import euclidSimilar,pearsonSimilar,cosSimilar

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print "similarity <dictionary file> <embed file> <similarity file>"
        sys.exit()

    dict_file = open(sys.argv[1], 'rb').readlines()
    word_dict = {}
    reverse_word_dict = {}
    for line in dict_file:
        word_dict[line.split('\t')[0]] = line.split('\t')[1]
        reverse_word_dict[int(line.split('\t')[1]) - 1] = line.split('\t')[0]
    dictionary_size = len(word_dict)

    embed_file = open(sys.argv[2], 'rb').readlines()
    embed_weight = np.zeros((dictionary_size, cfg.embedding_size))
    embed_dict = np.zeros((dictionary_size, dictionary_size))

    counter = 0
    str_list = []
    starttime = datetime.datetime.now()
    for iter,embed_item in enumerate(embed_file):
        if iter == 0:
            continue
        str_list = embed_item.split(',')
        embed_weight[iter - 1] = [float(item) for item in str_list]
    endtime = datetime.datetime.now()
    print ('set embedding weight cost: %d s' % ((endtime - starttime).seconds))
    norm_vector = np.apply_along_axis(la.norm, 1, embed_weight)

    sim_matrix = np.matmul(embed_weight, embed_weight.T)
    for i, v in enumerate(sim_matrix):
        sim_matrix[i] /= norm_vector
        sim_matrix[i][i] = 0.0

    '''
    sim_matrix = np.zeros(shape=(dictionary_size, dictionary_size))
    starttime = datetime.datetime.now()
    for i in range(dictionary_size):
        for j in range(i + 1, dictionary_size):
            sim_matrix[i][j] = cosSimilar(embed_weight[i], embed_weight[j])
            sim_matrix[j][i] = sim_matrix[i][j]
    endtime = datetime.datetime.now()
    print ('calculate similarity cost: %d s' % ((endtime - starttime).seconds))
    '''

    output_file = open(sys.argv[3], 'w')
    starttime = datetime.datetime.now()
    for i in range(dictionary_size):
        ind = sim_matrix[i].argsort()[cfg.top_k_sim:]
        sim_words = []
        for j in range(len(ind)):
            sim_words.append(reverse_word_dict[ind[j]])
        output_file.write(reverse_word_dict[i] + ':' + ','.join(sim_words) + '\n')
    output_file.close()
    endtime = datetime.datetime.now()
    print ('get top k similarity item cost: %d s' % ((endtime - starttime).seconds))
