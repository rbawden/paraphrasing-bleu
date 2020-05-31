#!/usr/bin/python
import numpy as np
import gzip
import re
import williams
from scipy.stats.stats import pearsonr


# normalise the names
def normalise(name, lp):
    name = name.replace('newstest2018.', '')
    name = name.replace('newstest2019.', '')
    name = name.replace('.' + lp, '')
    return name


# read the human DA scores file
def read_ref(filename):
    lp2scores = {}
    with open(filename) as fp:
        # discard first header line
        fp.readline()
        for line in fp:
            lp, score, system = line.strip().split()
            if lp[-2:] != 'en':
                continue
            if lp not in lp2scores:
                lp2scores[lp] = {}

            lp2scores[lp][system] = float(score)

    return lp2scores
    
# read the metrics scores file
def read_scores(filename):
    lp2scores = {}
    if '.gz' in filename:
        fp = gzip.open(filename, 'rt')
    else:
        fp = open(filename)

    for line in fp:
        metric, lp, testset, system, score = line.strip().split(maxsplit = 4)

        # some names are wrong
        testset = testset.replace("DAsegNews+newstest2019", "newstest2019")

        
        if testset not in ['newstest2018',  'newstest2019']:
            continue
        if lp not in lp2scores:
            lp2scores[lp] = {}
        score = score.split()[0] # trailing information on some submissions

        # normalise system name
        system = normalise(system, lp)
        
        lp2scores[lp][system] = float(score)
    fp.close()

    return lp2scores


def correlate(hscores, sscores, neg=False):
    # for each language pair
    c = {}
    for lp in sorted(list(hscores)):
        h, s = [], []

        if lp not in hscores or lp not in sscores:
            continue

        for score in sscores[lp]:
            if score in hscores[lp]:
                h.append(hscores[lp][score])
                if neg:
                    s.append(1 - sscores[lp][score])
                else:
                    s.append(sscores[lp][score])

        correlation = pearsonr(h, s)[0]

        c[lp] = (correlation, len(h))
    return c


def get_results(human_scores, baseline_scores, system_scores):

    hscores = read_ref(human_scores)
    sscores = read_scores(system_scores)
    bscores = read_scores(baseline_scores)

    r23 = correlate(bscores, sscores) # correlation of baseline to system    
    r12 = correlate(hscores, sscores) # correlation of system to human
    r13 = correlate(hscores, bscores) # correlation of baseline system to human


    #print(r12)
    #print(r13)
    #input()
        
    sigs = {}
    for lp in sorted(r12.keys()):
        if lp[-3:] != '-en':
            continue
        if r12[lp][0] > r13[lp][0]:
            sig = williams.williams_test(r12[lp][0], r13[lp][0], r23[lp][0], r12[lp][1])[1]
        else:
            sig = 1
            
        sig_str=''
        if str(sig) == 'nan':
            sig_str = '***'
        elif float(sig) <= 0.001:
            sig_str = '***'
        elif float(sig) <= 0.01:
            sig_str = '**'
        elif float(sig) <= 0.05:
            sig_str = '*'
        sigs[lp] = sig_str

    return r13, r12, sigs
                                                                        
    
def print_results(human_scores, baseline_scores, system_scores, just_scores=False, s=False):
    
    if s:
        hscores = read_scores(human_scores)
    else:
        hscores = read_ref(human_scores)
    sscores = read_scores(system_scores)
    bscores = read_scores(baseline_scores)

    # get correlation of baseline to system
    r23 = correlate(bscores, sscores)

    # get correlation of system to human
    r12 = correlate(hscores, sscores)

    # get correlation of baseline system to human
    r13 = correlate(hscores, bscores)

    # get line
    if not just_scores:
        print(' '.join([x for x in list(sorted(r12.keys())) if x[-3:] =='-en' ]))
        print(' '.join([str(r12[lp][1]) for lp in sorted(r12.keys()) if lp[-3:] == '-en']))
        
    for lp in sorted(r12.keys()):
        if lp[-3:] != '-en':
            continue
        if r12[lp][0] > r13[lp][0]:
            sig = williams.williams_test(r12[lp][0], r13[lp][0], r23[lp][0], r12[lp][1])[1]
        else:
            sig = 1

        sig_str=''
        if str(sig) == 'nan':
            sig_str = '***'
        elif float(sig) <= 0.001:
            sig_str = '***'
        elif float(sig) <= 0.01:
            sig_str = '**'
        elif float(sig) <= 0.05:
            sig_str = '*'
        print('& %.3f' % round(r12[lp][0], 3) + sig_str, end=' ')


    # add average
    ave = sum([r12[lp][0] for lp in r12.keys()])/len(r12)
    print()
    print('\\\\')
        
    


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('human_scores', help="Human DA scores file")
    parser.add_argument('baseline_scores', help="Baseline metric scores file (from sacreBLEU or Meteor)")
    parser.add_argument('system_scores', help="Metric scores file")
    parser.add_argument('-s', help="Human scores in actual fact a system", action='store_true', default=False)
    parser.add_argument('--just_scores', help="Only output scores", action="store_true", default=False)
    args = parser.parse_args()

    print_results(args.human_scores, args.baseline_scores, args.system_scores, args.just_scores, args.s)
