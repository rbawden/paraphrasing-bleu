#!/usr/bin/python
import numpy as np
import gzip
import re
import os

# normalise the names
def normalise(name, lp):
    name = name.replace('newstest2018.', '')
    name = name.replace('newstest2019.', '')
    name = name.replace('.' + lp, '')
    name = re.sub('\.' + lp + '$', '', name)
    name = name.replace('-', '_')
    return name


# read the human DA scores file
def read_ref(filename):
    lp2scores = {}
    with open(filename) as fp:
        # discard first header line
        fp.readline()
        for line in fp:
            # headers as follows (better system is first)
            lp, testset, segid, better, worse = line.strip().split()

            # skip these examples (including other test sets)
            if testset not in ['newstest2018', 'newstest2019'] or '-en' not in lp:
                continue

            # add comparison
            if lp not in lp2scores:
                lp2scores[lp] = {}
            if segid not in lp2scores[lp]:
                lp2scores[lp][segid] = {}

            # normalise names
            better = normalise(better, lp)
            worse = normalise(worse, lp)

            # should not be duplicate entries of comparisons
            if (better, worse) in lp2scores[lp][segid]:
                print("error")

            # add comparison - better is always first, but is also the value
            lp2scores[lp][segid][(better, worse)] = better
        
    return lp2scores


# read the metrics scores file
def read_scores(filename):
    lp2scores = {}
    systems = []
    if '.gz' in filename:
        fp = gzip.open(filename, 'rt')
    else:
        fp = open(filename)

    for line in fp:
        metric, lp, testset, system, segid, score = line.strip().split(maxsplit=5)

        # normalise systemname
        system = normalise(system, lp)

        if testset not in ['newstest2018', 'newstest2019']:
            continue
        
        # just take the core (can have trailing information on the score)
        score = score.split()[0] 
        
        # add metric score for individual systems
        if lp not in lp2scores:
            lp2scores[lp] = {}
        if segid not in lp2scores[lp]:
            lp2scores[lp][segid] = {}
        lp2scores[lp][segid][system] = float(score)
    fp.close()

    return lp2scores

# calculate significance levels
def bootstrap_resampling(hscores, bscores, sscores):
    repetitions = 1000
    fixed_values = []
    
    for segid in hscores:
        fixed_values.extend( [(segid, x) for x in hscores[segid].items()] )
    value_indices = np.arange(0, len(fixed_values))
        
    all_taus = []
    for i in range(repetitions):
        # draw with replacement
        # of format segid: {better_sys, worse_sys}
        sampled_indices = np.random.choice(value_indices, (len(value_indices)), replace=True)

        # get the samples (only need to select hscores really, because they are filtered
        # in correlate
        hsamples = {}
        for i in range(len(hscores)):
            idx = sampled_indices[i]
            segid, values = fixed_values[idx]
            if segid not in hsamples:
                hsamples[segid] = {}
            hsamples[segid][values[0]] = values[1]
            
        # calculate the scores
        stau, _ = correlate(hsamples, sscores)
        btau, _ = correlate(hsamples, bscores)

        all_taus.append((stau, btau))

    # number where s is better than b
    p = len([_ for x in all_taus if x[1] >= x[0]])/repetitions

    return p
        

def correlate_all_lps(hscores, bscores, sscores):
    # for each language pair
    results = {}
    p = {}
    for lp in sorted(list(hscores)):

        # print out first column
        if '-en' in lp:
            os.sys.stderr.flush()
            
        # only for languages specified in the system file
        if lp not in sscores:
            continue

        # calculate kendall Tau
        results[lp] = correlate(hscores[lp], sscores[lp])

        # get p through bootstrap-resampling
        p[lp] = bootstrap_resampling(hscores[lp], bscores[lp], sscores[lp])

    return results, p


def correlate(hscores, sscores):
    results = {'concord': 0, 'discord': 0}
    
    for segid in hscores:
        # get judgments for this segment
        for comparison in hscores[segid]:
            # ignore non-present ones (???)
            if segid not in sscores or comparison[0] not in sscores[segid] or \
               comparison[1] not in sscores[segid]:
                continue

            # get better and worse system
            better_system = hscores[segid][comparison]
            worse_system = comparison[1] if better_system == comparison[0] else comparison[0] # other system
            
            # get BLEU scores for each system
            better_sys_score = sscores[segid][better_system]
            worse_sys_score = sscores[segid][worse_system]
                
            # concordant or discordant
            if better_sys_score == worse_sys_score:
                results['discord'] += 1 
            elif better_sys_score > worse_sys_score:
                results['concord'] += 1
            else:
                results['discord'] += 1

    tau = (results['concord'] - results['discord']) / float(results['concord'] + results['discord'])

    return (tau, results['concord'] + results['discord'])
        

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('human_scores', help="Human RR scores file")
    parser.add_argument('baseline_scores', help="Baseline scores file")
    parser.add_argument('system_scores', help="Metric scores file")
    parser.add_argument('--just_scores', help="Only output scores", action="store_true", default=False)
    args = parser.parse_args()

    hscores = read_ref(args.human_scores)
    sscores = read_scores(args.system_scores)
    bscores = read_scores(args.baseline_scores)

    results, p = correlate_all_lps(hscores, bscores, sscores)

    if not args.just_scores:
        print(' '.join(sorted(results)))
        print(' '.join([str(results[lp][1]) for lp in sorted(results)]))
        
    for lp in sorted(results):
        sig_str = ''

        if p[lp] <= 0.001:
            sig_str = '***'
        elif p[lp] <= 0.01:
            sig_str = '**'
        elif p[lp] <= 0.05:
            sig_str = '*'
            
        print('& %.3f' % results[lp][0] + sig_str, end=' ')
    print('\\\\')
