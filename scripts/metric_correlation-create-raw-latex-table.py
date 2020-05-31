#!/bin/python

import metric_correlation_syslevel as mcsys
import metric_correlation_seglevel as mcseg
import os

def get_all_lang_correlations(mc, baseline, gold, system):
    # get correlation results
    baseline_results, system_results, sigs = mc.get_results(gold, baseline, system)

    lp_scores = []
    for lp in sorted(list(system_results.keys())):
        sig_txt = '' if sigs[lp] == '' else r'\textbf'
        score = system_results[lp][0]
        lp_scores.append(sig_txt + "{%.2f}" % score)

    # average of all
    ave = sum([system_results[lp][0] for lp in system_results]) / len(system_results)
    lp_scores.append('%.2f' % ave)
    print(' & '.join(lp_scores) + r' \\')


def write_table_content(testset, metric, level):


    mc = mcsys if level == 'sys' else mcseg
    thisdir=os.path.dirname(os.path.abspath(__file__)) + '/'
    if metric == 'bleu':
        baseline = thisdir + '../metric-scores/' + testset + '/sacreBLEU-' + level + 'level.tsv'
    else:
        baseline = thisdir + '../metric-scores/' + testset + '/Meteor-' + level + 'level.tsv'
    system_prefix = thisdir + '../metric-scores/' + testset + '/'
    gold_sys = thisdir + '../metrics-task/DA-syslevel-' + testset + '.csv'
    gold_seg = thisdir + '../metrics-task/RR-seglevel-' + testset + '.csv'
    gold = gold_sys if level == 'sys' else gold_seg
    
    # baseline
    if level == 'sys':
        print(r'\midrule' + '\n' + r'Baseline & ' + metric + r' & ', end=' ')
    else:
        print(r'\midrule' + '\n' + r'Baseline & sentence' + metric + r' & ', end=' ')
    system = baseline
    get_all_lang_correlations(mc, baseline, gold, system)

    # paraphrase baselines (+5)
    print(r'\midrule' + '\n' + r'\multirow{3}{*}{\pbox{1.5cm}{Paraphrase baselines (+5)}}')
    system = system_prefix + '/beam/par' + metric + '-beam.num=5-' + level + 'level.tsv'
    print(r' & \beam ', end=' & ')
    get_all_lang_correlations(mc, baseline, gold, system)
    system = system_prefix + '/random/par' + metric + '-random.num=5-' + level + 'level.tsv'
    print(r' & \random ', end=' & ')
    get_all_lang_correlations(mc, baseline, gold, system)
    system = system_prefix + '/sampled/par' + metric + '-sampled.num=5-' + level + 'level.tsv'
    print(r' & \sampled ', end=' & ')
    get_all_lang_correlations(mc, baseline, gold, system)
    
    # diversity (+1)
    print(r'\midrule' + '\n' + r'\multirow{3}{*}{\pbox{1.5cm}{Diversity (+1)}}')
    system = system_prefix + '/laser/par' + metric + '-laser.num=1-' + level + 'level.tsv'
    print(r' & \laser ', end=' & ')
    get_all_lang_correlations(mc, baseline, gold, system)
    system = system_prefix + '/treelstm/par' + metric + '-treelstm.num=1-' + level + 'level.tsv'
    print(r' & \treelstm ', end=' & ')
    get_all_lang_correlations(mc, baseline, gold, system)

    # diversity (+5)
    print(r'\midrule' + '\n' + r'\multirow{3}{*}{\pbox{1.5cm}{Diversity (+5)}}')
    system = system_prefix + '/laser/par' + metric + '-laser.num=5-' + level + 'level.tsv'
    print(r' & \laser ', end=' & ')
    get_all_lang_correlations(mc, baseline, gold, system)
    system = system_prefix + '/treelstm/par' + metric + '-treelstm.num=5-' + level + 'level.tsv'
    print(r' & \treelstm ', end=' & ')
    get_all_lang_correlations(mc, baseline, gold, system)
    
    # Output-specific (+1)
    print(r'\midrule' + '\n' + r'\multirow{3}{*}{\pbox{1.5cm}{Output-specific (+1)}}')
    system = system_prefix + '/mt-output-specific/par' + metric + '-constrained-laser.' + level + 'level.tsv'
    print(r' & \laser ', end=' & ')
    get_all_lang_correlations(mc, baseline, gold, system)
    system = system_prefix + '/mt-output-specific/par' + metric + '-constrained-treelstm.' + level + 'level.tsv'
    print(r' & \treelstm ', end=' & ')
    get_all_lang_correlations(mc, baseline, gold, system)

def write_raw_table(testset, metric, level):


    write_table_content(testset, metric, level)



if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('testset', choices=('newstest2018', 'newstest2019'))
    parser.add_argument('metric', choices=('bleu', 'meteor'))
    parser.add_argument('level', choices=('seg', 'sys'))
    args = parser.parse_args()

    write_raw_table(args.testset, args.metric, args.level)
