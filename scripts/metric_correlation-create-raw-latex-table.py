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
        lp_scores.append(sig_txt + "{%.3f}" % score)

    # average of all
    ave = sum([system_results[lp][0] for lp in system_results]) / len(system_results)
    lp_scores.append('%.3f' % ave)
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

    # list of languages
    langs = {}
    with open(baseline) as fp:
        for line in fp:
            lang = line.split()[1]
            lang.replace('-', '--')
            if '-en' in lang and lang not in langs:
                langs[lang] = 0
            if '-en' in lang:
                langs[lang] += 1

    print(' && ' + ' & '.join(sorted(list(langs.keys()))) + r' & Ave \\')
    print(' Approach & Method & (' + ') & ('.join([str(langs[x]) for x in sorted(list(langs.keys()))]) + r'\\')
    
    # baseline
    if level == 'sys':
        print(r'\midrule' + '\n' + r'Baseline & ' + '\\' + metric + r' & ', end=' ')
    else:
        print(r'\midrule' + '\n' + r'Baseline & sentence' + '\\' + metric + r' & ', end=' ')
    system = baseline
    get_all_lang_correlations(mc, baseline, gold, system)

    # paraphrase baselines (+5)
    print(r'\midrule' + '\n' + r'\multirow{3}{*}{\pbox{1.9cm}{Paraphrase baselines (+5)}}')
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
    print(r'\midrule' + '\n' + r'\multirow{2}{*}{\pbox{1.9cm}{Diversity (+1)}}')
    system = system_prefix + '/laser/par' + metric + '-laser.num=1-' + level + 'level.tsv'
    print(r' & \laser ', end=' & ')
    get_all_lang_correlations(mc, baseline, gold, system)
    system = system_prefix + '/treelstm/par' + metric + '-treelstm.num=1-' + level + 'level.tsv'
    print(r' & \treelstm ', end=' & ')
    get_all_lang_correlations(mc, baseline, gold, system)

    # diversity (+5)
    print(r'\midrule' + '\n' + r'\multirow{2}{*}{\pbox{1.9cm}{Diversity (+5)}}')
    system = system_prefix + '/laser/par' + metric + '-laser.num=5-' + level + 'level.tsv'
    print(r' & \laser ', end=' & ')
    get_all_lang_correlations(mc, baseline, gold, system)
    system = system_prefix + '/treelstm/par' + metric + '-treelstm.num=5-' + level + 'level.tsv'
    print(r' & \treelstm ', end=' & ')
    get_all_lang_correlations(mc, baseline, gold, system)
    
    # Output-specific (+1)
    print(r'\midrule' + '\n' + r'\multirow{2}{*}{\pbox{1.9cm}{Output-specific (+1)}}')
    system = system_prefix + '/mt-output-specific/par' + metric + '-constrained-laser.' + level + 'level.tsv'
    print(r' & \laser ', end=' & ')
    get_all_lang_correlations(mc, baseline, gold, system)
    system = system_prefix + '/mt-output-specific/par' + metric + '-constrained-treelstm.' + level + 'level.tsv'
    print(r' & \treelstm ', end=' & ')
    get_all_lang_correlations(mc, baseline, gold, system)

    
    print(r'\midrule')
    # constrained results (calculated externally)
    if metric == 'bleu':
        if level =='sys' and testset == 'newstest2019':
            print(r'constraints & 4-grams  & 0.922 & 0.983 & 0.809 & 0.963 & 0.989 & 0.924 & 0.921 & 0.930\\')
        elif level == 'seg' and testset == 'newstest2019':
            print(r'constraints & 4-grams & 0.090 & 0.242 & 0.161 & 0.271 & 0.323 & 0.122 & 0.314 & 0.218 \\')
        elif level == 'sys' and testset == 'newstest2018':
            print(r'constraints & 4-grams \\')
        elif level == 'seg' and testset == 'newstest2018':
            print(r'constraints & 4-grams \\')
    else:
        if level == 'sys' and testset== 'newstest2019':
            print(r'Constraints & 4-grams & 0.922 & 0.990 & 0.910 & 0.983 & 0.988 & 0.775 & 0.949 & 0.931 \\')
        elif level =='seg' and testset== 'newstest2019':
            print(r'Constraints & 4-grams & 0.098 & 0.237 & 0.193 & 0.272 & 0.318 & 0.145 & 0.351 & 0.230 \\')
        else:
            print(r'constraints & 4-grams \\')    
    

    print(r'\midrule')
    # best results (input manually)
    if level == 'sys' and testset == 'newstest2019':
        print(r'& WMT-19 best  & \textbf{0.950} & \textbf{0.995} & \textbf{0.993} & \textbf{0.998} & \textbf{0.989} & \textbf{0.979} & \textbf{0.988} & 0.985 \\')
        print(r'& & \tiny (\textsc{YiSi-1\_SRL}) & \tiny (\textsc{METEOR}) & \tiny (\textsc{YiSi-0}) & \tiny (\textsc{WMDO}) & \tiny (\textsc{ESIM}) & \tiny (\textsc{YiSi-1}) & \tiny (\textsc{ESIM}) \\')
        
    elif level == 'seg' and testset == 'newstest2019':
        print(r'& \tiny WMT-19 best  & \tiny\textbf{0.20} & \tiny\textbf{0.35} & \tiny\textbf{0.31} & \tiny\textbf{0.44} & \tiny\textbf{0.38} & \tiny\textbf{0.22}*** & \tiny\textbf{0.43} & 0.333 \\')
        print(r'& & \tiny \tiny \textsc{YiSi-1$_{\text{SRL}}$} & \tiny \textsc{YiSi-1} & \tiny \textsc{YiSi-1} & \tiny \textsc{YiSi-1$_{\text{SRL}}$} & \tiny \textsc{YiSi-1$_{\text{SRL}}$} & \tiny \textsc{YiSi-1$_{\text{SRL}}$} & \tiny \textsc{YiSi-1$_\text{SRL}$} \\')

    elif level == 'sys' and testset == 'newstest2018':
        print(r'& WMT-18 best  &  0.981 & \textbf{0.997} & 0.991 & \textbf{0.996} & \textbf{0.995} & \textbf{0.970} & 0.982 \\')
        print(r'& & \tiny (\textsc{RUSE})   & \tiny (\textsc{RUSE}) & \tiny (\textsc{WER}) & \tiny (\textsc{ITER}) & \tiny (\textsc{METEOR++}) & \tiny (\textsc{NIST}) & \tiny (\textsc{CDER}) \\')

    elif level == 'seg' and testset == 'newstest2018':
        print(r'& \tiny WMT-18 best  &  \tiny\textbf{0.35} & \tiny\textbf{0.50} & \tiny\textbf{0.37} & \tiny\textbf{0.27} & \tiny\textbf{0.31} & \tiny\textbf{0.26}*** & \tiny\textbf{0.22}  \\')
        print(r' & & \tiny \textsc{RUSE} & \tiny \textsc{RUSE} & \tiny \textsc{RUSE} & \tiny \textsc{RUSE} & \tiny \textsc{RUSE} & \tiny \textsc{RUSE} & \tiny \textsc{RUSE} \\')


        
        
def write_header():
    print(r'\scalebox{1}{')
    print(r'\centering\small')
    print(r'\begin{tabular}{lllllllllll}')
    print(r'\toprule')

    
def write_raw_table(testset, metric, level):

    write_header()
    write_table_content(testset, metric, level)
    print(r'\bottomrule')
    print(r'\end{tabular}}')


if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('testset', choices=('newstest2018', 'newstest2019'))
    parser.add_argument('metric', choices=('bleu', 'meteor'))
    parser.add_argument('level', choices=('seg', 'sys'))
    args = parser.parse_args()

    write_raw_table(args.testset, args.metric, args.level)
