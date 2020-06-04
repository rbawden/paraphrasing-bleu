#!/bin/python
import metric_correlation_syslevel as mcsys
import metric_correlation_seglevel as mcseg
import os

def get_all_lang_correlations(mc, baseline, gold, system):
    # get correlation results
    baseline_results, system_results, sigs = mc.get_results(gold, baseline, system)

    assert len(system_results) == 1 and 'de-en' in system_results
    
    sig_txt = '' if sigs['de-en'] == '' else r'\textbf'
            
    print(sig_txt + '{%.3f} ' % system_results['de-en'][0], end=' ')
            


def write_table(metric):

    testset = 'newstest2019'

    thisdir=os.path.dirname(os.path.abspath(__file__)) + '/'
    if metric == 'bleu':
        baseline = thisdir + '../metric-scores/' + testset + '/sacreBLEU'
    else:
        baseline = thisdir + '../metric-scores/' + testset + '/Meteor'
    system_prefix = thisdir + '../metric-scores/' + testset + '/'
    gold_sys = thisdir + '../metrics-task/DA-syslevel-' + testset + '.csv'
    gold_seg = thisdir + '../metrics-task/RR-seglevel-' + testset + '.500.csv'
        
    print(r'\centering\small')
    print(r'\resizebox{\linewidth}{!}{')
    print(r'\begin{tabular}{llll}')
    print(r'\toprule')
    print(r'&& \multicolumn{2}{c}{Correlation} \\')
    print(r'& Method & System & Segment \\')

    # baseline
    print(r'\midrule')
    print(r'Baseline & (sentence)' + '\\' +metric, end=' & ')
    get_all_lang_correlations(mcsys, baseline + '-syslevel.500.tsv', gold_sys, baseline + '-syslevel.500.tsv')
    print(r' & ', end=' ')
    get_all_lang_correlations(mcseg, baseline + '-seglevel.500.tsv', gold_seg, baseline + '-seglevel.500.tsv')
    print(r'\\')

    # Paraphrase baselines
    print(r'\midrule')
    print(r'\multirow{3}{*}{\pbox{1.9cm}{Baselines (+5)}}')
    print(r'& \beam & ', end=' ')
    get_all_lang_correlations(mcsys, baseline + '-syslevel.500.tsv', gold_sys, system_prefix + '/beam/par' + metric + '-beam.num=5-syslevel.500.tsv')
    print(r' & ', end=' ')
    get_all_lang_correlations(mcseg, baseline + '-seglevel.500.tsv', gold_seg, system_prefix + '/beam/par' + metric + '-beam.num=5-seglevel.500.tsv')
    print(r'\\')
    
    print(r'& \random & ', end=' ')
    get_all_lang_correlations(mcsys, baseline + '-syslevel.500.tsv', gold_sys, system_prefix + '/random/par' + metric + '-random.num=5-syslevel.500.tsv')
    print(r' & ', end=' ')
    get_all_lang_correlations(mcseg, baseline + '-seglevel.500.tsv', gold_seg, system_prefix + '/random/par' + metric + '-random.num=5-seglevel.500.tsv')
    print(r'\\')
    print(r'& \sampled & ', end=' ')
    get_all_lang_correlations(mcsys, baseline + '-syslevel.500.tsv', gold_sys, system_prefix + '/sampled/par' + metric + '-sampled.num=5-syslevel.500.tsv')
    print(r' & ', end=' ')
    get_all_lang_correlations(mcseg, baseline + '-seglevel.500.tsv', gold_seg, system_prefix + '/sampled/par' + metric + '-sampled.num=5-seglevel.500.tsv')
    print(r'\\')
    
    # Diversity (+1)
    print(r'\midrule')
    print(r'\multirow{2}{*}{\pbox{1.9cm}{Diversity (+1)}}')
    print(r'& \laser & ', end=' ')
    get_all_lang_correlations(mcsys, baseline + '-syslevel.500.tsv', gold_sys, system_prefix + '/laser/par' + metric + '-laser.num=1-syslevel.500.tsv')
    print(r' & ', end=' ')
    get_all_lang_correlations(mcseg, baseline + '-seglevel.500.tsv', gold_seg, system_prefix + '/laser/par' + metric + '-laser.num=1-seglevel.500.tsv')
    print(r'\\')
    
    print(r'& \treelstm & ', end=' ')
    get_all_lang_correlations(mcsys, baseline + '-syslevel.500.tsv', gold_sys, system_prefix + '/treelstm/par' + metric + '-treelstm.num=1-syslevel.500.tsv')
    print(r' & ', end=' ')
    get_all_lang_correlations(mcseg, baseline + '-seglevel.500.tsv', gold_seg, system_prefix + '/treelstm/par' + metric + '-treelstm.num=1-seglevel.500.tsv')
    print(r'\\')
    
    # Diversity (+5)
    print(r'\midrule')
    print(r'\multirow{2}{*}{\pbox{1.9cm}{Diversity (+5)}}')
    print(r'& \laser & ', end=' ')
    get_all_lang_correlations(mcsys, baseline + '-syslevel.500.tsv', gold_sys, system_prefix + '/laser/par' + metric + '-laser.num=5-syslevel.500.tsv')
    print(r' & ', end=' ')
    get_all_lang_correlations(mcseg, baseline + '-seglevel.500.tsv', gold_seg, system_prefix + '/laser/par' + metric + '-laser.num=5-seglevel.500.tsv')
    print(r'\\')
    
    print(r'& \treelstm & ', end=' ')
    get_all_lang_correlations(mcsys, baseline + '-syslevel.500.tsv', gold_sys, system_prefix + '/treelstm/par' + metric + '-treelstm.num=5-syslevel.500.tsv')
    print(r' & ', end=' ')
    get_all_lang_correlations(mcseg, baseline + '-seglevel.500.tsv', gold_seg, system_prefix + '/treelstm/par' + metric + '-treelstm.num=5-seglevel.500.tsv')
    print(r'\\')

    # Output-specific (+1)
    print(r'\midrule')
    print(r'\multirow{2}{*}{\pbox{1.9cm}{Output-specific (+1)}}')
    print(r'& \laser & ', end=' ')
    get_all_lang_correlations(mcsys, baseline + '-syslevel.500.tsv', gold_sys, system_prefix + '/mt-output-specific/par' + metric + '-constrained-laser.syslevel.500.tsv')
    print(r' & ', end=' ')
    get_all_lang_correlations(mcseg, baseline + '-seglevel.500.tsv', gold_seg, system_prefix + '/mt-output-specific/par' + metric + '-constrained-laser.seglevel.500.tsv')
    print(r'\\')

    print(r'& \treelstm & ', end=' ')
    get_all_lang_correlations(mcsys, baseline + '-syslevel.500.tsv', gold_sys, system_prefix + '/mt-output-specific/par' + metric + '-constrained-treelstm.syslevel.500.tsv')
    print(r' & ', end=' ')
    get_all_lang_correlations(mcseg, baseline + '-seglevel.500.tsv', gold_seg, system_prefix + '/mt-output-specific/par' + metric + '-constrained-treelstm.seglevel.500.tsv')
    print(r'\\')

    # Constrained outputs (calculated externally)
    print(r'\midrule')
    print(r'\multirow{1}{*}{Constraints}')
    print(r'& 4-gram & 0.933 & 0.064 \\') 
    
    # Human results
    print('\midrule')
    print(r'&& Human & ', end=' ')
    get_all_lang_correlations(mcsys, baseline + '-syslevel.500.tsv', gold_sys, system_prefix + '/human/par' + metric + '-human-syslevel.500.tsv')
    print(r' & ', end=' ')
    get_all_lang_correlations(mcseg, baseline + '-seglevel.500.tsv', gold_seg, system_prefix + '/human/par' + metric + '-human-seglevel.500.tsv')
    print(r'\\')


    # Footer
    print(r'\bottomrule')
    print(r'\end{tabular}}')
    print(r'\caption{Correlations on the 500-sentence subset.}')
    print(r'\label{tab:subset-seg-level}')
    
if __name__ == '__main__':

    write_table('bleu')
