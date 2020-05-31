#!/usr/bin/env python3
import os
import sys
import re

from sacrebleu.sacrebleu import sentence_bleu, TOKENIZERS
from subprocess import check_output

thisdir=os.path.dirname(os.path.abspath(__file__))
TREE_KERNEL_TOOL=thisdir + '/../tools/TreeKernel'

'''
Read one paraphrase file (one translation per line)
'''
def read_file(p_file):
    contents = []
    with open(p_file) as fp:
        for line in fp:
            contents.append(line.strip())
    return contents

'''
Read the paraphrased references

Args:

- para_folder: the path to the folder containing the paraphrases. Files should be named
               $langpair-n.en, where n is a number in the beam.
- [-r reference_folder]: (optional) the path to the folder containing the reference files (whose 
               names follow are of the form 'newstest2018-csen-ref.en')
- [-n max_n]: a maximum number of paraphrases to include (default=-1, do not filter)

Returns a dictionary containing for each language pair a list of tuples 
(as many as there are sentences), each containing n paraphrases
'''
def read_files(para_folder, reference_folder=None, langpair=None, max_n=-1, secondary_para_folder=None):
    paras = {}

    if reference_folder is not None:
        os.sys.stderr.write("Reading references...")
        for reffile in os.listdir(reference_folder):
            refmatch = re.match('.+?\-(....)\-ref\.en(\.parse)?', reffile)
            if not refmatch:
                continue
            lang = refmatch.group(1)
            if langpair is not None and lang != langpair:
                continue
            paras[lang] = [read_file(reference_folder + '/' + reffile)]
        print('Done ' +  str(len(paras)) + ' references')

    os.sys.stderr.write("Reading synthetic references...")
    filelist = [para_folder + '/' + x for x in os.listdir(para_folder) if re.match('.+?.en(\.parse)?$', x)]
    filelist.extend([secondary_para_folder + '/' + x for x in os.listdir(secondary_para_folder) if re.match('.+?.en(\.parse)?$', x)])


    for parafile in sorted(filelist, key=os.path.getmtime):
        # check that we want to look at this file

        # ignore file in these cases
        matchname = re.match('.+?([^/]+)\-(\d+)\...', parafile)
        if not matchname:
            continue
        num = matchname.group(2)
        lang = matchname.group(1)
        testset = re.match('.*?(newstest201[89])', parafile).group(1)
        if max_n != -1 and int(num) > max_n:
            continue

        if langpair is not None and lang != langpair:
            continue

        if lang + '-' + testset not in paras:
            paras[lang + '-' + testset] = []
        
        # add to paraphrases
        paras[lang + '-' + testset].append(read_file(parafile))

                  

    for lang in paras.keys():
        print('Done. Read ' + str(len(paras[lang])) + ' paraphrases for lang ' + lang)

    # organise by same sentence
    for lang in paras:
        zip_paras = list(zip(*paras[lang]))
        paras[lang] = zip_paras

    return paras


'''
BLEU-based diversity metric
'''
class BLEU:

    def __init__(self):
        pass

    def __call__(self, sent1, sent2):
        dp = (1 - sentence_bleu(sent1, sent2, smooth_method='exp').score / 100)
        return dp

'''
BOW lexical overlap
'''
class BOW:
    def __init__(self):
        pass

    def __call__(self, sent1, sent2):
        tok1 = TOKENIZERS['13a'](sent1).split(' ')
        tok2 = TOKENIZERS['13a'](sent2).split(' ')
        inter = set(tok1).intersection(tok2)
        lengths = (len(tok1)+len(tok2))/2
        dp = 1 - len(inter)/lengths
        return dp


def escape(string):
    for char in ['`', '"', "'"]:
        string = string.replace(char,'\\' + char)
    return string

'''
Tree kernel diversity metric
'''
class TreeKernel:

    def __init__(self):
        pass

    def __call__(self, sent1, sent2):

        command = 'echo \"' + escape(sent1) +'\t' + escape(sent2) + '\" | ' + TREE_KERNEL_TOOL + '/tree-kernel/compare-trees'

        result = float(check_output(command, shell=True))

        if str(result) != "nan":
            result = 1- result
        return result

'''
Calculates the diversity of the paraphrases
'''
def diversity(all_paras, metric_type, lowercase=True):

    if metric_type == 'bleu':
        metric = BLEU()
    elif metric_type == 'bow':
        metric = BOW()
    elif metric_type == 'syntax':
        metric = TreeKernel()

    os.sys.stderr.write("Calculating diversity metric "+ metric_type + "\n")
    dps = {}
    all_dp = 0
    for lang in sorted(list(all_paras.keys())):
        print('Doing ' + lang + '...\r')

        # sentence by sentence
        lang_dp = 0
        for p, paras in enumerate(all_paras[lang]):
            os.sys.stderr.write("\t\t" + str(p) + '\r')

            # Compute all pairs (once per pair)
            sentence_dp = 0
            num_comparisons = 0

            # do full set of comparisons for bleu meitrc
            if metric_type == 'bleu':
                for p1, p1str in enumerate(paras):
                    for p2, p2str in enumerate(paras):
                        if p1 != p2:
                            sentence_dp += metric(p1str, p2str)
                            num_comparisons += 1
            else:
                for p1, p1str in enumerate(paras):
                    for p2, p2str in enumerate(paras[p1+1:], p1+1):
                        score = metric(p1str, p2str)
                        if str(score) != 'nan':
                            
                            sentence_dp += score
                            num_comparisons += 1
            if num_comparisons > 0:
                sentence_dp /= num_comparisons
                lang_dp += sentence_dp

        num_sentences = len(all_paras[lang])

        # average DP for the language        
        average_sentence_dp = lang_dp / num_sentences
        dps[lang] = average_sentence_dp
        all_dp += lang_dp

        print('Done.' + lang + ' = ' + str(dps[lang]))

    # average over all languages
    print([len(all_paras[lang]) for lang in all_paras])
    all_dp /= float(sum([len(all_paras[lang]) for lang in all_paras]))

    print('DP ' + metric_type + ' averaged over all languages = ' + str(all_dp))
    # print out
    print("Per language direction = ")
    for lang in dps:
        print('\t' + lang + ' = ' + str(dps[lang]) + '\tCalculated on ' + str(len(all_paras[lang])) + ' sentences')
        
    return dps, all_dp


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('paraphrase_folder', help='folder containing referenced paraphrases')
    parser.add_argument('metric', default='bleu', choices = ['bleu', 'bow', 'syntax'])
    parser.add_argument('--reference-folder', '-r', default=None, help='original reference folder')
    parser.add_argument('--langpair', '-l', default=None)
    parser.add_argument('--n', '-n', default=-1, type=int)
    parser.add_argument('-s', '--secondary_paraphrase_folder', help='secondary reference folder to be included in analysis (e.g. for other year)')
    args = parser.parse_args()
    

    paras = read_files(args.paraphrase_folder, args.reference_folder, args.langpair, args.n, args.secondary_paraphrase_folder)
    diversity(paras, args.metric, lowercase=True)
