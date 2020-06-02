#!/usr/bin/env python3

import json
import sys

def main(args):
    
    cache = {}
    for line in open(args.translations):
        j = json.loads(line)
        cache[(j['sentno'], j['constraints'][0])] = j

    for line in sys.stdin:
        j = json.loads(line)
        print(json.dumps(cache[(j['sentno'], j['constraints'][0])], ensure_ascii=False), flush=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('translations')
    args = parser.parse_args()

    main(args)
