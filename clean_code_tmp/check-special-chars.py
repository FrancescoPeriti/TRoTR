import json
import string
import argparse

# Argument parser
parser = argparse.ArgumentParser(prog='jsonl corpus', add_help=True)
parser.add_argument('-i', '--input',
                    type=str,
                    help='input corpus')
args = parser.parse_args()

if __name__ == '__main__':
    filename = args.input

    s = set()
    quotes = set()
    for i,line in enumerate(open(filename, mode='r', encoding='utf-8').readlines()):
        line = line.strip()
        dict_ = json.loads(line)

        if not 'topic' in dict_.keys():
            dict_['topic'] = ""

        quotes.add(dict_['quote_id'])

        if dict_['quote'].lower() not in dict_['sentence'].lower() and 'start' not in dict_:
            print(dict_['sentence'])

        for c in dict_['sentence']:
            if c not in string.punctuation and c not in string.ascii_letters:
                s.add(c)

        s.update(list(dict_.keys()))

    print(s)
    print(sorted(list(quotes)))
