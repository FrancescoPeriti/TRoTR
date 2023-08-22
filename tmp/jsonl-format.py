import json
import argparse

# Argument parser
parser = argparse.ArgumentParser(prog='jsonl corpus', add_help=True)
parser.add_argument('-i', '--input',
                    type=str,
                    help='input corpus')
parser.add_argument('-o', '--output',
                    type=str,
                    help='output corpus')
parser.add_argument('-m', '--marker',
                    type=str,
                    help='corpus marker')
args = parser.parse_args()


if __name__ == '__main__':
    filename = args.input
    out = args.output

    # jsonl keys
    keys = ['id', 'topic', 'quote_id', 'sentence', 'quote', 'start', 'end', 'date']
    
    idx = 0
    records = list()
    for line in open(filename, mode='r', encoding='utf-8').readlines():
        line = line.strip()
        
        try:
            dict_ = json.loads(line)
        except:
            print('There is an error with this line:\n')
            print(line)
            raise Exception

        if 'sent' in dict_:
            dict_['sentence'] = dict_.pop('sent')

        if not 'topic' in dict_.keys():
            dict_['topic'] = ""

        if not 'date' in dict_.keys():
            dict_['date'] = ""

        if 'start' not in dict_:
            quote = dict_['quote'].lower()
            dict_['start'] = dict_['sentence'].lower().find(quote)
            dict_['end'] = dict_['start'] + len(quote)

        if dict_['start'] == -1:
            print('There is an error with this line:\n')
            print(line)
            raise Exception
      
        dict_['id'] = f'{idx}_{dict_["quote_id"]}' + '-' + args.marker
        idx+=1
        records.append(json.dumps({k: dict_[k] for k in keys})+'\n')

    with open(out, mode='w', encoding='utf-8') as f:
        f.writelines(records)
