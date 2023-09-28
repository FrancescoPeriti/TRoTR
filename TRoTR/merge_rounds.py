from pathlib import Path

def merge_rounds(folder):
    record_list = list()
    for i, filename in enumerate(Path(folder).glob('*.tsv')):
        filename = str(filename)
        records = open(filename, mode='r', encoding='utf-8').readlines()
        records = [f'{line}\n' if line[-1] != '\n' else line for line in records]

        if 'quality-check' in filename or 'TRoTR.tsv' in filename:
            continue

        # remove header
        if i > 0:
            records = records[1:]

        record_list.extend(records)

    return record_list

if __name__ == '__main__':

    rounds = merge_rounds('rounds')
    judgments = merge_rounds('judgments')

    open('rounds/TRoTR.tsv', mode='w', encoding='utf-8').writelines(rounds)
    open('judgments/TRoTR.tsv', mode='w', encoding='utf-8').writelines(judgments)