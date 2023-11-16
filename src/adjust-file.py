if __name__ == '__main__':
    tmp = list()
    with open('TRoTR/judgments/results.tsv', mode='r', encoding='utf-8') as f:
        tmp.append(f.readline())
        f = f.read().replace('shur\n', 'shur@@@').replace('Nisha\n', 'Nisha@@@').replace('AndreaMariaC\n',
                                                                                         'AndreaMariaC@@@').replace(
            '\n', '--')
        lines = f.split('@@@')
        if len(set(lines[-1])) == 1: lines=lines[:-1]
        tmp.extend([line + '\n' for line in lines])
    with open('TRoTR/judgments/results.tsv', mode='w', encoding='utf-8') as f:
        f.writelines(tmp)