import os
import codecs
import argparse

def remove_bom(filename):
    """Removes BOM mark, if it exists, from a file and rewrites it in-place"""
    with open(filename, "r", encoding='utf-8') as csvfile:
        text = csvfile.read().strip(codecs.BOM_UTF8.decode(csvfile.encoding))
        with open(filename, "w", encoding='utf-8') as outfile:
            outfile.write(text)


def remove_bom(filename):
    """Removes BOM mark, if it exists, from a file and rewrites it in-place"""
    buffer_size = 4096
    bom_length = len(codecs.BOM_UTF8)

    with open(filename, "r+b") as fp:
        chunk = fp.read(buffer_size)
        if chunk.startswith(codecs.BOM_UTF8):
            i = 0
            chunk = chunk[bom_length:]
            while chunk:
                fp.seek(i)
                fp.write(chunk)
                i += len(chunk)
                fp.seek(bom_length, os.SEEK_CUR)
                chunk = fp.read(buffer_size)
            fp.seek(-bom_length, os.SEEK_CUR)
            fp.truncate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Random sampling', add_help=True)
    parser.add_argument('-f', '--filename',
                        type=str,
                        default='data/uses.tsv',
                        help='Convert from utf-8 with bom to utf-8')
    args = parser.parse_args()

    remove_bom(args.filename)

