import argparse
import csv
from parser_folder.DFG import DFG_python, DFG_java, DFG_c
from parser_folder import (remove_comments_and_docstrings,
                           tree_to_token_index,
                           index_to_code_token,
                           tree_to_variable_index)
from tree_sitter import Language, Parser
import sys
sys.path.append('.')
sys.path.append('../')
from utils import is_valid_variable_name
from run_parser import extract_dataflow, unique, get_identifiers
path = 'parser_folder/my-languages.so'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default=None, type=str,
                        help="language.")
    args = parser.parse_args()
    f_write = open('attack_c_bible.csv', 'w')
    writer = csv.writer(f_write)
    with open('attack_result_c.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                write_data = ['Original Code', 'Extracted Variable']
                writer.writerow(write_data)
                line_count += 1
            elif line_count == 50:
                break;
            else:
                data, _ = get_identifiers(row[0], args.lang)
                write_data = [row[0], data]
                writer.writerow(write_data)
                line_count += 1
        print(f'Processed {line_count} lines.')

    f_write.close()


if __name__ == '__main__':
    main()

# import zlib

# # Original string
# original_string = "This is a sample string to compress using zlib."
# print(len(orit))
# # Convert the string to bytes (zlib works with bytes)
# bytes_to_compress = original_string.encode('utf-8')

# # Compress the string using zlib
# compressed_data = zlib.compress(bytes_to_compress, 9)

# # Decompress the compressed data (for verification purposes)
# decompressed_data = zlib.decompress(compressed_data)
# print(len(compressed_data))
# # Convert the decompressed bytes back to string
# decompressed_string = decompressed_data.decode('utf-8')

# # Print results
# print("Original String:", original_string)
# print("Compressed Data:", compressed_data)
# print("Decompressed String:", decompressed_string)