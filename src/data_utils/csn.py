import json 
import os 
from pathlib import Path 
from loguru import logger
import numpy as np 
from src.data_utils.dfg_parser.run_parser import get_identifiers

from concurrent.futures import ProcessPoolExecutor
from more_itertools import chunked
from torch.utils.data import Dataset
from src.common.utils import set_seed
import nltk
import re
from io import StringIO
import tokenize
from loguru import logger


STRING_MATCHING_PATTERN = re.compile(r'([bruf]*)(\"\"\"|\'\'\'|\"|\')(?:(?!\2)(?:\\.|[^\\]))*\2')
def replace_string_literal(source, target='___STR'):
    return re.sub(pattern=STRING_MATCHING_PATTERN, repl=target, string=source)

def remove_comments_and_docstrings(source, lang):
    if lang == 'python':
        try:
            io_obj = StringIO(source)
            out = ""
            prev_token_type = tokenize.INDENT
            last_lineno = -1
            last_col = 0
            for tok in tokenize.generate_tokens(io_obj.readline):
                token_type = tok[0]
                token_string = tok[1]
                start_line, start_col = tok[2]
                end_line, end_col = tok[3]
                # l_text = tok[4]
                if start_line > last_lineno:
                    last_col = 0
                if start_col > last_col:
                    out += (" " * (start_col - last_col))
                # Remove comments:
                if token_type == tokenize.COMMENT:
                    pass
                # This series of conditionals removes docstrings:
                elif token_type == tokenize.STRING:
                    if prev_token_type != tokenize.INDENT:
                        # This is likely a docstring; double-check we're not inside an operator:
                        if prev_token_type != tokenize.NEWLINE:
                            if start_col > 0:
                                out += token_string
                else:
                    out += token_string
                prev_token_type = token_type
                last_col = end_col
                last_lineno = end_line
            temp = []
            for x in out.split('\n'):
                if x.strip() != "":
                    temp.append(x)
            return '\n'.join(temp)
        except Exception:
            return source
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " "  # note: a space and not an empty string
            else:
                return s

        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp = []
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)


def clean_doc(s):
    """
    Clean docstring.

    Args:
        s (str): Raw docstring

    Returns:
        str: Cleaned docstring

    """
    # // Create an instance of  {@link RepresentationBaseType } and {@link RepresentationBaseType }.
    # // Create an instance of RepresentationBaseType and RepresentationBaseType
    # // Public setter for the {@code rowMapper}.
    # // Public setter for the rowMapper
    # comment = comment.replaceAll("\\{@link|code(.*?)}", "$1");
    # comment = comment.replaceAll("@see", "");

    s = re.sub(r'{@link|code(.*?)}', r'\1', s)
    s = re.sub(r'@see', '', s)

    # // Implementation of the <a href="http://www.tarsnap.com/scrypt/scrypt.pdf"/>scrypt KDF</a>.
    # // Implementation of the scrypt KDF
    # comment = comment.replaceAll("<a.*?>(.*?)a>", "$1");
    s = re.sub(r'<a.*?>(.*?)a>', r'\1', s)

    # // remove all tags like <p>, </b>
    # comment = comment.replaceAll("</?[A-Za-z0-9]+>", "");
    s = re.sub(r'</?[A-Za-z0-9]+>', '', s)

    # // Set the list of the watchable objects (meta data).
    # // Set the list of the watchable objects
    # comment = comment.replaceAll("\\(.*?\\)", "");
    s = re.sub(r'\(.*?\)', '', s)

    # // #dispatchMessage dispatchMessage
    # // dispatchMessage
    # comment = comment.replaceAll("#([\\w]+)\\s+\\1", "$1");
    s = re.sub(r'#([\w]+)\s+\1', r'\1', s)

    # // remove http url
    # comment = comment.replaceAll("http\\S*", "");
    s = re.sub(r'http\S*', '', s)

    # // characters except english and number are ignored.
    # comment = comment.replaceAll("[^a-zA-Z0-9_]", " ");
    s = re.sub(r'[^a-zA-Z0-9_]', ' ', s)

    # // delete empty symbols
    # comment = comment.replaceAll("[ \f\n\r\t]", " ").trim();
    # comment = comment.replaceAll(" +", " ");
    s = re.sub(r'[ \f\n\r\t]', ' ', s).strip()
    s = re.sub(r' +', ' ', s).strip()

    if len(s) == 0 or len(s.split()) < 3:
        return None
    else:
        return s
    

def regular_tokenize(source: str):
    # source = re.sub(r'(\S)[.=](\S)', r'\1 . \2', source)
    source = re.sub(r'(\S)[.](\S)', r'\1 . \2', source)
    source = re.sub(r'(\S)[=](\S)', r'\1 = \2', source)

    return ' '.join(nltk.word_tokenize(source))

def process_line(json_line):
    holder = []
    for line in json_line:
        dic = {}
        
        code = remove_comments_and_docstrings(line['original_string'], line['language'])
        try:
            identifiers, _ = get_identifiers(code, line['language'])
        except:
            continue
        if identifiers == None:
            continue
        
        dic['identifiers'] = [i[0] for i in identifiers]
        dic['code_wo_comment'] = code
        dic['original_code'] = line['original_string']
        
        code_tokens = replace_string_literal(code)
        code_tokens = regular_tokenize(code_tokens)
        
        dic['code_tokens'] = code_tokens
        dic['language'] = line['language']
        
        doc = clean_doc(line['docstring'])
        dic['docstring'] = doc if doc is not None else "" 
        dic['code_token_len'] = len(line['code_tokens'])
        
        holder.append(dic) 
    return holder

def load_pretrain_code():
    res = []

    files = [
                'cache/csn_python/test.jsonl',
                'cache/csn_python/valid.jsonl',
                'cache/csn_python/train.jsonl',
            ]
    
    tmp = []
    for file in files:
        with open(file, 'r') as f:
            for line in f.readlines():
                line = json.loads(line.split('\n')[0])
                tmp.append(line)
    logger.info(f'total lines:{len(tmp)}')
    
    tmp_batch = chunked(tmp, 20000)
    with ProcessPoolExecutor() as executor:
        jobs = []
        for batch in tmp_batch:
            jobs.append(executor.submit(process_line, batch))
        for job in jobs:
            res += job.result()
    return res

def generate_csn(saved_root='cache/csn_mia/pretrain_v1', save_chunked=False):
    
    if not Path(saved_root).exists():
        Path(saved_root).mkdir(exist_ok=True, parents=True)
    
    data_path = Path(saved_root) / 'csn_mia_all.json'
    
    if not data_path.exists():
        code = load_pretrain_code()
            
        logger.info(f'Number before process:{len(code)}')
        
        lengths = [i['code_token_len'] for i in code]
        lengths = np.array(lengths, dtype=np.int32)
        mean = lengths.mean()
        std = lengths.std()

        code = list(filter(lambda x: x['code_token_len'] >= mean - 3 * std \
                        and x['code_token_len'] <= mean + 3 * std and x['docstring'] != "" \
                        and len(x['identifiers']) != 0, code))
        logger.info(f'Number after process:{len(code)}')
        
        if save_chunked:
            tmp_batch = chunked(code, 10000)
            for idx, batch in enumerate(tmp_batch):
                tmp_root = Path(saved_root) / 'observe' 
                if not tmp_root.exists():
                    tmp_root.mkdir(exist_ok=True, parents=True)
                tmp_path = tmp_root / f'pretrain_{idx}.json'
                json.dump(batch, open(tmp_path, 'w'), indent=1)
                
        json.dump(code, open(data_path, 'w'), indent=1)
    else:
        code = json.load(open(data_path, 'r'))
    return code


class CsnMiaData:
    def __init__(self, args=None, tokenizer=None):
        self.data_root = Path('cache/csn_mia/pretrain_v1')

    def generate_csn(self):
        return generate_csn(save_chunked=True)
    
    def generate_data(self):
        csn_all = self.generate_csn()
        
        np.random.shuffle(csn_all)
        
        mem_pretrain = csn_all[:100000] # for target model
        non_shadow = csn_all[100000:150000] # for shadow model
        non_surrogate = csn_all[150000:180000]  
        non_test = csn_all[180000:190000] # for testing
        non_caliberate = csn_all[190000:240000] # for calibration model
        non_utils = csn_all[240000:] # for training inference model
        mem_test = mem_pretrain[:10000] # for testing
        
        '''
        数据描述：
        前10w条数据用来预训练目标模型，其中划分10000条出来作为测试
        第10w-15w用来作为影子模型的训练数据，
        第15w-18w暂时没用
        第18w-19w用来作为非成员变量的测试数据
        第19w-24w用来作为矫正模型的训练数据
        第24w之后作为分类模型的非成员变量训练数据。
        
        分类模型的训练数据：
            非成员变量，也就是label为0的数据，来自non_utils
            成员变量，也就是label为1的数据，白盒来自mem_pretrain，灰盒来自non_shadow
        '''
        
        json.dump(mem_pretrain, open(os.path.join(self.data_root, 'mem_pretrain.json'), 'w'), indent=1)
        json.dump(non_shadow, open(os.path.join(self.data_root, 'non_shadow.json'), 'w'), indent=1)
        json.dump(non_surrogate, open(os.path.join(self.data_root, 'non_surrogate.json'), 'w'), indent=1)
        json.dump(non_test, open(os.path.join(self.data_root, 'non_test.json'), 'w'), indent=1)
        json.dump(non_utils, open(os.path.join(self.data_root, 'non_utils.json'), 'w'), indent=1)
        json.dump(mem_test, open(os.path.join(self.data_root, 'mem_test.json'), 'w'), indent=1)
        json.dump(non_caliberate, open(os.path.join(self.data_root, 'non_caliberate.json'), 'w'), indent=1)
        
        white_box_mem_train = mem_pretrain[10000:]
        white_box_non_train = non_shadow + non_surrogate
        white_box_mem_test = mem_test # 和mem test是一样的
        white_box_non_test = non_test
        
        json.dump(white_box_mem_train, open(os.path.join(self.data_root, 'wb_mem_train.json'), 'w'), indent=1)
        json.dump(white_box_mem_test, open(os.path.join(self.data_root, 'wb_mem_test.json'), 'w'), indent=1)
        json.dump(white_box_non_train, open(os.path.join(self.data_root, 'wb_non_train.json'), 'w'), indent=1)
        json.dump(white_box_non_test, open(os.path.join(self.data_root, 'wb_non_test.json'), 'w'), indent=1)

    def data(self, data_type='mem_pre'):        
        return json.load(open(os.path.join(self.data_root, f'{data_type}.json')))



if __name__ == "__main__":
    set_seed(42)
    
    a = CsnMiaData()
    a.generate_data()
    
    pretrain = a.data('mem_pretrain')
    logger.info(len(pretrain))
    
    shadow_model = a.data('non_shadow')
    logger.info(len(shadow_model))
    
    surrogate_model = a.data('non_surrogate')
    logger.info(len(surrogate_model))
    
    caliberate_model = a.data('non_caliberate')
    logger.info(len(caliberate_model))


