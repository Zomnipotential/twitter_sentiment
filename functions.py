import packages as pkg
import constants as const

def string_affected_tokens(corpus: str, signum: str, omnia_signa: str) -> bool:
    """Generate a list of all the tokens in the corpus that contain a particular string.

    It also counts the number of tokens that contain the string, and prints it to the console together with its ratio to the whole corpus.

    Parameters:
    corpus (str): name of a txt-file in the repository at const.repo_path
    signum (str): the token whose ocurrences in the corpus are to be listed and counted
    omnia_signa (str): name of the txt-file to which we want to save the list of all the tokens that contain the string

    Returns:
    str: the name of the txt-file that contains the list of all the tokens that contain the string
    """
    print('signum = ',signum)
    if type(corpus) == type(signum) == type(omnia_signa) == str:
        pattern = r'\S*' + pkg.re.escape(signum) + r'\S*'
        all_tokens = []
        with open(''.join([const.repo_path, corpus, '.txt']), 'r') as f:
            text = f.read()
            all_tokens = pkg.re.findall(pattern, text)
            if not len(all_tokens) == 0:
                corpus_supplementum = ''.join([const.repo_path, omnia_signa, '.txt'])
                print(f'There are {format(len(all_tokens), ",")} tokens in \'{corpus_supplementum}\' containing \'{signum}\'')
                with open(corpus_supplementum, 'w') as f:
                        f.write('  '.join(all_tokens))
                return True
    else:
        raise TypeError('All three "corpus", "signum", and "omnia_signa_continens" must be strings. Don\'t include the file extension!')
    
    return False

def list_affected_tokens_detailed(corpus: str, signa: dict, omnia_signa: str) -> bool:
    """Generate a text with all the tokens in the corpus that contain the strings in the list 'signa'.

    It also counts the number of tokens that contain the strings, and prints it to the console together with its ratio to the whole corpus.

    Parameters:
    corpus (str): name of a txt-file in the repository at const.repo_path
    signa (dict): a dictionary of tokens, with explanation of its meaning, whose ocurrences in the corpus are to be listed and counted
    omnia_signa (str): name of the txt-file to which we want to save the list of all the tokens that contain the string

    Returns:
    str: the name of the txt-file, including the path 'const.repo_path', that contains the list of all the tokens that contain the string
    """
    global print
    print = pkg.functools.partial(print, flush=True)

    with open(''.join([const.repo_path, 'corpus_mensura', 'txt']), 'r') as f:
        const.CORPUS_MENSURA
    if type(corpus) == type(omnia_signa) == str and type(signa) == dict:
        with open(''.join([const.repo_path, corpus, '.txt']), 'r') as f:
            text = f.read()
            corpus_size = len(text)
            omnes_affectus_signa = ''
            locus = ''.join([const.repo_path, omnia_signa, '.txt'])
            for signum, valorem in signa.items():
                omnes_affectus_signa += "'" + signum + "' = '" + valorem + "'\n\n"
                pattern = r'\S*' + pkg.re.escape(signum) + r'\S*'
                omnes_affectus_signa += '  '.join(pkg.re.findall(pattern, text))
                print(omnes_affectus_signa)
                omnes_affectus_signa += '\n\n\n'
                if not len(omnes_affectus_signa) == 0:
                    locus = ''.join([const.repo_path, omnia_signa, '.txt'])
                    print(f'There are {format(len(omnes_affectus_signa), ",")} tokens in \'{locus}\' containing \'{" ".join(list(signum))}\'')
                    print(f'This makes up {"{:.2f}".format(float(len(omnes_affectus_signa))/float(const.CORPUS_MENSURA)*100)}%'+
                        f' of the whole \'{corpus}.txt\'')
                    print(f'Writing all affected tokens to \'{locus}\' ...')
                    with open(locus, 'w') as f:
                        for symbolum in omnes_affectus_signa:
                            f.write(symbolum)
            return True
    else:
        raise TypeError('"corpus" and "omnia_signa", must be string, file names, and "signa" a dictionary. Don\'t include the file extensions!')
    return False

def list_affected_tokens(corpus: str, signa: dict, omnia_signa: str) -> bool:
    """Generate a text with all the tokens in the corpus that contain the strings in the list 'signa'.

    It also counts the number of tokens that contain the strings, and prints it to the console together with its ratio to the whole corpus.

    Parameters:
    corpus (str): name of a txt-file in the repository at const.repo_path
    signa (dict): a dictionary of tokens, with explanation of its meaning, whose ocurrences in the corpus are to be listed and counted
    omnia_signa (str): name of the txt-file to which we want to save the list of all the tokens that contain the string

    Returns:
    str: the name of the txt-file, including the path 'const.repo_path', that contains the list of all the tokens that contain the string
    """
    global print
    print = pkg.functools.partial(print, flush=True)


    if type(corpus) == type(omnia_signa) == str and type(signa) == dict:
        with open(''.join([const.repo_path, corpus, '.txt']), 'r') as f:
            text = f.read()
            corpus_size = len(text)
            omnes_affectus_signa = ''
            for signum, valorem in signa.items():
                omnes_affectus_signa += "'" + signum + "' = '" + valorem + "'\n\n"
                pattern = r'\S*' + pkg.re.escape(signum) + r'\S*'
                omnes_affectus_signa += '  '.join(pkg.re.findall(pattern, text))
                omnes_affectus_signa += '\n\n\n'
            if not len(omnes_affectus_signa) == 0:
                locus = ''.join([const.repo_path, omnia_signa, '.txt'])
                print(f'There are {format(len(omnes_affectus_signa), ",")} tokens in \'{locus}\' containing \'{"   ".join(list(signa.keys()))}\'')
                print(f'This makes up {"{:.2f}".format(float(len(omnes_affectus_signa))/float(const.CORPUS_MENSURA)*100)}%'+
                    f' of the whole \'{corpus}.txt\'')
                print(f'Writing all affected tokens to \'{locus}\' ...')
                with open(locus, 'w') as f:
                    for symbolum in omnes_affectus_signa:
                        f.write(symbolum)
                return True
            return False
    else:
        raise TypeError('"corpus" and "omnia_signa", must be string, file names, and "signa" a dictionary. Don\'t include the file extensions!')

def remove_token(corpus: str, signum: str, corpus_sine_signis: str) -> str:
    """Remove all "unnecessary" tokens from the corpus, that contain the string 'signum'.

    It also counts the number of tokens that contain the strings, and prints it to the console together with its ratio to the whole corpus.

    Parameters:
    corpus (str): name of a txt-file in the repository
    signum (str): the token whose all occrrences are to be removed from the 'corpus'
    cropus_sine_signis (string) name of the txt-file to which the result of the removal are saved; includes the path 'const.repo_path'

    Returns:
    str: the name of the txt-file, including the path 'const.repo_path', that contains the result of the removal
    """
#    with open(''.join([const.repo_path, 'CORPUS_MENSURA.txt']), 'r') as f:
#        cpma = int(f.read())
    if type(corpus) == type(signum) == type(corpus_sine_signis) == str:
        vetus_corpus = ''.join([const.repo_path, corpus, '.txt'])
        with open(vetus_corpus, 'r') as f:
            text = f.read()
            metam = pkg.re.sub(r'\S*'+signum+r'\S*', '', text)
            old = len(text.split())
            print(f'The previous corpus had {format(old), ","} tokens and the new one has')
            new = len(metam.split())
            print(f' {format(new, ",")}; this is {format(old - new, ",")} less.')
            print(f'The previous corpus had {format(old), ","} tokens and the new one has'+
                  f' {format(new, ",")}; this is {format(old - new, ",")} less.')
            corpus_supplementum = ''.join([const.repo_path, corpus_sine_signis, '.txt'])
            print(f'Writing the new, reduced corpus to \'{corpus_supplementum}\' ...')
            with open(corpus_supplementum, 'w') as f:
                f.write(metam + ' ')
        return corpus_supplementum
    else:
        raise TypeError('All three "corpus", "signum", and "corpus_sine_signis" must be strings. Don\'t include the file extension!')

def remove_tokens(corpus: str, signa: list, corpus_sine_signis: str) -> str:
    """Remove several "unnecessary" tokens from the corpus.

    Parameters:
    corpus (str): name of a txt-file in the repository
    signa (list): a list of tokens whose all occrrences are to be removed from the 'corpus'
    cropus_sine_signis (string) name of the txt-file to which the result of the removal are saved; includes the path 'const.repo_path'

    Returns:
    str: the name of the txt-file, including the path 'const.repo_path', that contains the result of the removal
    """
    with open(''.join([const.repo_path, 'CORPUS_MENSURA.txt']), 'r') as f:
        cpm = int(f.read())
    if type(corpus) == type(signa) == type(corpus_sine_signis) == str:
        vetus_corpus = ''.join([const.repo_path, corpus, '.txt'])
        with open(vetus_corpus, 'r') as f:
            text = f.read()
            metam = text
            for signum in signa:
                metam = pkg.re.sub(r'\S*'+signum+r'\S*', '', metam)
            diff = int(cpm) - len(metam.split())
            print(f'The previous corpus had {format(len(text.split()), ",")} tokens and the new one has'+
                  f' {format(len(metam.split()), ",")}; this is {format(diff, ",")} less.')
            corpus_supplementum = ''.join([const.repo_path, corpus_sine_signis, '.txt'])
            print(f'Writing the new, reduced corpus to \'{corpus_supplementum}\' ...')
            with open(corpus_supplementum, 'w') as f:
                f.write(metam + ' ')
        return corpus_supplementum
    else:
        raise TypeError('All three "corpus", "signum", and "corpus_sine_signis" must be strings. Don\'t include the file extension!')
    
# count all characters in the corpus that match the regex and write them to a file, 
# then print the number of characters and the ratio to the whole corpus
# return true if the file was created, false if not
def count_regex(corpus: str, regex: str, misfits: str) -> bool:
    """Count all characters in the corpus that match the regex and write them to a file.
    
    Then print the number of characters and the ratio to the whole corpus.

    Parameters:
    corpus (str): name of a txt-file in the repository
    regex (str): a regular expression

    Returns:
    bool: True if the file was created, False if not
    """
    if type(corpus) == type(regex) == type(misfits) == str:
        with open(''.join([const.repo_path, corpus, '.txt']), 'r') as f:
            text = f.read()
            match_temp = pkg.re.findall(regex, text)
            matches = pkg.re.sub(' +', ' ', ''.join(match_temp))
            with open(''.join([const.repo_path, misfits, '.txt']), 'w') as f:
                f.write(''.join(matches))
            print(f'The corpus contains {format(len(matches), ",")} characters that match the regex \'{regex}\'.')
            print(f'This is {format(len(matches)/len(text), ".2%")} of the whole corpus.')
            return True
    else:
        raise TypeError('Both "corpus", "regex", and misfits must be strings. Don\'t include the file extension!')

def get_corpus_chunk(corpus: str, start: int, end: int) -> str:
    """Find the chunk of the txt-file limited to two indices.

    It returns the string that starts with the character corpus[start] and ends with the character corpus[end].

    Parameters:
    corpus (str): name of a txt-file in the repository
    start (int): the index of the first character of the chunk
    end (int): the index of the last character of the chunk

    Returns:
    str: the chunk of the txt-file limited to two indices
    """
    if type(corpus) == str:
        with open(''.join([const.repo_path, corpus, '.txt']), 'r') as f:
            text = f.read()
            if type(start) == type(end) == int:
                if start < end and end-start <= 5000 and start > 0 and end > 0 and end < len(text):
                    return text[start:end]
            else:
                raise TypeError('Both "start" and "end" must be integers.')
            return "Something else is wrong with the indices!"
    else:
        raise TypeError('The "corpus" is a txt file name and should be a string. Don\'t include the file extension!')
    
#write a function that takes replaces all the keys from the dictionary with their values in the a corpus
def key_value_exchange(corpus: str, dictionary: dict, clavis_valorem_commutationem: str) -> str:
    """Replace all the keys from the dictionary with their values in the corpus.

    Parameters:
    corpus (str): name of a txt-file in the repository
    dictionary (dict): a dictionary whose keys are to be replaced with their values in the corpus
    clavis_valorem_commutationem (string) name of the txt-file to which the result of the replacement are saved; includes the path 'const.repo_path'

    Returns:
    str: the name of the txt-file, including the path 'const.repo_path', that contains the result of the replacement
    """
    if type(corpus) == str and type(dictionary) == dict and type(clavis_valorem_commutationem) == str:
        vetus_corpus = ''.join([const.repo_path, corpus, '.txt'])
        with open(vetus_corpus, 'r') as f:
            text = f.read()
            metam = text
            for key in dictionary.keys():
                metam = metam.replace(key, dictionary[key])
            corpus_supplementum = ''.join([const.repo_path, clavis_valorem_commutationem, '.txt'])
            print(f'Writing the new, reduced corpus to \'{corpus_supplementum}\' ...')
            with open(corpus_supplementum, 'w') as f:
                f.write(metam + ' ')
        return corpus_supplementum
    else:
        raise TypeError('All three "corpus", "dictionary", and "clavis_valorem_commutationem" must be strings. Don\'t include the file extension!')

def strip_file_name(filename: str) -> str:
    """Strip the "filename" of all its extra tokens, such as the path to the file, the file extension, and so on.

    Is should be used if function remove_token() is to be used in several steps.

    Parameters:
    filename (str): the complete name of a txt-file including const.repo_path

    Returns:
    str: name of the txt-file, saved in const.repopath, without the path to the file and the file extension
    """
    if type(filename) == str:
        if filename.endswith('.txt'):
            return filename.replace(const.repo_path, "")[:-4]
        else:
            raise TypeError('The filename should be a file name that ends with ".txt"!')
    else:
        raise TypeError('The "filename" is a file name and should be a string!')
    
def print_variable_sizes(raw: pkg.Union[str, list, pkg.pd.Series, pkg.pd.DataFrame]) -> None:
    """Format the size of a file in a human-readable way.

    Parameters:
    variable (Union[list, str, int, float]): the variable whose size is to be printed

    Returns:
    None: it prints the size of the variable in the human-readable way
    """
    size = 0
    if (type(raw) == list) or (type(raw) == str):
        size = pkg.sys.getsizeof(raw)
    elif type(raw) == pkg.pd.Series:
        size = raw.memory_usage(deep=True)
    elif type(raw) == pkg.pd.DataFrame:
        size = raw.memory_usage(deep=True).sum()
    else:
        raise TypeError('The "raw" is not a list, string, or a pandas Series or DataFrame.')
    get_human_readable_size(size)

def total_occupied_space(locals: dict, keys_to_remove: list) -> str:
    """Calculate the total space occupied by all variables in the locals dictionary.

    Parameters:
    locals (dict): a dictionary of all local variables
    keys_to_remove (list): a list of all variables that are to be removed

    Returns:
    int: the total space occupied by all variables in the locals dictionary
    """
    total = 0
    for key in locals.keys():
        if key in keys_to_remove:
            total += pkg.sys.getsizeof(locals[key])
    return get_human_readable_size(total)

def get_human_readable_size(size: int) -> str:
    """format the size of a file in a human-readable way.

    Parameters:
    size (int): the size of the variable in bytes

    Returns:
    None: it prints the size of the variable in the human-readable way
    """
    if size < 1024:
        return f'{size} bytes'
    elif size < 1048576:
        return f'{size/1024:.2f} KB'
    elif size < 1073741824:
        return f'{size/1048576:.2f} MB'
    else:
        return f'is {size/1073741824:.2f} GB'

def cleanup_variable_space(locals: dict, key_copies: list) -> None:
    """A function to delete all files that are not necessary any more to save space.

    Parameters:
    locals (dict): a dictionary of all local variables

    Returns:
    None: it deletes all local variables that are not necessary any more
    """
    if(key_copies == []):
        print('No variables to delete')
        return
    else:
        print('Following variables are begin deleted:')
        print(key_copies)
    
    for key in key_copies:
        del locals[key]
        pkg.gc.collect()
    
def my_funct():
    for name in globals():
        print(id(globals()[name]))

    # Define some local variables
    x = 1
    y = 2
    z = 3

    # Get the local variables
    local_vars = locals()

    # Print the local variable names and values
    for var_name in local_vars:
        print(f"{var_name} = {local_vars[var_name]}")
