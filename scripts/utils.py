def read_file(fname, n=None):
    data = []
    i = 0
    with open(fname) as f:
        for line in f:
            data.append(line.strip())
            i+=1
            if n is not None and i > n:
                break
    return data
    
def get_data( tgt_lang, domain, split, data_dir="internal_split"):
    source = read_file(f"{data_dir}/en-{tgt_lang}/{split}.{domain}.en")
    formal_translations = read_file(f"{data_dir}/en-{tgt_lang}/{split}.{domain}.formal.{tgt_lang}")
    informal_translations = read_file(f"{data_dir}/en-{tgt_lang}/{split}.{domain}.informal.{tgt_lang}")
    return source, formal_translations, informal_translations