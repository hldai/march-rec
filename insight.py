import re


f = open('d:/projects/python/march-rec/log/tmp.txt', encoding='utf-8')
fout = open('d:/projects/python/march-rec/log/tmp-result.txt', 'w', encoding='utf-8', newline='\n')
line = next(f)

try:
    while True:
        fout.write(line)
        next(f)
        next(f)
        min_err = 1e10
        min_line = None
        while True:
            line = next(f)
            if 'marchrec.py:89' in line:
                break
            else:
                m = re.search('err_val_pr=(.*?),', line)
                # if m.group(1) == 'nan':
                #     continue
                ev = float(m.group(1))
                if ev < min_err:
                    min_err = ev
                    min_line = line
        if min_line:
            fout.write(min_line)
except StopIteration:
    pass
f.close()
fout.close()