import json,re,sys
src=open('cells.py').read()
parts=re.split(r'^# %%(.*)$',src,flags=re.M)
cells=[]
for i in range(1,len(parts),2):
    kind=parts[i].strip(); body=parts[i+1].strip('\n')
    if kind=='[md]':
        body='\n'.join(l[2:] if l.startswith('# ') else l.lstrip('#') for l in body.split('\n'))
        cells.append({'cell_type':'markdown','metadata':{},'source':body})
    else:
        cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':body})
nb={'cells':cells,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3'},'colab':{'provenance':[]}},'nbformat':4,'nbformat_minor':5}
json.dump(nb,open(sys.argv[1],'w'),ensure_ascii=False,indent=1)
if len(sys.argv)>2:  # local test script
    R=sys.argv[2]; code=[]
    for c in cells:
        if c['cell_type']!='code': continue
        s=c['source']
        if 'drive.mount' in s:
            s=f"import os\nRESULTS={R!r}\nOUT='{sys.argv[3]}'\nos.makedirs(OUT,exist_ok=True)"
        if s.startswith('!cd'): continue
        s=s.replace('plt.show()','plt.close("all")')
        if s.rstrip().endswith('\ntab'): s+='\nprint(tab.to_string())'
        code.append(s)
    open('test_run.py','w').write('\n\n'.join(code))
