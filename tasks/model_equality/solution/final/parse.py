import re, json, sys
P='/Users/matisgeran/Pro/Hackathon/European_AI_Championship_Paris/challenge/task2_questions_reponses.md'
L=open(P,encoding='utf-8').read().split('\n')
fam={}
for l in L[:70]:
    m=re.match(r'\| ep-(\d\d) \| (F\d) \| (.*?) \| (.*?) \|',l)
    if m: fam['ep-'+m.group(1)]=(m.group(2),m.group(3),m.group(4))
i3=next(i for i,l in enumerate(L) if l.startswith('## 3.'))
# quiz
quiz={}  # (qid,item) -> {ep: [answers]}
qid=None; item=None
for l in L[:i3]:
    m=re.match(r'<summary><b>(q\d+)</b>',l)
    if m: qid=m.group(1); continue
    m=re.match(r'\*\*(\d+)\. (.*)\*\*',l)
    if m and qid: item=int(m.group(1)); quiz[(qid,item)]={'q':m.group(2),'ans':{}}; continue
    m=re.match(r'- `(.*?)` \((\d+)\) — (.*)$',l)
    if m and qid and item:
        a=m.group(1); n=int(m.group(2))
        eps=[]
        for part in m.group(3).split('·'):
            part=part.strip(); mm=re.match(r'F\d : (.*)',part)
            if mm: eps+=['ep-'+x for x in mm.group(1).split()]
        for e in eps: quiz[(qid,item)]['ans'].setdefault(e,[]).append((a,n,len(eps)))
# free
free={} # pid -> {ep: text}
pid=None; ep=None; buf=[]; meta={}
def flush():
    global buf
    if pid and ep: free[pid][ep]='\n'.join(buf).rstrip()
    buf=[]
prompt_buf=None
for l in L[i3:]:
    m=re.match(r'### (\w+) · (.*?) · (\d+) jetons',l)
    if m: flush(); pid=m.group(1); ep=None; free[pid]={}; meta[pid]={'kind':m.group(2),'max':int(m.group(3)),'prompt':[]}; continue
    m=re.match(r'\*\*(ep-\d\d)\*\*',l)
    if m: flush(); ep=m.group(1); continue
    if l.startswith('<') : 
        if ep: flush(); ep=None
        continue
    if l.startswith('>'):
        t=l[1:]; t=t[1:] if t.startswith(' ') else t
        t=re.sub(r'  $','',t)
        if ep: buf.append(t)
        elif pid: meta[pid]['prompt'].append(t)
flush()
json.dump({'fam':fam,'quiz':{f"{k[0]}:{k[1]}":v for k,v in quiz.items()},'free':free,'meta':meta},open('ep.json','w'),ensure_ascii=False)
print(len(fam),len(quiz),len(free),{p:len(v) for p,v in list(free.items())[:5]})
for p in list(meta)[:30]: print(p,meta[p]['kind'],meta[p]['max'],'|',' '.join(meta[p]['prompt'])[:110])
