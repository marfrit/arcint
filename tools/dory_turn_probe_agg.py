# dory_turn_probe_agg.py: aggregates the output of tools/dory_turn_probe.cpp (a green run).
#
#   python3 -I tools/dory_turn_probe_agg.py probe.out [earlier-probe.out] [-t]     (-t may stand anywhere)
#
# Prints the set-up lines, the probe count, the IDENT totals per loop count (bit-equal rows, maxKL, rates), the TVT totals,
# and the PAIR table: per count transition x->y the rows, the mixed run's mean KL and argmax agreement, the floor's (the
# same count, batched against step by step), the pure-x arm's, dKL and dAgree (mixed minus floor, agreement in points),
# the worst single conversation's agreement change (name and points) and the number of conversations whose greedy text
# equals the load at y. Then the CHAIN lines. With a second file (the output of an earlier probe run against the unpatched
# library, not in the repository): how many EXACT texts equal that run's. With -t:
# the answer tails of the exact and mixed texts per pair. It is the table of the 0118 section of contrib/llama.cpp/README.md.
# Not part of the arcint build.
import re,sys,collections
FILES=[a for a in sys.argv[1:] if a!='-t']
fl={}; pairs=collections.defaultdict(list); ex={}; mixed={}; ident=collections.defaultdict(list); tvt=collections.defaultdict(list)
probe=[]; chain=[]; ctext=[]; pre=[]; misc=[]
for ln in open(FILES[0],errors='replace'):
    ln=ln.rstrip('\n')
    m=re.match(r'FLOOR loops=(\d+) case=(\w+) meanKL=(\S+) maxKL=(\S+) agree=(\d+)/(\d+)',ln)
    if m: fl[(m[1],m[2])]=(float(m[3]),float(m[4]),int(m[5]),int(m[6])); continue
    m=re.match(r'PAIR (\d+)->(\d+) case=(\w+) n=(\d+) mixed: meanKL=(\S+) maxKL=(\S+) agree=(\d+)/(\d+) \| pure\d+: meanKL=(\S+) maxKL=(\S+) agree=(\d+)/(\d+) \| same_text=(\d)',ln)
    if m: pairs[(m[1],m[2])].append((m[3],int(m[4]),float(m[5]),float(m[6]),int(m[7]),float(m[9]),float(m[10]),int(m[11]),int(m[13]))); continue
    m=re.match(r'EXACT loops=(\d+) case=(\w+) n=(\d+) (\S+) t/s text="(.*)"$',ln)
    if m: ex[(m[1],m[2])]=(m[5],float(m[4]),int(m[3])); continue
    m=re.match(r'MIXED (\d+)->(\d+) case=(\w+) n=(\d+) text="(.*)"$',ln)
    if m: mixed[(m[1],m[2],m[3])]=m[5]; continue
    m=re.match(r'IDENT loops=(\d+) case=(\w+) same_tokens=(\d) rows=(\d+) bits_equal=(\d+) meanKL=(\S+) maxKL=(\S+) maxabs=(\S+) agree=(\d+)/(\d+) \| (\S+) t/s \(baseline (\S+)\)',ln)
    if m: ident[m[1]].append((m[2],int(m[3]),int(m[4]),int(m[5]),float(m[7]),float(m[11]),float(m[12]))); continue
    m=re.match(r'TVT (\d+)->(\d+) case=(\w+) .*rows=(\d+) bits_equal=(\d+) meanKL=(\S+) maxabs=(\S+)',ln)
    if m: tvt[(m[1],m[2])].append((int(m[4]),int(m[5]),float(m[7]))); continue
    if ln.startswith('PROBE'): probe.append(ln); continue
    if ln.startswith('CHAIN '): chain.append(ln); continue
    if ln.startswith('CHAINTEXT') or ln.startswith('CHAIN-DONE'): ctext.append(ln); continue
    if ln.startswith('PREFILL'): pre.append(ln); continue
    if re.match(r'(API|LOAD|TOK|MODE|DTURN|SETLOOPS|TRANSFER)',ln): misc.append(ln)
for l in misc+pre: print(l)
print(f"\nPROBES: {sum(1 for p in probe if p.endswith('| OK'))} OK of {sum(1 for p in probe if p.startswith('PROBE ') and '->' in p)}; " + ' '.join(p for p in probe if 'SUMMARY' in p))
print("\nIDENT (one context, count set per turn, vs a stock-style load at that count; greedy answers + logits)")
for y,rows in sorted(ident.items()):
    R=sum(r[2] for r in rows); B=sum(r[3] for r in rows)
    print(f"  loops={y}: cases {len(rows)} same_tokens {sum(r[1] for r in rows)}/{len(rows)} rows {R} bits_equal {B} maxKL {max(r[4] for r in rows):g} | t/s turn {sum(r[5] for r in rows)/len(rows):.1f} baseline {sum(r[6] for r in rows)/len(rows):.1f}")
print("\nTVT (turn switch in one context vs state transfer between two loads, the state-transfer arm)")
for k,rows in sorted(tvt.items()):
    print(f"  {k[0]}->{k[1]}: rows {sum(r[0] for r in rows)} bits_equal {sum(r[1] for r in rows)} maxabs {max(r[2] for r in rows):g}")
print("\npair   n   | mixed meanKL agree%  | floor(Y) meanKL agree% | pure(X) meanKL agree% | dKL(mixed-floor) dAgree(pt) | worst case dAgree")
for (x,y),rows in sorted(pairs.items()):
    N=sum(r[1] for r in rows)
    mk=sum(r[2]*r[1] for r in rows)/N; ma=100*sum(r[4] for r in rows)/N
    pk=sum(r[5]*r[1] for r in rows)/N; pa=100*sum(r[7] for r in rows)/N
    fn=sum(fl[(y,r[0])][3] for r in rows); fk=sum(fl[(y,r[0])][0]*fl[(y,r[0])][3] for r in rows)/fn; fa=100*sum(fl[(y,r[0])][2] for r in rows)/fn
    worst=min(((100*r[4]/r[1]-100*fl[(y,r[0])][2]/fl[(y,r[0])][3]),r[0]) for r in rows)
    print(f"{x}->{y} {N:4d} | {mk:.4f} {ma:5.1f} | {fk:.4f} {fa:5.1f} | {pk:.4f} {pa:5.1f} | {mk-fk:+.4f} {ma-fa:+.1f} | {worst[1]} {worst[0]:+.1f} | cases {len(rows)} maxKL {max(r[3] for r in rows):.3f} same_text {sum(r[8] for r in rows)}")
print("\nCHAIN"); 
for l in chain+ctext: print("  "+l[:420])
def tail(t):
    i=t.find('</think>'); return (t[i+8:] if i>=0 else '[no </think> in limit] ...'+t[-90:])[:230]
if len(FILES)>1:
    old={}
    for ln in open(FILES[1],errors='replace'):
        m=re.match(r'EXACT loops=(\d+) case=(\w+) n=(\d+) \S+ t/s text="(.*)"$',ln.rstrip('\n'))
        if m: old[(m[1],m[2])]=m[4]
    same=sum(1 for k in ex if k in old and old[k]==ex[k][0]); print(f"\nEXACT texts equal to the earlier run of the unpatched library: {same} of {len([k for k in ex if k in old])} (patched baseline loads vs unpatched libllama)")
    for k in sorted(ex):
        if k in old and old[k]!=ex[k][0]: print("  differs:",k)
if '-t' in sys.argv:
    for k in sorted(mixed):
        x,y,c=k
        print(f"\n{x}->{y} {c}\n  exact{y}: {tail(ex[(y,c)][0])}\n  mixed : {tail(mixed[k])}")
