"""Offline replay of arcint's prefix cache over pi session trees.
Usage: python3 tools/cachesim.py <dir of pi session .jsonl files>
The transcripts are private operator data and are NOT in this repository.
Policy as read from src/core/prefix_cache.cpp + backend_ov.cpp (2026-08-30):
one snapshot per request at floor((len-1)/grid)*grid; lookup = deepest entry
whose tokens are a prefix of the prompt; LRU on hit or insert; byte budget
counts GDN rows only (=> max entries); KV pages by refcount with LRU eviction
under pool pressure. Tokens per message are estimated from characters with a
per-session least-squares fit against the recorded usage.input."""
import json, os, sys, collections, statistics
CFG = {"coder": dict(grid=128,  entries_max=21, pool_tok=8357*16,  n_ctx=98304),
       "agent": dict(grid=2048, entries_max=85, pool_tok=23593*16, n_ctx=262144)}
def text_of(m):
    c = m.get("content"); out=[]
    if isinstance(c,str): return c
    for p in c or []:
        if not isinstance(p,dict): continue
        t=p.get("type")
        if t=="text": out.append(p.get("text",""))
        elif t=="toolCall" or t=="tool_call": out.append(json.dumps(p.get("arguments") or p.get("input") or p.get("args") or "")); out.append(p.get("name","") or p.get("toolName",""))
        elif t=="toolResult": out.append(str(p.get("content","")))
    return "\n".join(out)
sessions=[]
COMPACT=set()
ROOT = sys.argv[1] if len(sys.argv) > 1 else "pi"
for d,_,fs in os.walk(ROOT):
    for f in fs:
        if not f.endswith(".jsonl"): continue
        ev=[]
        for ln in open(os.path.join(d,f), errors="replace"):
            try: ev.append(json.loads(ln))
            except Exception: pass
        sessions.append((os.path.join(d,f), ev))
# Build per-session turns: (timestamp, path_msg_ids, path_chars_per_msg, usage_input)
turns=[]  # global list
CAL=[]
PENDING=[]
for path, ev in sessions:
    by_id={e["id"]:e for e in ev if "id" in e}
    chars={}; comp_summary={}
    for e in ev:
        if e.get("type")=="message": chars[e["id"]]=len(text_of(e["message"]))
        if e.get("type")=="compaction": chars[e["id"]]=len(e.get("summary","") or ""); COMPACT.add(e["id"])
    sess_turns=[]
    for e in ev:
        if e.get("type")!="message" or e["message"].get("role")!="assistant": continue
        # walk up to root collecting message-bearing events; a compaction cuts history: keep summary + entries after firstKeptEntryId
        chain=[]; cur=by_id.get(e.get("parentId")); comp=None
        while cur is not None:
            if cur.get("type")=="message": chain.append(cur["id"])
            elif cur.get("type")=="compaction": comp=cur; chain.append(cur["id"]); break
            cur=by_id.get(cur.get("parentId"))
        chain.reverse()
        usage=(e["message"].get("usage") or {}).get("input",0) or 0
        ts=e["message"].get("timestamp") or 0
        sess_turns.append((int(ts) if str(ts).isdigit() else 0, chain, usage, path))
    # calibrate chars->tokens per session: input = S + k*chars  (least squares), fall back to k=1/3.6, S=1500
    xs=[sum(chars.get(i,0) for i in ch) for _,ch,u,_ in sess_turns]; ys=[u for _,_,u,_ in sess_turns]
    pts=[(x,y) for x,y in zip(xs,ys) if y>0]
    k,S=1/3.6,1500.0
    if len(pts)>=3:
        n=len(pts); mx=sum(x for x,_ in pts)/n; my=sum(y for _,y in pts)/n
        vx=sum((x-mx)**2 for x,_ in pts)
        if vx>0:
            k2=sum((x-mx)*(y-my) for x,y in pts)/vx; S2=my-k2*mx
            if 0.1<k2<1.0 and 0<=S2<60000: k,S=k2,S2
    ok = not (abs(k-1/3.6)<1e-9 and S==1500.0)
    CAL.append((k,S,len(pts)))
    PENDING.append((sess_turns, chars, k if ok else None, S if ok else None))
fk=[c[0] for c in CAL if not (abs(c[0]-1/3.6)<1e-9 and c[1]==1500.0)]; fS=[c[1] for c in CAL if not (abs(c[0]-1/3.6)<1e-9 and c[1]==1500.0)]
K0=statistics.median(fk) if fk else 1/3.6; S0=statistics.median(fS) if fS else 1500.0
for sess_turns, chars, k, S in PENDING:
    k = k if k is not None else K0; S = S if S is not None else S0
    for ts,ch,u,p in sess_turns:
        toks=[max(1,int(round(chars.get(i,0)*k))) for i in ch]
        turns.append((ts, p, ch, toks, int(S), u))
turns.sort(key=lambda t:t[0])
fitted=[c for c in CAL if c[2]>=3 and not (abs(c[0]-1/3.6)<1e-9 and c[1]==1500.0)]
print("turns %d, sessions %d | calibration: %d sessions fitted (median chars/token %.2f, median system tokens %d), %d fell back" % (len(turns), len(sessions), len(fitted), statistics.median(1/c[0] for c in fitted) if fitted else 0, int(statistics.median(c[1] for c in fitted)) if fitted else 0, len(CAL)-len(fitted)))

def simulate(name, grid, entries_max, pool_tok, n_ctx, frontier_only=False):
    entries=[]  # LRU list, front = most recent: dict(len, path(tuple of msg ids incl. 'sys'), bounds(cum tokens))
    st=collections.Counter(); prefilled=[]; hits_tok=0; prompt_tok=0; classes=collections.Counter(); class_tok=collections.Counter()
    last_len_by_session={}; lost_reason={}  # session -> why its frontier entry left
    def union_tokens(es, extra=None):
        seen=set(); tot=0
        for e in es + ([extra] if extra else []):
            for mid,tk in zip(e["path"], e["toklens"]):
                if mid not in seen: seen.add(mid); tot+=tk
        return tot
    for ts, sess, chain, toks, S, usage in turns:
        path=("sys:%s"%sess,)+tuple(chain); toklens=[S]+toks
        L=sum(toklens)
        if L>n_ctx: st["unservable"]+=1; continue
        st["requests"]+=1; prompt_tok+=L
        # lookup: deepest entry whose snapshot lies within the common prefix
        best=None; best_len=0
        for e in entries:
            common=0
            for a,b,tk in zip(path, e["path"], toklens):
                if a!=b: break
                common+=tk
            if e["len"]<=common and e["len"]>best_len: best,best_len=e,e["len"]
        if best is not None:
            entries.remove(best); entries.insert(0,best); st["hits"]+=1; hits_tok+=best_len
        pre=L-best_len; prefilled.append(pre)
        prev=last_len_by_session.get(sess)
        if best_len==0:
            cls="front" if prev else "first"
        elif prev and best_len < (prev-1)//grid*grid: cls="mid"
        else: cls="append"
        classes[cls]+=1; class_tok[cls]+=pre
        if cls=="front":
            has_entry = any(e["path"][0]==path[0] for e in entries)
            why = "compaction" if (chain and chain[0] in COMPACT) else ("front-changed" if has_entry else lost_reason.get(sess,"evicted-unknown"))
            st["front:"+why]+=1; st["fronttok:"+why]+=pre
        if best is not None: lost_reason.pop(sess,None)
        last_len_by_session[sess]=L
        # snapshot at the last grid multiple below L
        snap=(L-1)//grid*grid
        if snap>0:
            # live request needs its own pages (L tokens, prefix shared); evict LRU under pool pressure
            new={"len":snap,"path":path,"toklens":toklens}
            if frontier_only:
                # drop ancestors of this lineage: same session, path is a prefix of ours (a superseded snapshot)
                keep=[]
                for e in entries:
                    anc = e["path"][0]==path[0] and len(e["path"])<=len(path) and tuple(path[:len(e["path"])])==e["path"] and e["len"]<=snap
                    if anc: st["superseded"]+=1
                    else: keep.append(e)
                entries=keep
            while entries and union_tokens(entries,new) > pool_tok:
                v=entries.pop(); st["evict_pool"]+=1; lost_reason[v["path"][0][4:]]="pool"
            entries.insert(0,new)
            while len(entries)>entries_max:
                v=entries.pop(); st["evict_budget"]+=1; lost_reason.setdefault(v["path"][0][4:],"budget")
        st["max_entries"]=max(st["max_entries"],len(entries))
    n=st["requests"]; prefilled.sort()
    q=lambda p: prefilled[min(len(prefilled)-1,int(p*len(prefilled)))] if prefilled else 0
    print("\n=== %s: grid %d, %d entries, pool %d tok, n_ctx %d ===" % (name,grid,entries_max,pool_tok,n_ctx))
    print("  requests %d (unservable > n_ctx: %d) | hits %d = %.1f%% of requests | tokens from cache %.1f%% of prompt tokens (%d of %d)" % (n, st["unservable"], st["hits"], 100*st["hits"]/max(n,1), 100*hits_tok/max(prompt_tok,1), hits_tok, prompt_tok))
    print("  prefilled per request: p50 %d  p90 %d  p99 %d  max %d  | total prefilled %d" % (q(.5),q(.9),q(.99),prefilled[-1] if prefilled else 0, sum(prefilled)))
    print("  miss classes (requests / prefilled tokens): " + ", ".join("%s %d / %d" % (c, classes[c], class_tok[c]) for c in ("first","front","mid","append")))
    print("  evictions: pool-pressure %d, entry-budget %d, superseded-ancestors %d | max live entries %d" % (st["evict_pool"], st["evict_budget"], st["superseded"], st["max_entries"]))
    print("  front misses by cause (requests / tokens): " + ", ".join("%s %d / %d" % (w, st["front:"+w], st["fronttok:"+w]) for w in ("compaction","front-changed","pool","budget","evicted-unknown")))
    grid_loss=sum(min(p, grid) for p,c in zip(prefilled,[None]*len(prefilled)))  # placeholder
    return classes, class_tok
for name,c in CFG.items(): simulate(name, **c)
print("\n##### levers, agent configuration #####")
for g in (256, 128):
    c=dict(CFG["agent"]); c["grid"]=g; simulate("agent, snapshot grid %d" % g, **c)
c=dict(CFG["agent"]); simulate("agent, frontier-only entries", frontier_only=True, **c)
c=dict(CFG["agent"]); c["grid"]=128; simulate("agent, grid 128 + frontier-only", frontier_only=True, **c)
c=dict(CFG["coder"]); simulate("coder, frontier-only entries", frontier_only=True, **c)
