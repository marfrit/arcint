# The depth mix's task battery, a measurement body (a bash fragment; no builds). Four tasks at
# temperature 0, each answer saved in full and judged by a script against its known answer, in four arms:
# the chain (off) and the mix (fold) at the unit's q8_0 KV by default; the FA port arms at f16. Texts need not be
# identical across arms; the first differing character against the off arm is printed.
# Uses: $B (must hold llama-completion), $G, $O, log MSG, run NAME CMD..., chk DESC CMD...
# Optional: $ARMS (default "off mix"; "port" and "both" need KV=f16), $KV (q8_0, the unit's; or f16), $NGL (99), $NREC (JSON records, 100 ~ 4k tokens),
#   $NFUN (code samples before the binary search, 139 = the full-run prompt), $TMO seconds a generation (900),
#   $TASKS (default all four). The model reasons in its answer first: the capital gets 160 tokens for that.
ARMS=${ARMS:-off mix}; KV=${KV:-q8_0}; TASKS=${TASKS:-capital arith json bsearch}; NGL=${NGL:-99}; NREC=${NREC:-100}; NFUN=${NFUN:-139}; TMO=${TMO:-900}
DEPLOY="GGML_OPENCL_CPY_TRANSPOSE=1 GGML_OPENCL_KQ_MM_ID_I8=1 GGML_OPENCL_KQ_MM_ID_I8_WIDE=1"
chk "llama-completion is built" [ -x "$B/llama-completion" ]
KVF=""; [ "$KV" = q8_0 ] && KVF="-ctk q8_0 -ctv q8_0"
case " $ARMS " in *" port "*|*" both "*) chk "the port arms need KV=f16" [ "$KV" = f16 ];; esac
log "battery: arms [$ARMS], KV $KV, tasks [$TASKS], -ngl $NGL"
cat > "$O/battery.py" <<'PY'
import json, random, re, sys
def gen(out, nrec, nfun):
    open(out + "/p-capital.txt", "w").write("What is the capital of France? Answer with one word.")
    open(out + "/p-arith.txt", "w").write(
        "Start with 17. Multiply by 6. Subtract 28. Add 135. Divide by 11. Multiply by 7. Subtract 45.\n"
        "Work through the steps, then give the result on the last line in the form ANSWER: <integer>.")
    first = ["Jonas", "Marit", "Pavel", "Ines", "Tariq", "Sofia", "Emeka", "Yuki", "Anton", "Leila", "Oskar", "Nadia"]
    last = ["Weber", "Oyelaran", "Novak", "Castro", "Haddad", "Lindqvist", "Okafor", "Tanaka", "Petrov", "Rahimi", "Berg", "Moreau"]
    city = ["Graz", "Leeuwarden", "Brno", "Porto", "Aarhus", "Tartu", "Gdansk", "Lyon", "Turku", "Bergen", "Ghent", "Basel"]
    r = random.Random(4201); recs = []
    for i in range(nrec):
        oid = "QX-%04d" % (1000 + 37*i)
        rec = dict(id=oid, customer=r.choice(first) + " " + r.choice(last), city=r.choice(city), quantity=r.randint(2, 90),
                   item="H-%03d" % r.randint(100, 999), total_eur=r.randint(120, 4800), day=r.randint(1, 28))
        if i == (nrec*3)//5:
            rec.update(id="QX-7731", customer="Marit Oyelaran", city="Leeuwarden", quantity=37, total_eur=1924)
        recs.append(rec)
    body = "".join("Order %(id)s was placed by %(customer)s from %(city)s for %(quantity)d units of item %(item)s, "
                   "total %(total_eur)d EUR, shipped on 2026-03-%(day)02d.\n" % x for x in recs)
    open(out + "/p-json.txt", "w").write("Here is an order log.\n\n" + body + "\nFrom the log, extract order QX-7731. "
        "Reply with only a JSON object with the keys customer, city, quantity, total_eur.")
    src = "".join("def lookup_%d(index: int) -> int:\n    return (index * %d + %d) %% 65521\n\n" % (i, i % 71 + 1, i % 177 + 3) for i in range(nfun))
    cut = "def lookup_139(index: int) -> int:\n    return (index * 69 +" if nfun == 139 else ""
    open(out + "/p-bsearch.txt", "w", newline="").write("# Audit run short-4096\n# Implementation samples follow.\n" + src + cut +
        "\n# End of samples. Write a Python implementation and tests for a binary search function.\ndef binary_search(values, target):\n")
def judge(task, text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if task == "capital":
        return "paris" in text.lower(), "expects Paris"
    if task == "arith":
        m = re.findall(r"ANSWER:\s*\**\s*(-?\d+)", text) or re.findall(r"-?\d+", text)
        return bool(m) and int(m[-1]) == 88, "expects 88, got %s" % (m[-1] if m else "no integer")
    if task == "json":
        want = dict(customer="Marit Oyelaran", city="Leeuwarden", quantity=37, total_eur=1924)
        for m in re.finditer(r"\{[^{}]*\}", text):
            try: got = json.loads(m.group(0))
            except Exception: continue
            ok = all(str(got.get(k, "")).strip().lower() == str(v).lower() if isinstance(v, str) else
                     str(got.get(k, "")).replace(".0", "") == str(v) for k, v in want.items())
            return ok, "got %s" % json.dumps(got, sort_keys=True)
        return False, "no JSON object"
    if task == "bsearch":
        body = []
        for line in text.split("\n"):
            if line.strip() and not line.startswith((" ", "\t")): break
            body.append(line)
        ns = {}
        try: exec("def binary_search(values, target):\n" + "\n".join(body), ns)
        except Exception as e: return False, "does not compile: %r" % (e,)
        r = random.Random(7); f = ns["binary_search"]
        for case in range(300):
            v = sorted(r.sample(range(-500, 500), r.randint(0, 40))); t = r.randint(-520, 520)
            try: i = f(list(v), t)
            except Exception as e: return False, "raises %r on case %d" % (e, case)
            if (t in v and not (isinstance(i, int) and 0 <= i < len(v) and v[i] == t)) or (t not in v and i != -1):
                return False, "wrong on case %d: %r for %r in %r" % (case, i, t, v)
        return True, "300 random cases"
if sys.argv[1] == "gen":
    gen(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
elif sys.argv[1] == "judge":
    ok, why = judge(sys.argv[2], open(sys.argv[3], errors="replace").read()); print(("right" if ok else "wrong") + " (" + why + ")")
elif sys.argv[1] == "diff":
    a = open(sys.argv[2], errors="replace").read(); b = open(sys.argv[3], errors="replace").read()
    n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    print("identical" if a == b else "differs at char %d of %d / %d" % (min(len(a), len(b)) if n is None else n, len(a), len(b)))
PY
run bat-gen python3 "$O/battery.py" gen "$O" $NREC $NFUN
for t in $TASKS; do chk "prompt $t" [ -s "$O/p-$t.txt" ]; done
log "prompts: json $(wc -c < "$O/p-json.txt") bytes ($NREC records), bsearch $(wc -c < "$O/p-bsearch.txt") bytes ($NFUN samples)"
arm_env() { case $1 in off) echo "GGML_OPENCL_DEPTH_MIX=0";; mix) echo "GGML_OPENCL_DEPTH_MIX=1";;
            port) echo "GGML_OPENCL_DEPTH_MIX=0 GGML_OPENCL_FA_PORT=1";; both) echo "GGML_OPENCL_DEPTH_MIX=1 GGML_OPENCL_FA_PORT=1";; esac; }
for a in $ARMS; do for t in $TASKS; do n=bat-$a-$t
  case $t in capital) NT=160; M="-cnv -st --jinja";; arith) NT=400; M="-cnv -st --jinja";; json) NT=160; M="-cnv -st --jinja";; bsearch) NT=320; M="-no-cnv";; esac
  run $n sh -c "env -u GGML_OPENCL_FA_PORT $DEPLOY $(arm_env $a) timeout -s KILL $TMO '$B/llama-completion' -m '$G' -ngl $NGL -fa on $KVF -c 8192 -b 2048 -ub 512 --temp 0 -n $NT $M -f '$O/p-$t.txt' --no-display-prompt --no-warmup 2> '$O/$n.err' > '$O/ans-$a-$t.txt'"
  chk "$n: no abort, assert or backtrace" sh -c "! grep -qiE 'abort|backtrace|GGML_ASSERT|segmentation|clWaitForEvents|error -14|reaches [0-9]+ bytes' '$O/$n.err'"
  chk "$n: prompt evaluated" grep -qE 'prompt eval time' "$O/$n.err"
  chk "$n: an answer" [ -s "$O/ans-$a-$t.txt" ]
  log "$n: $(python3 "$O/battery.py" judge $t "$O/ans-$a-$t.txt") | prompt $(grep -oE 'prompt eval time.*/ +[0-9]+ tokens' "$O/$n.err" | grep -oE '[0-9]+ tokens' | tail -1) | vs off: $([ $a = off ] && echo - || python3 "$O/battery.py" diff "$O/ans-off-$t.txt" "$O/ans-$a-$t.txt")"
  log "$(printf '%s answer:\n%s\n----' "$n" "$(cat "$O/ans-$a-$t.txt")")"
done; done
for t in $TASKS; do
  log "battery $t: $(for a in $ARMS; do printf '%s %s; ' $a "$(python3 "$O/battery.py" judge $t "$O/ans-$a-$t.txt" | cut -d' ' -f1)"; done)"; done
