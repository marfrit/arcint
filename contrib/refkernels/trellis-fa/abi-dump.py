#!/usr/bin/env python3
# abi-dump.py — read a SPIR-V module's kernel entry point and say, in order, what
# arguments it takes: the count, the SPIR-V type of each, its size in bytes, and
# which field of trellis-serve's fa::Kern<D=128> functor it corresponds to. Also
# lists the capabilities/extensions the driver's SPIR-V reader must accept and the
# execution modes the compiler set.
#
#   python3 abi-dump.py <module.spvasm> <module.spv> <outdir>
#                       [--abi-name abi.ref.txt] [--names-name kernel-names.ref.txt]
#
# The .spvasm is the disassembly: spirv-dis output when SPIRV-Tools is present,
# llvm-spirv --to-text otherwise. Both are accepted here: this parser drops the
# word-count column and an optional "Op" prefix, so it reads either form.
import sys, hashlib, os

args = sys.argv[1:]
asm, spv, out = args[0], args[1], args[2]
abi_name, names_name = 'abi.txt', 'kernel-names.txt'
for i in range(3, len(args) - 1):
    if args[i] == '--abi-name':
        abi_name = args[i + 1]
    elif args[i] == '--names-name':
        names_name = args[i + 1]
lines = open(asm, encoding='utf-8', errors='replace').read().splitlines()


def op(line):
    t = line.split()
    if t and t[0].isdigit():
        t = t[1:]
    if t and t[0].startswith("Op"):
        t[0] = t[0][2:]
    return t


types = {}          # id -> (kind, detail, width_bytes or None)
func_ty = {}        # type id -> (ret type id, [param type ids])
funcs = {}          # func id -> function type id
fn_params = {}      # func id -> [(result type id, result id)]
entry = None
modes, caps, exts, names = [], [], [], {}
cur_fn = None

for line in lines:
    t = op(line)
    if not t:
        continue
    k = t[0]
    if k == 'Capability':
        caps.append(" ".join(t[1:]))
    elif k == 'Extension':
        exts.append(" ".join(t[1:]))
    elif k == 'TypeInt':
        types[t[1]] = ('int', t[2], int(t[2]) // 8)
    elif k == 'TypeFloat':
        types[t[1]] = ('float', t[2], int(t[2]) // 8)
    elif k == 'TypeBool':
        types[t[1]] = ('bool', '1', 1)
    elif k == 'TypeUntypedPointerKHR':
        types[t[1]] = ('ptr-untyped', t[2], None)
    elif k == 'TypePointer':
        types[t[1]] = ('ptr', t[2], None)
    elif k == 'TypeArray':
        types[t[1]] = ('array', (t[2], t[3]), None)
    elif k == 'TypeVector':
        types[t[1]] = ('vector', (t[2], t[3]), None)
    elif k == 'TypeStruct':
        types[t[1]] = ('struct', t[2:], None)
    elif k == 'EntryPoint':
        entry = (t[1], t[2], t[3], t[4:])
    elif k == 'ExecutionMode':
        modes.append((t[1], t[2], t[3:]))
    elif k == 'TypeFunction':
        func_ty[t[1]] = (t[2], t[3:])
    elif k == 'Function':
        cur_fn = t[2]
        funcs[t[2]] = t[4]
        fn_params[cur_fn] = []
    elif k == 'FunctionParameter':
        fn_params.setdefault(cur_fn, []).append((t[1], t[2]))
    elif k == 'Name':
        names[t[1]] = t[2]

mode_name = {
    '17': 'LocalSizeHint', '18': 'LocalSize', '22': 'VecTypeHint',
    '30': 'VecTypeHint(patch)', '31': 'ContractionOff', '35': 'SubgroupSize',
}


def size_of(tid, ptr_bytes, depth=0):
    if tid not in types or depth > 8:
        return ('?', None)
    kind, detail, w = types[tid]
    if kind in ('int', 'float'):
        return (kind, w)
    if kind == 'bool':
        return ('bool', 1)
    if kind in ('ptr', 'ptr-untyped'):
        return ('ptr', ptr_bytes)
    if kind == 'vector':
        el, n = detail
        ek, ew = size_of(el, ptr_bytes, depth + 1)
        return (f'{ek}x{n}', ew * int(n) if ew else None)
    if kind == 'array':
        el, n = detail
        ek, ew = size_of(el, ptr_bytes, depth + 1)
        return (f'{ek}[{n}]', ew * int(n) if ew else None)
    if kind == 'struct':
        tot, tys = 0, []
        for f in detail:
            ek, ew = size_of(f, ptr_bytes, depth + 1)
            tys.append(ek)
            tot += ew or 0
        return ('{' + ','.join(tys) + '}', tot)
    return (kind, w)


if not entry:
    print("ABI DUMP: no OpEntryPoint found -- the module carries no kernel", file=sys.stderr)
    sys.exit(4)

model_names = {'5': 'GLCompute', '6': 'Kernel', '5267': 'TaskINTEL'}
model = model_names.get(entry[0], entry[0])
func_id = entry[1]
params = fn_params.get(func_id, [])
ret, ptys = func_ty.get(funcs.get(func_id, '?'), ('?', []))

# The reference's functor layout, Kern<128>: xpu/csrc/inkling2_ops.sycl:547-549.
functor_fields = [
    ('q', 'const fp16*'), ('k', 'const fp16*'), ('v', 'const fp16*'), ('o', 'float*'),
    ('table', 'const uint16_t*'), ('sink', 'const float*'),
    ('Lq', 'int'), ('Lk', 'int'), ('Hq', 'int'), ('Hk', 'int'), ('G', 'int'),
    ('qpos0', 'int'), ('kpos0', 'int'), ('window', 'int'), ('E', 'int'), ('QTW', 'int'),
    ('scale_log2', 'float'),
]

L = []
L.append(f"module              {spv}")
L.append(f"size                {os.path.getsize(spv)} bytes")
L.append(f"sha256              {hashlib.sha256(open(spv, 'rb').read()).hexdigest()}")
L.append("")
L.append(f"entry point         {entry[2]}")
L.append(f"execution model     {model} ({entry[0]})")
L.append(f"interface variables {len(entry[3])}: " + ", ".join(names.get(v, v) for v in entry[3]))
L.append("")
L.append("capabilities (what the driver's SPIR-V reader must accept):")
L.extend("  " + c for c in caps)
L.append("")
L.append("extensions:")
L.extend("  " + e for e in exts)
L.append("")
L.append("execution modes:")
for fid, m, ops in modes:
    L.append(f"  {mode_name.get(m, 'mode#' + m)}" + (f" = {' '.join(ops)}" if ops else ""))
L.append("")
L.append("kernel arguments, in order (the OpFunctionParameter of the entry point's function body):")
L.append(f"  {'#':>3}  {'spir-v id':>9}  {'type':<14} {'bytes':>6}  reference field      field type")
L.append("       (pointer width 8 B: the module states the storage class, not a width;")
L.append("        CL_DEVICE_ADDRESS_BITS = 64 on the Intel Xe GPUs)")
for i, p in enumerate(params):
    kind, b = size_of(p[0], 8)
    field = functor_fields[i][0] if i < len(functor_fields) else '?'
    ftype = functor_fields[i][1] if i < len(functor_fields) else '?'
    L.append(f"  {i:>3}  id {p[0]:>6}  {kind:<14} {('' if b is None else b):>6}  {field:<20} {ftype}")
L.append("")
L.append(f"count: {len(params)} argument(s); the reference functor holds {len(functor_fields)} fields")
extra = len(params) - len(functor_fields)
L.append(f"implicit arguments: {'none' if extra == 0 else str(extra) + ' beyond the functor fields'}")
L.append("  the only interface variable is __spirv_BuiltInGlobalInvocationId (the NDRange global id);")
L.append("  no local-size, global-offset, printf or work-group-count builtin is declared.")
L.append(f"entry function type: {funcs.get(func_id, '?')} -> params {' '.join(ptys)}")
print("\n".join(L))

with open(os.path.join(out, abi_name), 'w') as f:
    f.write("# trellis-serve fa::Kern<128>, read out of the compiled SPIR-V\n")
    f.write(f"entry_point\t{entry[2]}\n")
    f.write(f"model\t{model}\n")
    for i, p in enumerate(params):
        kind, b = size_of(p[0], 8)
        f.write(f"arg\t{i}\t{kind}\t{b if b is not None else '-'}\n")
    for c in caps:
        f.write(f"capability\t{c}\n")
    for e in exts:
        f.write(f"extension\t{e}\n")
with open(os.path.join(out, names_name), 'w') as f:
    f.write(entry[2] + "\n")
