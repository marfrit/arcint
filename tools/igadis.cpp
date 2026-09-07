// igadis: disassemble raw Xe2 ISA dumps (the OpenCL intercept layer's CLI_DumpKernelISABinaries
// output, or any raw kernel ISA) through Intel's libiga64 and count instructions by class:
// sends (memory messages), movs and the indirect ones among them (r[a0...]: variable-index
// shuffles), mads, shifts, compares, selects. Used for DESIGN §7.0.2bh (the K-quant decode
// kernels: Q6_K 4,729 instructions per sixteen-row body against Q4_K's 2,845).
//
// Build against the installed compiler's library with the header of the same IGC version:
//   curl -o iga/iga.h https://raw.githubusercontent.com/intel/intel-graphics-compiler/v2.38.2/visa/iga/IGALibrary/api/iga.h
//   g++ -O1 -Iiga igadis.cpp -o igadis /usr/local/lib/libiga64.so.2 -Wl,-rpath,/usr/local/lib
// Usage: igadis file...   (one line per file; the disassembly text is owned by the IGA context)
#include "iga.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
static int starts(const char* s, const char* p) { return strncmp(s, p, strlen(p)) == 0; }
int main(int argc, char** argv) {
    for (int a = 1; a < argc; ++a) {
        FILE* f = fopen(argv[a], "rb"); if (!f) { perror(argv[a]); continue; }
        fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
        void* buf = malloc(len); fread(buf, 1, len, f); fclose(f);
        iga_context_options_t co = IGA_CONTEXT_OPTIONS_INIT(IGA_XE2);
        iga_context_t ctx; iga_status_t st = iga_context_create(&co, &ctx);
        if (st != IGA_SUCCESS) { fprintf(stderr, "context: %s\n", iga_status_to_string(st)); return 1; }
        iga_disassemble_options_t dopts = IGA_DISASSEMBLE_OPTIONS_INIT();
        char* text = NULL; char* copy = NULL;
        st = iga_context_disassemble(ctx, &dopts, buf, (uint32_t)len, NULL, NULL, &text);
        if (st != IGA_SUCCESS || !text) { fprintf(stderr, "%s: disassemble: %s\n", argv[a], iga_status_to_string(st)); iga_context_release(ctx); continue; }
        // IGADIS_DUMP (set to anything) writes the disassembly text beside the input as <file>.asm
        if (getenv("IGADIS_DUMP")) { std::string o = std::string(argv[a]) + ".asm"; FILE* d = fopen(o.c_str(), "w"); if (d) { fputs(text, d); fclose(d); } }
        long instr = 0, send = 0, mov = 0, indirect = 0, mad = 0, shifts = 0, cmp = 0, sel = 0, dpas = 0;
        copy = strdup(text);  // the text belongs to the context; strtok needs its own copy
        for (char* line = strtok(copy, "\n"); line; line = strtok(NULL, "\n")) {
            char* s = line; while (*s == ' ' || *s == '\t') ++s;
            if (!*s || *s == '/' || *s == '.' || starts(s, "L") ) continue;
            if (starts(s, "(W)")) { s += 3; while (*s == ' ') ++s; }
            if (*s == '(') { while (*s && *s != ')') ++s; if (*s) ++s; while (*s == ' ') ++s; }  // predication
            if (!(*s >= 'a' && *s <= 'z')) continue;
            ++instr;
            if (starts(s, "send")) ++send;
            if (starts(s, "mov")) { ++mov; if (strstr(s, "r[a0")) ++indirect; }
            if (starts(s, "mad")) ++mad;
            if (starts(s, "shl") || starts(s, "shr") || starts(s, "asr")) ++shifts;
            if (starts(s, "cmp")) ++cmp;
            if (starts(s, "sel")) ++sel;
            if (starts(s, "dpas")) ++dpas;
        }
        printf("%s: %ld bytes, %ld instr, %ld dpas, %ld send, %ld mov (%ld indirect), %ld mad, %ld shift, %ld cmp, %ld sel\n", argv[a], len, instr, dpas, send, mov, indirect, mad, shifts, cmp, sel);
        free(copy); iga_context_release(ctx); free(buf);
    }
    return 0;
}
