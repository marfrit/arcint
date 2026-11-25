// Before the fix: the folded-norm path's norm partial sums were written at red[q*LS+lane],
// which for (k=1,nq=3) covers [512..767] and for (k=2,nq=4) [768..1023] - the slots the first
// reduction's dot results sit at and other sub-groups read. Prints 4 overlaps, exit 1.
// the has_nw path's local-memory aliasing: the first reduce leaves results at red[p*LS]
// (p = 0..ns-1 sums of squares, then ns+q*ns+s dots); the norm partials are written at
// red[q*LS + lane]. Prints every overlap between a written byte range and a result the
// later phase reads. Before the fix: q >= 2 overwrites the dot results at (ns+q'*ns+s)*LS
// while other sub-groups may still read them.
#include <stdio.h>
int main(void) {
    int bad = 0;
    for (int k = 1; k <= 4; ++k) for (int nq = 1; nq <= 4; ++nq) {
        const int ns = k + 1, np1 = ns*(1 + nq);
        // dot result slot offsets (byte index in floats) read by the pw step
        int rd[64], nrd = 0;
        for (int q = 0; q < nq; ++q) for (int s = 0; s < ns; ++s) rd[nrd++] = (ns + q*ns + s)*256;
        // norm partial writes: [q*256, q*256+255] and the norm result slot q*256
        for (int q = 0; q < nq; ++q) {
            const int wlo = q*256, whi = q*256 + 255;
            for (int i = 0; i < nrd; ++i) if (rd[i] >= wlo && rd[i] <= whi) { printf("OVERLAP k=%d nq=%d: norm partials [%d..%d] cover dot result at %d\n", k, nq, wlo, whi, rd[i]); bad++; }
        }
        (void) np1;
    }
    printf("%d overlaps in the folded-norm path's local layout\n", bad);
    return bad ? 1 : 0;
}
