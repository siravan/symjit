/* test_pi.c: computes pi with kernels compiled by symjit and written as object files
   by pi.py, and checks the results against the known value.

   pi.py builds it with
       cc -Wall -O2 -I out test_pi.c out/pi_series.o out/pi_atan.o out/pi_bbp.o -lm -o out/test_pi
*/

#include <math.h>
#include <stdio.h>

#include "pi_atan.h"
#include "pi_bbp.h"
#include "pi_series.h"

/* the double nearest to pi */
static const double PI = 3.14159265358979323846264338327950288;

static int failures = 0;

/* the error of v in units in the last place of pi */
static double ulps(double v) {
    return fabs(v - PI) / (nextafter(PI, 4.0) - PI);
}

static void check(const char *what, double v, double max_ulps) {
    int ok = ulps(v) <= max_ulps;
    printf("%-44s %.17g  (%g ulp)%s\n", what, v, ulps(v), ok ? "" : "  <-- WRONG");
    failures += !ok;
}

int main(void) {
    /* Machin's formula: pi = 4 (4 arctan(1/5) - arctan(1/239)) */
    const double x = 1.0 / 5, y = 1.0 / 239;

    /* direct mode: the states, then the output, in one array */
    double mem[PI_SERIES_MEM_SIZE] = {x, y};
    if (pi_series(mem, NULL, 0, NULL) != 0) {
        return 2;
    }
    check("Machin, arctan series (pi_series)", mem[PI_SERIES_COUNT_STATES], 1);

    /* single-output models without parameters also have a `_fast` kernel */
    check("Machin, arctan series (pi_series_fast)", pi_series_fast(x, y), 1);
    check("Machin, atan from libm (pi_atan_fast)", pi_atan_fast(x, y), 1);

    /* The BBP series, sum over k = 0..n of (4/(8k+1) - 2/(8k+4) - 1/(8k+5) - 1/(8k+6)) / 16^k:
       the number of terms is an input and the sum a loop in the kernel. Indirect mode
       evaluates it for an array of n at once: `arrays` holds the state array (n), then
       the output array. */
    enum { K = 7 };
    double n[K], approx[K];
    for (int i = 0; i < K; i++) {
        n[i] = 2 * i;
    }
    symjit_slice arrays[PI_BBP_COUNT_STATES + PI_BBP_COUNT_OBS] = {{n, K}, {approx, K}};
    for (size_t i = 0; i < K; i++) {
        if (pi_bbp(NULL, arrays, i, NULL) != 0) {
            return 2;
        }
    }

    printf("\nBBP series:\n");
    for (int i = 0; i < K; i++) {
        printf("  n = %2.0f   %.17g   (%g ulp)\n", n[i], approx[i], ulps(approx[i]));
    }
    check("BBP, n = 12 (pi_bbp)", approx[K - 1], 0);

    printf("\n%s\n", failures ? "FAILED" : "ok!");
    return failures != 0;
}
