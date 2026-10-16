#include <stdio.h>
#include "factorial.h"

int main() {
    double inputs[FACTORIAL_COUNT_PARAMS];
    double outs[FACTORIAL_COUNT_OBS];

    inputs[0] = 8;
    factorial(outs, NULL, 0, inputs);
    printf("%lf\n", outs[0]);
    return 0;
}
