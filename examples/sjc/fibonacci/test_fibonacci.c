#include <stdio.h>
#include "fibonacci.h"

int main() {
    double inputs[FIBONACCI_COUNT_PARAMS] = {25.0};
    double outs[FIBONACCI_COUNT_OBS];

    int status = fibonacci(outs, NULL, 0, inputs);    // 0 on success
    fibonacci(outs, NULL, 0, inputs);
    printf("%lf\n", outs[0]);
    return 0;
}
