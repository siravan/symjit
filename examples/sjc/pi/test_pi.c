#include <stdio.h>
#include "pi.h"

int main() {
    double mem[PI_MEM_SIZE] = {0};

    pi(mem, NULL, 0, NULL);
    printf("%lf\t%lf\n", mem[1], mem[2]);
    return 0;
}
