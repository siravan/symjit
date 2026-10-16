/* Runs symjit models through the library's C ABI (the functions engine.py calls), for
   test_emulated.py: cross-compiled for aarch64/riscv64 and run under qemu-user with the
   symjit library built for the same target.

       emulated_driver LIBSYMJIT MANIFEST

   Each line of MANIFEST is a case:

       func <ty> <opt> <npoints> <fast> <prefix>
           compile(<prefix>.model, ty, opt); <prefix>.in holds the states as count_states
           rows of npoints doubles, then the count_params parameters. Writes count_obs
           rows of npoints doubles: <prefix>.scalar (execute, one point at a time),
           <prefix>.matrix (execute_matrix: the SIMD kernels and threads) and, if <fast>
           and the model has a fast kernel, <prefix>.fast (one row).
       eval <ty> <opt> <npoints> <num_params> <prefix>
           translate(<prefix>.model, ...) (a Symbolica evaluator); <prefix>.in holds
           npoints rows of count_params doubles; writes <prefix>.eval (evaluate_matrix,
           npoints rows of count_obs doubles).

   A compilation error is written to <prefix>.err. Counts are in doubles (complex values are
   (re, im) pairs).
*/
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void *(*compile_t)(const char *, const char *, unsigned, const void *);
typedef void *(*translate_t)(const char *, const char *, unsigned, void *, size_t);
typedef const char *(*status_t)(const void *);
typedef size_t (*count_t)(const void *);
typedef double *(*ptr_t)(void *);
typedef _Bool (*exec_t)(void *);
typedef void *(*create_t)(void);
typedef void (*add_row_t)(void *, double *, size_t);
typedef _Bool (*exec_matrix_t)(void *, void *, void *);
typedef void (*fin_t)(void *);
typedef const void *(*fast_t)(void *);
typedef _Bool (*eval_t)(void *, const double *, size_t, double *, size_t);

static compile_t compile_;
static translate_t translate_;
static status_t check_status_;
static count_t count_states_, count_params_, count_obs_;
static ptr_t ptr_states_, ptr_params_, ptr_obs_;
static exec_t execute_;
static create_t create_defuns_, create_matrix_;
static add_row_t add_row_;
static exec_matrix_t execute_matrix_;
static fin_t finalize_, finalize_matrix_;
static fast_t fast_func_;
static eval_t evaluate_matrix_;

static void *sym(void *lib, const char *name) {
    void *p = dlsym(lib, name);
    if (!p) {
        fprintf(stderr, "missing symbol %s\n", name);
        exit(2);
    }
    return p;
}

static char *read_text(const char *path) {
    FILE *fd = fopen(path, "rb");
    if (!fd) return NULL;
    fseek(fd, 0, SEEK_END);
    long n = ftell(fd);
    fseek(fd, 0, SEEK_SET);
    char *s = malloc(n + 1);
    if (fread(s, 1, n, fd) != (size_t)n) exit(3);
    s[n] = 0;
    fclose(fd);
    return s;
}

static double *read_doubles(const char *path, size_t n) {
    FILE *fd = fopen(path, "rb");
    double *v = malloc((n + 1) * sizeof(double));
    if (!fd || fread(v, sizeof(double), n, fd) != n) {
        fprintf(stderr, "cannot read %zu doubles from %s\n", n, path);
        exit(3);
    }
    fclose(fd);
    return v;
}

static void write_doubles(const char *prefix, const char *ext, const double *v, size_t n) {
    char path[4096];
    snprintf(path, sizeof path, "%s.%s", prefix, ext);
    FILE *fd = fopen(path, "wb");
    fwrite(v, sizeof(double), n, fd);
    fclose(fd);
}

static int failed(void *q, const char *prefix) {
    const char *msg = check_status_(q);
    if (strcmp(msg, "Success") == 0) return 0;
    char path[4096];
    snprintf(path, sizeof path, "%s.err", prefix);
    FILE *fd = fopen(path, "w");
    fputs(msg, fd);
    fclose(fd);
    return 1;
}

static double call_fast(const void *f, int k, const double *x) {
    switch (k) {
    case 1: return ((double (*)(double))f)(x[0]);
    case 2: return ((double (*)(double, double))f)(x[0], x[1]);
    case 3: return ((double (*)(double, double, double))f)(x[0], x[1], x[2]);
    case 4: return ((double (*)(double, double, double, double))f)(x[0], x[1], x[2], x[3]);
    default: return 0.0;
    }
}

static void run_func(const char *ty, unsigned opt, size_t n, int want_fast, const char *prefix) {
    char path[4096];
    snprintf(path, sizeof path, "%s.model", prefix);
    char *model = read_text(path);
    void *df = create_defuns_();
    void *q = compile_(model, ty, opt, df);
    if (failed(q, prefix)) return;

    size_t ns = count_states_(q), np = count_params_(q), no = count_obs_(q);
    snprintf(path, sizeof path, "%s.in", prefix);
    double *in = read_doubles(path, ns * n + np);
    double *states = ptr_states_(q), *params = ptr_params_(q), *obs = ptr_obs_(q);
    memcpy(params, in + ns * n, np * sizeof(double));

    /* one point at a time */
    double *out = calloc(no * n + 1, sizeof(double));
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < ns; j++) states[j] = in[j * n + i];
        execute_(q);
        for (size_t j = 0; j < no; j++) out[j * n + i] = obs[j];
    }
    write_doubles(prefix, "scalar", out, no * n);

    /* all points: one row per state and per output */
    void *ms = create_matrix_(), *mo = create_matrix_();
    for (size_t j = 0; j < ns; j++) add_row_(ms, in + j * n, n);
    memset(out, 0, (no * n + 1) * sizeof(double));
    for (size_t j = 0; j < no; j++) add_row_(mo, out + j * n, n);
    execute_matrix_(q, ms, mo);
    write_doubles(prefix, "matrix", out, no * n);
    finalize_matrix_(ms);
    finalize_matrix_(mo);

    const void *f = want_fast ? fast_func_(q) : NULL;
    if (f && no == 1 && ns >= 1 && ns <= 4) {
        double x[4];
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < ns; j++) x[j] = in[j * n + i];
            out[i] = call_fast(f, (int)ns, x);
        }
        write_doubles(prefix, "fast", out, n);
    }
    free(out);
    free(in);
    free(model);
    finalize_(q);
}

static void run_eval(const char *ty, unsigned opt, size_t n, size_t num_params, const char *prefix) {
    char path[4096];
    snprintf(path, sizeof path, "%s.model", prefix);
    char *model = read_text(path);
    void *q = translate_(model, ty, opt, create_defuns_(), num_params);
    if (failed(q, prefix)) return;
    size_t np = count_params_(q), no = count_obs_(q);
    snprintf(path, sizeof path, "%s.in", prefix);
    double *in = read_doubles(path, np * n);
    double *out = calloc(no * n + 1, sizeof(double));
    evaluate_matrix_(q, in, np * n, out, no * n);
    write_doubles(prefix, "eval", out, no * n);
    free(out);
    free(in);
    free(model);
    finalize_(q);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s LIBSYMJIT MANIFEST\n", argv[0]);
        return 2;
    }
    void *lib = dlopen(argv[1], RTLD_NOW);
    if (!lib) {
        fprintf(stderr, "%s\n", dlerror());
        return 2;
    }
    compile_ = (compile_t)sym(lib, "compile");
    translate_ = (translate_t)sym(lib, "translate");
    check_status_ = (status_t)sym(lib, "check_status");
    count_states_ = (count_t)sym(lib, "count_states");
    count_params_ = (count_t)sym(lib, "count_params");
    count_obs_ = (count_t)sym(lib, "count_obs");
    ptr_states_ = (ptr_t)sym(lib, "ptr_states");
    ptr_params_ = (ptr_t)sym(lib, "ptr_params");
    ptr_obs_ = (ptr_t)sym(lib, "ptr_obs");
    execute_ = (exec_t)sym(lib, "execute");
    create_defuns_ = (create_t)sym(lib, "create_defuns");
    create_matrix_ = (create_t)sym(lib, "create_matrix");
    add_row_ = (add_row_t)sym(lib, "add_row");
    execute_matrix_ = (exec_matrix_t)sym(lib, "execute_matrix");
    finalize_ = (fin_t)sym(lib, "finalize");
    finalize_matrix_ = (fin_t)sym(lib, "finalize_matrix");
    fast_func_ = (fast_t)sym(lib, "fast_func");
    evaluate_matrix_ = (eval_t)sym(lib, "evaluate_matrix");

    FILE *fd = fopen(argv[2], "r");
    char kind[16], ty[64], prefix[4096];
    unsigned opt;
    size_t n, extra;
    while (fscanf(fd, "%15s %63s %u %zu %zu %4095s", kind, ty, &opt, &n, &extra, prefix) == 6) {
        printf("%s\n", prefix); /* the last line printed locates a crash */
        fflush(stdout);
        if (strcmp(kind, "func") == 0)
            run_func(ty, opt, n, (int)extra, prefix);
        else
            run_eval(ty, opt, n, extra, prefix);
    }
    fclose(fd);
    printf("done\n");
    return 0;
}
