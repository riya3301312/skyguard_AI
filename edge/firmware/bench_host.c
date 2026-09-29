/* Host harness: reads "t,p,rh" lines (empty field = missing) from stdin, runs sg_edge_check on
 * each, prints "flags,health" per line to stdout and timing to stderr.
 * Timing is measured on the HOST CPU, not on an ESP32. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "skyguard_edge.h"

#define MAXN 200000

static float vals[MAXN][SG_NVARS];
static uint8_t pres[MAXN][SG_NVARS];
static uint32_t flags[MAXN];
static uint8_t health[MAXN];

int main(void) {
    char line[256];
    int n = 0;
    while (n < MAXN && fgets(line, sizeof line, stdin)) {
        char *p = line;
        for (int v = 0; v < SG_NVARS; v++) {
            char *end;
            float x = strtof(p, &end);
            pres[n][v] = (end != p);
            vals[n][v] = pres[n][v] ? x : 0.0f;
            p = strchr(end, ',');
            p = p ? p + 1 : end;
        }
        n++;
    }
    sg_edge_config_t cfg;
    sg_edge_default_config(&cfg);
    sg_edge_state_t st;
    struct timespec a, b;
    const int reps = 20;
    timespec_get(&a, TIME_UTC);
    for (int r = 0; r < reps; r++) {
        sg_edge_init(&st);
        for (int i = 0; i < n; i++) flags[i] = sg_edge_check(&st, &cfg, vals[i], pres[i], &health[i]);
    }
    timespec_get(&b, TIME_UTC);
    double ns = ((double)(b.tv_sec - a.tv_sec) * 1e9 + (double)(b.tv_nsec - a.tv_nsec)) / ((double)n * reps);
    for (int i = 0; i < n; i++) printf("%u,%u\n", flags[i], health[i]);
    fprintf(stderr, "{\"observations\": %d, \"repetitions\": %d, \"ns_per_observation\": %.2f, "
                    "\"state_bytes\": %zu, \"config_bytes\": %zu}\n", n, reps, ns, sg_edge_state_size(), sg_edge_config_size());
    return 0;
}
