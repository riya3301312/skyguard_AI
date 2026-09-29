#include "skyguard_edge.h"
#include <math.h>

void sg_edge_default_config(sg_edge_config_t *c) {
    const float mn[SG_NVARS] = {-60.0f, 850.0f, 0.0f}, mx[SG_NVARS] = {60.0f, 1090.0f, 100.0f};
    const float step[SG_NVARS] = {6.0f, 4.0f, 30.0f};
    for (int v = 0; v < SG_NVARS; v++) { c->min[v] = mn[v]; c->max[v] = mx[v]; c->max_step[v] = step[v]; }
    c->flat_n = 8;
    c->mv_dt_min = 4.0f;
    c->mv_drh_max = 0.5f;
    c->mv_dtd_max = 8.0f;
}

void sg_edge_init(sg_edge_state_t *st) {
    for (int v = 0; v < SG_NVARS; v++) { st->last[v] = 0.0f; st->flat_run[v] = 0; }
    st->history = 0;
    st->has_last = 0;
}

static float dewpoint(float t, float rh) {
    if (rh < 1.0f) rh = 1.0f;
    float g = logf(rh / 100.0f) + 17.625f * t / (243.04f + t);
    return 243.04f * g / (17.625f - g);
}

static uint8_t popcount32(uint32_t x) {
    uint8_t n = 0;
    while (x) { x &= x - 1u; n++; }
    return n;
}

uint32_t sg_edge_check(sg_edge_state_t *st, const sg_edge_config_t *c, const float x[SG_NVARS],
                       const uint8_t present[SG_NVARS], uint8_t *health_out) {
    uint32_t f = 0;
    uint8_t ok[SG_NVARS];
    for (int v = 0; v < SG_NVARS; v++) {
        ok[v] = 0;
        if (!present[v] || isnan(x[v])) { f |= SG_FLAG(v, SG_MISSING); continue; }
        if (x[v] < c->min[v] || x[v] > c->max[v]) { f |= SG_FLAG(v, SG_RANGE); continue; } /* includes -9999 etc. */
        ok[v] = 1;
        if (st->has_last) {
            float d = x[v] - st->last[v];
            if (fabsf(d) > c->max_step[v]) f |= SG_FLAG(v, SG_SPIKE);
            st->flat_run[v] = (d == 0.0f) ? (uint16_t)(st->flat_run[v] + 1u) : 0u;
            if (st->flat_run[v] + 1u >= c->flat_n) f |= SG_FLAG(v, SG_FLAT);
        }
    }
    if (st->has_last && ok[0] && ok[2]) {
        float dt = x[0] - st->last[0], drh = x[2] - st->last[2];
        float dtd = dewpoint(x[0], x[2]) - dewpoint(st->last[0], st->last[2]);
        if ((fabsf(dt) >= c->mv_dt_min && fabsf(drh) <= c->mv_drh_max) || fabsf(dtd) >= c->mv_dtd_max)
            f |= SG_MV_INCONSISTENT;
    }
    for (int v = 0; v < SG_NVARS; v++) if (ok[v]) st->last[v] = x[v];
    st->has_last = (uint8_t)(st->has_last | ok[0] | ok[1] | ok[2]);
    st->history = (st->history << 1) | (f ? 1u : 0u);
    if (health_out) *health_out = (uint8_t)(100u - (60u * popcount32(st->history)) / 32u);
    return f;
}

size_t sg_edge_state_size(void) { return sizeof(sg_edge_state_t); }
size_t sg_edge_config_size(void) { return sizeof(sg_edge_config_t); }
