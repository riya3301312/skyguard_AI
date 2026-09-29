/* SkyGuard AI edge checks: lightweight per-observation QC for an ESP32-class AWS logger.
 * Portable C99, no heap, fixed-size state. The full server model does NOT run here.
 * Checks: missing, range/sentinel, spike (rate of change), flatline, basic T/RH consistency.
 */
#ifndef SKYGUARD_EDGE_H
#define SKYGUARD_EDGE_H
#include <stddef.h>
#include <stdint.h>

#define SG_NVARS 3 /* 0 = temperature_c, 1 = pressure_hpa, 2 = relative_humidity_pct */

/* flag bits: per variable v, bit (4*v + k) with k = 0 missing, 1 range/sentinel, 2 spike, 3 flatline */
#define SG_MISSING 0u
#define SG_RANGE 1u
#define SG_SPIKE 2u
#define SG_FLAT 3u
#define SG_FLAG(v, k) (1u << (4u * (v) + (k)))
#define SG_MV_INCONSISTENT (1u << 12)

typedef struct {
    float min[SG_NVARS], max[SG_NVARS];
    float max_step[SG_NVARS];   /* spike threshold: |x - previous| per observation */
    uint16_t flat_n;            /* identical consecutive values => flatline */
    float mv_dt_min;            /* T/RH check: |dT| above this ... */
    float mv_drh_max;           /* ... with |dRH| below this, or */
    float mv_dtd_max;           /* dew-point jump above this => inconsistent */
} sg_edge_config_t;

typedef struct {
    float last[SG_NVARS];
    uint16_t flat_run[SG_NVARS];
    uint32_t history;           /* 1 bit per recent observation: flagged or not (last 32) */
    uint8_t has_last;
} sg_edge_state_t;

void sg_edge_default_config(sg_edge_config_t *cfg);
void sg_edge_init(sg_edge_state_t *st);
/* values: measurements; present[v] = 0 if value v was not received. Returns flag bitmask. */
uint32_t sg_edge_check(sg_edge_state_t *st, const sg_edge_config_t *cfg, const float values[SG_NVARS],
                       const uint8_t present[SG_NVARS], uint8_t *health_out);
size_t sg_edge_state_size(void);
size_t sg_edge_config_size(void);
#endif
