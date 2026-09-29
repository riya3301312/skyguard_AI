// SkyGuard AI edge firmware sketch for ESP32 (Arduino core).
// STATUS: compiles against the same skyguard_edge.c verified on the host; NOT tested on physical
// ESP32 hardware in this prototype. Copy skyguard_edge.c/.h next to this sketch.
// read_sensor() is a placeholder for the station's T/P/RH driver (e.g. a BME280).
#include "skyguard_edge.h"

static sg_edge_config_t cfg;
static sg_edge_state_t st;

static bool read_sensor(float out[SG_NVARS]) {
  // TODO(station integrator): replace with the real sensor driver.
  out[0] = 25.0f; out[1] = 1008.0f; out[2] = 60.0f;
  return true;
}

void setup() {
  Serial.begin(115200);
  sg_edge_default_config(&cfg);
  sg_edge_init(&st);
}

void loop() {
  float x[SG_NVARS];
  uint8_t present[SG_NVARS] = {1, 1, 1};
  if (!read_sensor(x)) { present[0] = present[1] = present[2] = 0; }
  uint8_t health = 0;
  uint32_t t0 = micros();
  uint32_t flags = sg_edge_check(&st, &cfg, x, present, &health);
  uint32_t dt = micros() - t0;  // on-device processing time (report only if measured on hardware)
  // Payload for the store-and-forward uplink (see edge/station_client.py for buffering semantics).
  Serial.printf("{\"temperature_c\":%.2f,\"pressure_hpa\":%.2f,\"relative_humidity_pct\":%.2f,"
                "\"edge_flags_mask\":%lu,\"edge_health_score\":%u,\"edge_us\":%lu}\n",
                x[0], x[1], x[2], (unsigned long)flags, health, (unsigned long)dt);
  delay(60000);
}
