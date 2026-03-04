#ifndef CSI_STREAM_V2_H
#define CSI_STREAM_V2_H

#include <stddef.h>
#include <stdint.h>

#include "csi_frame_v2.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t timestamp_us;
    int8_t rssi;
    uint8_t channel;
    uint16_t csi_len;
    int8_t csi_iq[CSI_FRAME_V2_MAX_CSI_LEN];
} csi_sample_t;

size_t csi_frame_v2_pack(
    uint8_t *buffer,
    size_t buffer_size,
    uint32_t seq,
    uint16_t node_id,
    const csi_sample_t *sample
);

#ifdef __cplusplus
}
#endif

#endif
