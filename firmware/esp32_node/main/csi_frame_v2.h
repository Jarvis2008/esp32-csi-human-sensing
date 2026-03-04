#ifndef CSI_FRAME_V2_H
#define CSI_FRAME_V2_H

#include <stdint.h>

#define CSI_FRAME_V2_MAGIC 0xC511
#define CSI_FRAME_V2_VERSION 2
#define CSI_FRAME_V2_TYPE_CSI 0x04
#define CSI_FRAME_V2_MAX_CSI_LEN 512

#pragma pack(push, 1)
typedef struct {
    uint16_t magic;
    uint8_t version;
    uint8_t type;
    uint32_t seq;
    uint64_t timestamp_us;
    uint16_t payload_len;
} csi_frame_v2_header_t;

typedef struct {
    int16_t node_id;
    int8_t rssi;
    uint8_t channel;
    uint16_t csi_len;
    int8_t csi_iq[CSI_FRAME_V2_MAX_CSI_LEN];
} csi_frame_v2_payload_t;
#pragma pack(pop)

#endif
