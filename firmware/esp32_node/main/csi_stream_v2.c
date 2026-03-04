#include "csi_stream_v2.h"

#include <arpa/inet.h>
#include <stdbool.h>
#include <string.h>

#define CSI_PAYLOAD_PREFIX_SIZE (sizeof(int16_t) + sizeof(int8_t) + sizeof(uint8_t) + sizeof(uint16_t))

static uint64_t htonll_u64(uint64_t value)
{
    uint8_t out[8];
    out[0] = (uint8_t)((value >> 56) & 0xFF);
    out[1] = (uint8_t)((value >> 48) & 0xFF);
    out[2] = (uint8_t)((value >> 40) & 0xFF);
    out[3] = (uint8_t)((value >> 32) & 0xFF);
    out[4] = (uint8_t)((value >> 24) & 0xFF);
    out[5] = (uint8_t)((value >> 16) & 0xFF);
    out[6] = (uint8_t)((value >> 8) & 0xFF);
    out[7] = (uint8_t)(value & 0xFF);

    uint64_t net_value = 0;
    memcpy(&net_value, out, sizeof(net_value));
    return net_value;
}

size_t csi_frame_v2_pack(
    uint8_t *buffer,
    size_t buffer_size,
    uint32_t seq,
    uint16_t node_id,
    const csi_sample_t *sample)
{
    if (!buffer || !sample) {
        return 0;
    }

    if (sample->csi_len > CSI_FRAME_V2_MAX_CSI_LEN) {
        return 0;
    }

    size_t payload_len = CSI_PAYLOAD_PREFIX_SIZE + sample->csi_len;
    size_t frame_size = sizeof(csi_frame_v2_header_t) + payload_len;
    if (buffer_size < frame_size) {
        return 0;
    }

    csi_frame_v2_header_t header = {
        .magic = htons(CSI_FRAME_V2_MAGIC),
        .version = CSI_FRAME_V2_VERSION,
        .type = CSI_FRAME_V2_TYPE_CSI,
        .seq = htonl(seq),
        .timestamp_us = htonll_u64(sample->timestamp_us),
        .payload_len = htons((uint16_t)payload_len),
    };

    memcpy(buffer, &header, sizeof(header));

    size_t offset = sizeof(header);
    int16_t node_id_be = htons((uint16_t)node_id);
    uint16_t csi_len_be = htons(sample->csi_len);

    memcpy(buffer + offset, &node_id_be, sizeof(node_id_be));
    offset += sizeof(node_id_be);

    memcpy(buffer + offset, &sample->rssi, sizeof(sample->rssi));
    offset += sizeof(sample->rssi);

    memcpy(buffer + offset, &sample->channel, sizeof(sample->channel));
    offset += sizeof(sample->channel);

    memcpy(buffer + offset, &csi_len_be, sizeof(csi_len_be));
    offset += sizeof(csi_len_be);

    if (sample->csi_len > 0) {
        memcpy(buffer + offset, sample->csi_iq, sample->csi_len);
    }

    return frame_size;
}
