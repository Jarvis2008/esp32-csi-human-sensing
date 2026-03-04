# CSI_FRAME_V2 Contract

Network byte order for multi-byte scalar fields.

## Header
- `magic` (`uint16`) = `0xC511`
- `version` (`uint8`) = `2`
- `type` (`uint8`) = `0x04` (CSI)
- `seq` (`uint32`)
- `timestamp_us` (`uint64`)
- `payload_len` (`uint16`)

## Payload
- `node_id` (`int16`)
- `rssi` (`int8`)
- `channel` (`uint8`)
- `csi_len` (`uint16`) number of int8 values in `csi_iq`
- `csi_iq` (`int8[csi_len]`) interleaved IQ (`I0,Q0,I1,Q1,...`)

## Validation Rules
- `magic` must match.
- `version` must be exactly `2`.
- `type` must be `0x04`.
- `payload_len` must match actual payload bytes.
- `csi_len` must be even and `<= 512`.
