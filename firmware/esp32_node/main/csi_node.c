#include <arpa/inet.h>
#include <errno.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/ringbuf.h"
#include "freertos/task.h"
#include "nvs.h"
#include "nvs_flash.h"

#include "csi_stream_v2.h"

#define WIFI_CONNECTED_BIT BIT0
#define WIFI_FAIL_BIT BIT1

#define CSI_RING_SLOTS 32

static const char *TAG = "CSI_NODE";

static EventGroupHandle_t wifi_event_group;
static RingbufHandle_t csi_ring;
static volatile int64_t last_capture_us = 0;
static volatile uint32_t csi_dropped = 0;
static volatile uint32_t csi_captured = 0;

typedef struct {
    uint16_t node_id;
    char target_ip[16];
    uint16_t target_port;
    uint16_t capture_interval_ms;
    uint16_t traffic_gen_interval_ms;
    char ssid[33];
    char password[65];
} runtime_config_t;

static runtime_config_t cfg = {
    .node_id = CONFIG_CSI_DEFAULT_NODE_ID,
    .target_ip = CONFIG_CSI_DEFAULT_TARGET_IP,
    .target_port = CONFIG_CSI_DEFAULT_TARGET_PORT,
    .capture_interval_ms = CONFIG_CSI_DEFAULT_CAPTURE_INTERVAL_MS,
    .traffic_gen_interval_ms = CONFIG_CSI_DEFAULT_TRAFFIC_GEN_INTERVAL_MS,
    .ssid = CONFIG_CSI_WIFI_SSID,
    .password = CONFIG_CSI_WIFI_PASSWORD,
};

static uint32_t csi_seq = 0;

static void load_runtime_config(void)
{
    nvs_handle_t nvs;
    if (nvs_open("csi_cfg", NVS_READONLY, &nvs) != ESP_OK) {
        ESP_LOGI(TAG, "NVS namespace csi_cfg not found, using defaults");
        return;
    }

    uint16_t u16;
    size_t len;

    if (nvs_get_u16(nvs, "node_id", &u16) == ESP_OK) {
        cfg.node_id = u16;
    }
    if (nvs_get_u16(nvs, "target_port", &u16) == ESP_OK) {
        cfg.target_port = u16;
    }
    if (nvs_get_u16(nvs, "capture_ms", &u16) == ESP_OK) {
        cfg.capture_interval_ms = u16;
    }
    if (nvs_get_u16(nvs, "traffic_ms", &u16) == ESP_OK) {
        cfg.traffic_gen_interval_ms = u16;
    }

    len = sizeof(cfg.target_ip);
    nvs_get_str(nvs, "target_ip", cfg.target_ip, &len);

    len = sizeof(cfg.ssid);
    nvs_get_str(nvs, "ssid", cfg.ssid, &len);

    len = sizeof(cfg.password);
    nvs_get_str(nvs, "password", cfg.password, &len);

    nvs_close(nvs);

    ESP_LOGI(TAG,
             "Runtime config: node_id=%u target=%s:%u capture=%ums traffic=%ums ssid=%s",
             cfg.node_id,
             cfg.target_ip,
             cfg.target_port,
             cfg.capture_interval_ms,
             cfg.traffic_gen_interval_ms,
             cfg.ssid);
}

static void wifi_event_handler(void *arg, esp_event_base_t event_base, int32_t event_id, void *event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        xEventGroupSetBits(wifi_event_group, WIFI_FAIL_BIT);
        esp_wifi_connect();
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        xEventGroupSetBits(wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

static void wifi_init(void)
{
    wifi_event_group = xEventGroupCreate();
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t wifi_cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&wifi_cfg));

    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, wifi_event_handler, NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, wifi_event_handler, NULL, NULL));

    wifi_config_t station_cfg = {0};
    strncpy((char *)station_cfg.sta.ssid, cfg.ssid, sizeof(station_cfg.sta.ssid) - 1);
    strncpy((char *)station_cfg.sta.password, cfg.password, sizeof(station_cfg.sta.password) - 1);

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &station_cfg));
    ESP_ERROR_CHECK(esp_wifi_start());

    wifi_csi_config_t csi_cfg = {
        .lltf_en = true,
        .htltf_en = true,
        .stbc_htltf2_en = true,
        .ltf_merge_en = true,
        .channel_filter_en = false,
        .manu_scale = false,
        .shift = false,
    };

    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_cfg));
}

static void wifi_csi_cb(void *ctx, wifi_csi_info_t *info)
{
    if (!info || !info->buf || !csi_ring) {
        return;
    }

    int64_t now_us = esp_timer_get_time();
    int64_t min_interval_us = ((int64_t)cfg.capture_interval_ms) * 1000;
    if (min_interval_us > 0 && (now_us - last_capture_us) < min_interval_us) {
        csi_dropped++;
        return;
    }
    last_capture_us = now_us;

    csi_sample_t sample = {
        .timestamp_us = (uint64_t)now_us,
        .rssi = info->rx_ctrl.rssi,
        .channel = info->rx_ctrl.channel,
    };

    sample.csi_len = info->len > CSI_FRAME_V2_MAX_CSI_LEN ? CSI_FRAME_V2_MAX_CSI_LEN : info->len;
    memcpy(sample.csi_iq, info->buf, sample.csi_len);

    BaseType_t ok = xRingbufferSendFromISR(csi_ring, &sample, sizeof(sample), NULL);
    if (ok == pdTRUE) {
        csi_captured++;
    } else {
        csi_dropped++;
    }
}

static void traffic_generator_task(void *arg)
{
    if (cfg.traffic_gen_interval_ms == 0) {
        vTaskDelete(NULL);
        return;
    }

    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (sock < 0) {
        ESP_LOGE(TAG, "traffic socket create failed: errno=%d", errno);
        vTaskDelete(NULL);
        return;
    }

    struct sockaddr_in dst = {0};
    dst.sin_family = AF_INET;
    dst.sin_port = htons(12345);

    esp_netif_t *netif = esp_netif_get_handle_from_ifkey("WIFI_STA_DEF");
    esp_netif_ip_info_t ip_info = {0};
    if (!netif || esp_netif_get_ip_info(netif, &ip_info) != ESP_OK) {
        close(sock);
        vTaskDelete(NULL);
        return;
    }
    dst.sin_addr.s_addr = ip_info.gw.addr;

    const uint8_t ping[4] = {0xDE, 0xAD, 0xBE, 0xEF};
    while (1) {
        sendto(sock, ping, sizeof(ping), 0, (struct sockaddr *)&dst, sizeof(dst));
        vTaskDelay(pdMS_TO_TICKS(cfg.traffic_gen_interval_ms));
    }
}

static void csi_udp_stream_task(void *arg)
{
    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (sock < 0) {
        ESP_LOGE(TAG, "udp socket create failed: errno=%d", errno);
        vTaskDelete(NULL);
        return;
    }

    if (strcmp(cfg.target_ip, "255.255.255.255") == 0) {
        int broadcast = 1;
        setsockopt(sock, SOL_SOCKET, SO_BROADCAST, &broadcast, sizeof(broadcast));
    }

    struct sockaddr_in dest = {0};
    dest.sin_family = AF_INET;
    dest.sin_port = htons(cfg.target_port);
    dest.sin_addr.s_addr = inet_addr(cfg.target_ip);

    uint8_t frame_buf[sizeof(csi_frame_v2_header_t) + sizeof(int16_t) + sizeof(int8_t) + sizeof(uint8_t) + sizeof(uint16_t) + CSI_FRAME_V2_MAX_CSI_LEN];
    int64_t last_stats_us = esp_timer_get_time();
    uint32_t sent_count = 0;

    while (1) {
        size_t item_size = 0;
        csi_sample_t *sample = (csi_sample_t *)xRingbufferReceive(csi_ring, &item_size, pdMS_TO_TICKS(200));
        if (!sample) {
            continue;
        }

        if (item_size == sizeof(csi_sample_t)) {
            size_t frame_size = csi_frame_v2_pack(frame_buf, sizeof(frame_buf), csi_seq++, cfg.node_id, sample);
            if (frame_size > 0) {
                int sent = sendto(sock, frame_buf, frame_size, 0, (struct sockaddr *)&dest, sizeof(dest));
                if (sent > 0) {
                    sent_count++;
                }
            }
        }

        vRingbufferReturnItem(csi_ring, sample);

        int64_t now_us = esp_timer_get_time();
        if ((now_us - last_stats_us) >= 5000000) {
            float elapsed = (float)(now_us - last_stats_us) / 1000000.0f;
            ESP_LOGI(TAG, "CSI stats: sent=%u captured=%u dropped=%u rate=%.1f pkt/s",
                     sent_count,
                     (unsigned)csi_captured,
                     (unsigned)csi_dropped,
                     sent_count / elapsed);
            sent_count = 0;
            last_stats_us = now_us;
        }
    }
}

void app_main(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    load_runtime_config();

    if (strlen(cfg.ssid) == 0) {
        ESP_LOGE(TAG, "WiFi SSID is empty. Set CONFIG_CSI_WIFI_SSID or NVS key 'ssid'");
        return;
    }

    csi_ring = xRingbufferCreate(CSI_RING_SLOTS * sizeof(csi_sample_t), RINGBUF_TYPE_NOSPLIT);
    if (!csi_ring) {
        ESP_LOGE(TAG, "failed to allocate csi ring buffer");
        return;
    }

    wifi_init();

    EventBits_t bits = xEventGroupWaitBits(
        wifi_event_group,
        WIFI_CONNECTED_BIT | WIFI_FAIL_BIT,
        pdTRUE,
        pdFALSE,
        pdMS_TO_TICKS(15000));

    if (!(bits & WIFI_CONNECTED_BIT)) {
        ESP_LOGE(TAG, "WiFi connect timeout or failure");
        return;
    }

    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(wifi_csi_cb, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));

    xTaskCreatePinnedToCore(csi_udp_stream_task, "csi_udp", 4096, NULL, 5, NULL, 1);
    xTaskCreatePinnedToCore(traffic_generator_task, "traffic", 2048, NULL, 4, NULL, 0);
}
