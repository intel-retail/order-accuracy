# Troubleshooting

This article covers common issues and how to resolve them. If you encounter a problem not
listed here, see [Order Accuracy Issues](https://github.com/intel-retail/order-accuracy/issues).

## Build Fails (network / pip)

```bash
docker compose build --no-cache
```

## Model File Not Found

```bash
# Verify models were correctly set up
ls ../ovms-service/models/
ls models/easyocr/
ls models/yolo11n_int8_openvino_model/
```

## OVMS Not Starting

```bash
# Check logs
docker logs oa_ovms_vlm

# Verify model files exist
ls -la ../ovms-service/models/
```

## Connection Refused to OVMS (port 8001)

OVMS can take 2–5 minutes to load the model. Wait and check:

```bash
docker logs -f oa_ovms_vlm | grep "Serving"
```

## MinIO Bucket Errors

```bash
# Recreate MinIO with fresh volumes
make down
docker volume rm take-away_minio_data
make up
```

## GPU Not Detected

```bash
sudo usermod -aG render $USER
# Log out and log back in, then restart services
make down && make up
```

## GPU Out of Memory

```bash
# Switch to CPU: set both in .env, then re-export model
TARGET_DEVICE=CPU
OPENVINO_DEVICE=CPU
# Then:
cd ../ovms-service && ./setup_models.sh --app take-away
cd ../take-away && make down && make up
```

## OVMS Returns HTTP 404 for Every Request

`OVMS_MODEL_NAME` must exactly match the name registered in
`ovms-service/models/config.json`, which `setup_models.sh` generates including the
precision suffix (for example `openbmb/MiniCPM-V-4_5-int4`). Any mismatch makes
every inference request return 404.

```bash
# Compare the registered name with the configured one
grep '"name"' ../ovms-service/models/config.json
grep OVMS_MODEL_NAME .env
```

## Items Missing from Detections / High Latency

MiniCPM-V-4.5 is a hybrid reasoning model. If `VLM_ENABLE_THINKING` is enabled, the
model emits a `<think>` reasoning block that consumes the entire
`max_completion_tokens` budget, so the answer is truncated — orders come back with
items missing and latency is several times higher.

```bash
# Confirm thinking is disabled
grep VLM_ENABLE_THINKING .env        # expected: false
docker exec oa_service env | grep VLM_ENABLE_THINKING
```

Set `VLM_ENABLE_THINKING=false` in `.env` and restart (`make down && make up`).

## Benchmark Reports Zero Transactions

An all-zero result (`total_transactions: 0`, `No vlm_metrics_logger files found`)
usually means every order failed before inference. The most common cause is an
invalid or edited `config/orders.json` — a trailing comma or a missing entry makes
the file unparseable, and the service logs `Order not found in orders.json`.

```bash
# Validate the file before benchmarking
python3 -c "import json; print(json.load(open('config/orders.json')).keys())"
docker logs oa_service 2>&1 | grep -i "orders.json"
```

Order IDs in `config/orders.json` must also match the order numbers that appear in
the benchmark video, and each order's expected items must reflect what is actually
visible — otherwise correct detections are still reported as mismatches.
