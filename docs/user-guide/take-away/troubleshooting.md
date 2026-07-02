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
