# Troubleshooting

This article covers common issues and how to resolve them. If you encounter a problem not
listed here, see [Order Accuracy Issues](https://github.com/intel-retail/order-accuracy/issues).

## Build Fails (network / pip)

```bash
docker compose build --no-cache
```

## OVMS Not Starting

```bash
# Check logs
docker logs dinein_ovms_vlm

# Verify model files exist
ls -la ../ovms-service/models/
```

OVMS can take 2–5 minutes to load the model. Wait for `"Server started"` in the logs.

## GPU Not Detected

```bash
sudo usermod -aG render $USER
# Log out and log back in, then restart services
make down && make up
```

## No Scenarios in UI

Ensure images are in `images/` and `configs/orders.json` has entries with matching `image_id`
values (filename without extension). See [Step 4: Prepare Test Data](./get-started.md#step-4-prepare-test-data) for details.
