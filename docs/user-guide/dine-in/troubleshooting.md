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

If the container fails with a GPU permission error or an empty `group_add` entry (or a GPU permission error on a host whose render group ID is not the default), run `make up` so the host's group IDs are saved to `.env`, or export them before using `docker compose` directly:

```bash
export RENDER_GID=$(getent group render | cut -d: -f3) VIDEO_GID=$(getent group video | cut -d: -f3)
```

Otherwise make sure your user is in the `render` group:

```bash
sudo usermod -aG render $USER
# Log out and log back in, then restart services
make down && make up
```

## No Scenarios in UI

Ensure images are in `images/` and `configs/orders.json` has entries with matching `image_id`
values (filename without extension). See [Step 4: Prepare Test Data](./get-started.md#step-4-prepare-test-data) for details.
