#!/bin/bash
set -e

# Align the in-container application user with the host user that owns the
# bind-mounted volumes. Without this, files written to ./results are owned by
# the container's UID (1000); on hosts where the invoking user is not uid 1000
# (e.g. EMT) the host user then cannot chmod/remove them, which breaks a
# second `make up`. See ITEP-93258.
APP_USER=dlstreamer
TARGET_UID="${HOST_UID:-}"
TARGET_GID="${HOST_GID:-}"

if [ "$(id -u)" = "0" ] && [ -n "$TARGET_UID" ] && [ -n "$TARGET_GID" ]; then
    CURRENT_UID="$(id -u "$APP_USER")"
    CURRENT_GID="$(id -g "$APP_USER")"

    # Never remap to root; that would defeat the privilege drop below.
    if [ "$TARGET_UID" = "0" ] || [ "$TARGET_GID" = "0" ]; then
        echo "entrypoint: refusing to remap $APP_USER to root; keeping ${CURRENT_UID}:${CURRENT_GID}" >&2
    elif [ "$TARGET_UID" != "$CURRENT_UID" ] || [ "$TARGET_GID" != "$CURRENT_GID" ]; then
        # -o permits a non-unique id, in case it collides with an existing entry.
        if [ "$TARGET_GID" != "$CURRENT_GID" ]; then
            groupmod -o -g "$TARGET_GID" "$APP_USER"
        fi
        if [ "$TARGET_UID" != "$CURRENT_UID" ]; then
            usermod -o -u "$TARGET_UID" "$APP_USER"
        fi
        echo "entrypoint: remapped $APP_USER ${CURRENT_UID}:${CURRENT_GID} -> ${TARGET_UID}:${TARGET_GID}"
        # Re-own files still carrying the old ids. Restricted to paths owned by
        # the previous uid/gid so large bind mounts are not rewritten wholesale.
        find /app -xdev \( -uid "$CURRENT_UID" -o -gid "$CURRENT_GID" \) \
            -exec chown -h "$TARGET_UID:$TARGET_GID" {} + 2>/dev/null || true
        if [ -n "$HOME" ] && [ -d "$HOME" ]; then
            chown -R "$TARGET_UID:$TARGET_GID" "$HOME" 2>/dev/null || true
        fi
    fi
fi

# Fix ownership of mounted volumes that Docker may create as root.
# Runs as root before dropping privileges.
if [ "$(id -u)" = "0" ] && [ -d "/app/results" ]; then
    chown -R "$APP_USER:$APP_USER" /app/results 2>/dev/null || true
fi

# Drop privileges and exec the CMD as the application user. Using the user name
# (not a numeric id) preserves supplementary groups such as `video`, which GPU
# access depends on.
if [ "$(id -u)" = "0" ]; then
    exec gosu "$APP_USER" "$@"
fi

exec "$@"
