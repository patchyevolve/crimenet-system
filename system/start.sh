#!/usr/bin/env bash
#
# CrimeNet — one-shot stack startup.
#
#   ./start.sh              build (cached) + start + wait healthy + open/print the UI
#   ./start.sh --no-open    same, but never launch a browser
#   ./start.sh --no-build   skip `docker compose build` (reuse existing images)
#   ./start.sh down|restart|status|logs
#
# Cold-start hazards it deals with for you:
#   * host 9000/9001 owned by a neighbouring stack -> MinIO republished on
#     9100/9101 via docker-compose.override.yml (in-network URL unchanged)
#   * quay.io/minio/minio unfetchable -> local bitnamilegacy/minio substituted
#   * root-owned MinIO volume -> ownership repaired when storage crash-loops
#
# The override file is gitignored. It is regenerated when it carries our
# "start.sh" / "Local-only (gitignored)" marker; anything else is assumed
# hand-written and left alone.
#
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

API_URL="http://localhost:8000/api/meta"
UI_URL="http://localhost:8080"
API_CN="criminal_network_api"
FRONT_CN="criminal_network_frontend"
STORAGE_CN="criminal_network_storage"
OVERRIDE="docker-compose.override.yml"
FALLBACK_MINIO_IMAGE="bitnamilegacy/minio:2025.7.23"
START_TIMEOUT="${START_TIMEOUT:-300}"
NO_BUILD=0
NO_OPEN=0

log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m ok \033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m !  \033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mFAIL\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
CrimeNet stack control

  ./start.sh              build (cached) + start + wait until healthy + open the UI
  ./start.sh --no-open    never launch a browser
  ./start.sh --no-build   reuse existing images (skip docker compose build)
  ./start.sh status       compose ps + API health
  ./start.sh logs         follow container logs
  ./start.sh restart      restart the API after editing api/ or src/
  ./start.sh down         stop the stack (volumes are kept)

Frontend: http://localhost:8080   API: http://localhost:8000/api/meta
EOF
    exit "${1:-0}"
}

port_busy() {
    if command -v ss >/dev/null 2>&1; then
        ss -ltn 2>/dev/null | grep -qE "[:.]$1[[:space:]]"
    else
        if (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; then return 0; fi
        return 1
    fi
}

# True when something outside this stack holds the host port.
port_foreign() {
    port_busy "$1" || return 1
    local owners
    owners=$(docker ps --filter "publish=$1" --format '{{.Names}}' 2>/dev/null | grep '^criminal_network_' || true)
    [ -z "$owners" ]
}

minio_quay_available() {
    docker image inspect quay.io/minio/minio:latest >/dev/null 2>&1 && return 0
    docker manifest inspect quay.io/minio/minio:latest >/dev/null 2>&1 && return 0
    return 1
}

ensure_override() {
    if [ -f "$OVERRIDE" ] && ! grep -qE 'start\.sh|Local-only \(gitignored\)' "$OVERRIDE"; then
        warn "$OVERRIDE looks hand-written — leaving it untouched"
        return 0
    fi

    local image_line="" ports_block="" base="" alt=""

    if ! minio_quay_available; then
        if ! docker image inspect "$FALLBACK_MINIO_IMAGE" >/dev/null 2>&1; then
            log "Pulling $FALLBACK_MINIO_IMAGE (quay.io/minio unreachable) ..."
            docker pull "$FALLBACK_MINIO_IMAGE" >/dev/null || die "cannot fetch any MinIO image"
        fi
        image_line="    image: $FALLBACK_MINIO_IMAGE"$'\n'
    fi

    if port_foreign 9000 || port_foreign 9001; then
        local pair
        for pair in "9100 9101" "9200 9201" "9300 9301"; do
            read -r base alt <<<"$pair"
            if ! port_foreign "$base" && ! port_foreign "$alt"; then break; fi
            base=""
        done
        [ -n "$base" ] || die "no free host port pair for MinIO (9000/9100/9200/9300 all taken)"
        ports_block=$(printf '    ports: !override\n      - "%s:9000"\n      - "%s:9001"\n' "$base" "$alt")
    fi

    local body="$image_line$ports_block" desired
    desired="# Managed by start.sh — local-only (gitignored). Regenerated when stale."
    if [ -n "$body" ]; then
        desired+=$'\n''services:'$'\n''  minio:'$'\n'"$body"
    else
        desired+=$'\n''services: {}'
    fi
    desired="${desired%$'\n'}"   # printf re-adds the final newline below

    if [ -f "$OVERRIDE" ] && [ "$(cat "$OVERRIDE")" = "$desired" ]; then
        return 0
    fi
    printf '%s\n' "$desired" >"$OVERRIDE"
    log "wrote $OVERRIDE (minio image and/or host ports)"
}

container_state() { docker inspect --format '{{.State.Status}}' "$1" 2>/dev/null || echo absent; }

repair_minio_volume() {
    local img uid
    img=$(docker inspect --format '{{.Config.Image}}' "$STORAGE_CN" 2>/dev/null || true)
    img="${img:-$FALLBACK_MINIO_IMAGE}"
    uid=$(docker run --rm --entrypoint id "$img" -u 2>/dev/null || echo 1001)
    warn "MinIO storage is crash-looping — repairing volume ownership for uid $uid"
    docker compose stop minio >/dev/null 2>&1 || true
    docker compose run --rm --no-deps --user 0 --entrypoint chown minio -R "$uid:$uid" /data >/dev/null
    docker compose start minio >/dev/null
    ok "MinIO volume now writable by uid $uid"
}

wait_for_api() {
    local waited=0 state
    while [ "$waited" -lt "$START_TIMEOUT" ]; do
        if curl -sf -o /dev/null "$API_URL"; then return 0; fi
        state=$(container_state "$API_CN")
        if [ "$state" = "exited" ]; then
            docker compose logs --tail 40 "$API_CN" >&2 || true
            die "$API_CN exited — see logs above (./start.sh logs)"
        fi
        if [ "$waited" -gt 0 ] && [ $((waited % 15)) -eq 0 ]; then
            printf '   ... waiting for the API (%ss, container %s)\n' "$waited" "$state"
        fi
        sleep 3
        waited=$((waited + 3))
    done
    docker compose logs --tail 40 "$API_CN" >&2 || true
    die "API not healthy after ${START_TIMEOUT}s"
}

# nginx finishes forking its workers a beat after the container reports
# "Started", so the first curl can race it — poll instead of firing once.
wait_for_url() {
    local url=$1 timeout=${2:-60} waited=0
    while [ "$waited" -lt "$timeout" ]; do
        if curl -sf -o /dev/null "$url"; then return 0; fi
        sleep 2
        waited=$((waited + 2))
    done
    return 1
}

open_browser() {
    if [ "$NO_OPEN" = 1 ]; then return 0; fi
    if [ "$(uname)" = "Darwin" ] && command -v open >/dev/null 2>&1; then
        open "$UI_URL" >/dev/null 2>&1 &
    elif command -v xdg-open >/dev/null 2>&1 && { [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; }; then
        xdg-open "$UI_URL" >/dev/null 2>&1 &
    fi
    return 0
}

storage_port() {
    local p
    p=$(docker port "$STORAGE_CN" 9000/tcp 2>/dev/null | head -1 | sed 's/.*://' || true)
    echo "${p:-9000}"
}

cmd_up() {
    docker info >/dev/null 2>&1 || die "docker daemon is not running"
    docker compose version >/dev/null 2>&1 || die "docker compose plugin missing"

    ensure_override

    log "Starting the stack (postgres, neo4j, minio, api, frontend) ..."
    if [ "$NO_BUILD" = 1 ]; then
        docker compose up -d
    else
        docker compose up -d --build
    fi

    local restarts
    restarts=$(docker inspect --format '{{.RestartCount}}' "$STORAGE_CN" 2>/dev/null || echo 0)
    restarts="${restarts:-0}"
    if [ "${restarts:-0}" -gt 0 ] || { [ "$(container_state "$STORAGE_CN")" != "running" ] && [ "$(container_state "$STORAGE_CN")" != "created" ]; }; then
        repair_minio_volume
    fi

    log "Waiting for the API ..."
    wait_for_api
    ok "api      $API_URL"

    wait_for_url "$UI_URL/" 60 || {
        docker compose logs --tail 20 "$FRONT_CN" >&2 || true
        die "frontend not answering on :8080 (logs above; ./start.sh logs for more)"
    }
    ok "frontend $UI_URL"

    wait_for_url "$UI_URL/api/meta" 60 || die "nginx -> api proxy not answering on $UI_URL/api/meta"
    ok "proxy    $UI_URL/api/meta  (nginx -> api)"

    printf '\n\033[1m  CrimeNet is up.\033[0m\n\n'
    printf '   UI       \033[1m%s\033[0m\n' "$UI_URL"
    printf '   API      http://localhost:8000\n'
    printf '   MinIO    http://localhost:%s\n' "$(storage_port)"
    printf '   Postgres localhost:5432    Neo4j http://localhost:7474\n\n'
    printf '   Sign in: identity picker — any display name, choose a role\n'
    printf '   (INSPECTOR / SUPERVISOR / ADMIN / AUDIT_LOGGER), no password.\n'
    printf '   Logs: ./start.sh logs     Stop: ./start.sh down\n\n'

    open_browser
}

cmd="up"
for arg in "$@"; do
    case "$arg" in
        -h|--help|help) usage ;;
        --no-open)  NO_OPEN=1 ;;
        --no-build) NO_BUILD=1 ;;
        up|start|down|stop|restart|status|logs) cmd="$arg" ;;
        *) usage 1 ;;
    esac
done

case "$cmd" in
    up|start)   cmd_up ;;
    down|stop)  docker compose down; ok "stack stopped (volumes kept)" ;;
    restart)    docker compose restart api; ok "api restarted (picks up edited api/ and src/ files)" ;;
    status)     docker compose ps
                if curl -sf -o /dev/null "$API_URL"; then ok "api $API_URL"; else warn "api not answering"; fi ;;
    logs)       docker compose logs -f --tail 100 ;;
esac
