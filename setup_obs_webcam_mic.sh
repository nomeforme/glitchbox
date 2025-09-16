#!/usr/bin/env bash

set -euo pipefail

# Small helper to route the Trust webcam microphone into a virtual bus
# that OBS can capture via the monitor source.
#
# Usage:
#   ./setup_obs_webcam_mic.sh up   [SOURCE_NAME] [SINK_NAME]
#   ./setup_obs_webcam_mic.sh down [SOURCE_NAME] [SINK_NAME]
#
# Defaults:
#   SOURCE_NAME: auto (tries to find Trust webcam, then falls back to default source)
#   SINK_NAME:   obs_mic_bus
#
# After running `up`, select "Monitor of OBS_Mic_Bus" (device name: SINK_NAME.monitor)
# as your Audio Input in OBS.

ACTION="${1:-up}"
SOURCE_NAME_INPUT="${2:-auto}"
SINK_NAME="${3:-obs_mic_bus}"

die() { echo "[ERR] $*" >&2; exit 1; }

get_default_source_name() {
  pactl info | awk -F": " '/Default Source/ {print $2}'
}

auto_detect_trust_source() {
  # Try to find the Trust webcam source by its USB identifier first
  local name
  name=$(pactl list short sources | awk '/usb-SC_Trust_QHD_Webcam/ {print $2; exit}')
  if [[ -n "${name:-}" ]]; then
    echo "$name"
    return 0
  fi
  # Fallback to default source
  get_default_source_name
}

get_source_index_by_name() {
  local src_name="$1"
  pactl list short sources | awk -v n="$src_name" '$2==n {print $1; exit}'
}

get_sink_index_by_name() {
  local sink_name="$1"
  pactl list short sinks | awk -v n="$sink_name" '$2==n {print $1; exit}'
}

get_module_id_for_null_sink() {
  local sink_name="$1"
  pactl list short modules | awk -v n="$sink_name" '$2=="module-null-sink" && $0 ~ ("sink_name=" n) {print $1; exit}'
}

get_module_id_for_loopback() {
  local src_name="$1"; local sink_name="$2"
  pactl list short modules | awk -v s="$src_name" -v k="$sink_name" '$2=="module-loopback" && $0 ~ ("source=" s) && $0 ~ ("sink=" k) {print $1; exit}'
}

ensure_null_sink() {
  local sink_name="$1"; local desc="${2:-OBS_Mic_Bus}"
  local sink_idx; sink_idx=$(get_sink_index_by_name "$sink_name" || true)
  if [[ -n "${sink_idx:-}" ]]; then
    echo "[OK] Null sink '$sink_name' already exists (index $sink_idx)"
    return 0
  fi
  local mod_id
  mod_id=$(pactl load-module module-null-sink sink_name="$sink_name" sink_properties=device.description="$desc")
  echo "[OK] Created null sink '$sink_name' (module $mod_id)"
}

ensure_loopback() {
  local src_name="$1"; local sink_name="$2"
  local existing; existing=$(get_module_id_for_loopback "$src_name" "$sink_name" || true)
  if [[ -n "${existing:-}" ]]; then
    echo "[OK] Loopback already exists (module $existing)"
    return 0
  fi
  local mod_id
  mod_id=$(pactl load-module module-loopback source="$src_name" sink="$sink_name" latency_msec=1)
  echo "[OK] Created loopback from '$src_name' -> '$sink_name' (module $mod_id)"
}

unload_loopback() {
  local src_name="$1"; local sink_name="$2"
  local mod_id; mod_id=$(get_module_id_for_loopback "$src_name" "$sink_name" || true)
  if [[ -n "${mod_id:-}" ]]; then
    pactl unload-module "$mod_id" || true
    echo "[OK] Unloaded loopback module $mod_id"
  else
    echo "[OK] No matching loopback to unload"
  fi
}

unload_null_sink() {
  local sink_name="$1"
  local mod_id; mod_id=$(get_module_id_for_null_sink "$sink_name" || true)
  if [[ -n "${mod_id:-}" ]]; then
    pactl unload-module "$mod_id" || true
    echo "[OK] Unloaded null sink module $mod_id"
  else
    echo "[OK] No null sink module to unload for '$sink_name'"
  fi
}

set_levels() {
  local src_name="$1"; local sink_name="$2"
  local src_idx; src_idx=$(get_source_index_by_name "$src_name" || true)
  if [[ -n "${src_idx:-}" ]]; then
    pactl set-source-mute "$src_idx" 0 || true
    pactl set-source-volume "$src_idx" 0dB || true
  fi
  local mon_name="${sink_name}.monitor"
  local mon_idx; mon_idx=$(get_source_index_by_name "$mon_name" || true)
  if [[ -n "${mon_idx:-}" ]]; then
    pactl set-source-mute "$mon_idx" 0 || true
    pactl set-source-volume "$mon_idx" 0dB || true
  fi
}

main_up() {
  local src_name
  if [[ "$SOURCE_NAME_INPUT" == "auto" ]]; then
    src_name=$(auto_detect_trust_source)
  else
    src_name="$SOURCE_NAME_INPUT"
  fi
  [[ -n "$src_name" ]] || die "Could not determine source name. Specify explicitly as second argument."

  echo "[INFO] Using source: $src_name"
  echo "[INFO] Creating/ensuring null sink: $SINK_NAME"
  ensure_null_sink "$SINK_NAME" "OBS_Mic_Bus"

  echo "[INFO] Creating/ensuring loopback from '$src_name' to '$SINK_NAME'"
  ensure_loopback "$src_name" "$SINK_NAME"

  echo "[INFO] Unmuting and setting levels to 0dB"
  set_levels "$src_name" "$SINK_NAME"

  echo
  echo "[DONE] OBS can capture: 'Monitor of OBS_Mic_Bus' (device: ${SINK_NAME}.monitor)"
}

main_down() {
  local src_name
  if [[ "$SOURCE_NAME_INPUT" == "auto" ]]; then
    src_name=$(auto_detect_trust_source)
  else
    src_name="$SOURCE_NAME_INPUT"
  fi
  [[ -n "$src_name" ]] || src_name=""

  if [[ -n "$src_name" ]]; then
    echo "[INFO] Unloading loopback for source '$src_name' and sink '$SINK_NAME'"
    unload_loopback "$src_name" "$SINK_NAME"
  else
    echo "[WARN] Could not auto-detect source; skipping loopback unload"
  fi
  echo "[INFO] Unloading null sink '$SINK_NAME'"
  unload_null_sink "$SINK_NAME"
  echo "[DONE] Torn down OBS mic bus"
}

case "$ACTION" in
  up)
    main_up
    ;;
  down)
    main_down
    ;;
  *)
    echo "Usage: $0 {up|down} [SOURCE_NAME|auto] [SINK_NAME|obs_mic_bus]" >&2
    exit 2
    ;;
esac

