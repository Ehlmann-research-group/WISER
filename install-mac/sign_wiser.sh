#!/usr/bin/env bash
# Sign every executable object in a WISER .app bundle.
#
# Usage: sign_wiser.sh [path/to/WISER.app]
#
# The signing identity comes from $AD_CODESIGN_KEY_NAME. When that is unset, Secret.sh is
# sourced for it (the local developer path); CI sets the variable directly.
set -euo pipefail

APP="${1:-dist/WISER.app}"

if [ -z "${AD_CODESIGN_KEY_NAME:-}" ]; then
  SECRETS="$(dirname "${BASH_SOURCE[0]}")/../Secret.sh"
  if [ ! -f "$SECRETS" ]; then
    echo "ERROR: AD_CODESIGN_KEY_NAME is unset and $SECRETS does not exist." >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$SECRETS"
fi

if [ -z "${AD_CODESIGN_KEY_NAME:-}" ]; then
  echo "ERROR: AD_CODESIGN_KEY_NAME is empty." >&2
  exit 1
fi

if [ ! -d "$APP" ]; then
  echo "ERROR: No app bundle at $APP" >&2
  exit 1
fi

IDENT="${AD_CODESIGN_KEY_NAME}"
ENTITLEMENTS="$(dirname "${BASH_SOURCE[0]}")/entitlements.plist"

echo "Signing $APP with identity: $IDENT"

echo "-- Signing leaf binaries (dylib, so, exec)..."
find "$APP/Contents" -type f -print0 | xargs -0 file | grep 'Mach-O' | cut -d: -f1 |
while read -r BIN; do
  codesign --force --options runtime --timestamp --sign "$IDENT" "$BIN"
done

echo "-- Signing nested bundles (Frameworks, plug-ins)..."
find "$APP/Contents" -type d \( -name '*.framework' -o -name '*.bundle' \) |
while read -r BND; do
  codesign --force --options runtime --timestamp --sign "$IDENT" "$BND"
done

echo "-- Signing top-level app..."
codesign --force --options runtime --timestamp \
         --entitlements "$ENTITLEMENTS" \
         --sign "$IDENT" "$APP"

echo "-- Verifying..."
codesign --verify --strict --deep --verbose=2 "$APP"
echo "Signed: $APP"
