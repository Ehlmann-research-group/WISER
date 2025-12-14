#!/usr/bin/env bash
# Sign every executable object in dist/${APP_NAME}.app
set -euo pipefail

APP_NAME="WISER"
APP_VERSION="$(python src/wiser/version.py)"

# load secrets
source "$(dirname "${BASH_SOURCE[0]}")/../Secret.sh"

echo "Signing $APP_VERSION with key $AD_CODESIGN_KEY_NAME"

# now you can reference $AD_CODESIGN_KEY_NAME, $AD_USERNAME, etc.
echo "-- Signing with key: $AD_CODESIGN_KEY_NAME"

APP="dist/${APP_NAME}.app"
IDENT="${AD_CODESIGN_KEY_NAME}"
ENTITLEMENTS="install-mac/entitlements.plist"

echo "▶︎ Signing leaf binaries (dylib, so, exec)…"
find "$APP/Contents" -type f -print0 | xargs -0 file | grep 'Mach-O' | cut -d: -f1 |
while read -r BIN; do
  if file "$BIN" | grep -q 'Mach-O'; then
    codesign --force --options runtime --timestamp \
             --sign "$IDENT" "$BIN"
  fi
done

echo "▶︎ Signing nested bundles (Frameworks, plug-ins)…"
find "$APP/Contents" -type d \( -name '*.framework' -o -name '*.bundle' \) |
while read -r BND; do
  codesign --force --options runtime --timestamp \
           --sign "$IDENT" "$BND"
done

echo "▶︎ Signing top-level app…"
codesign --force --options runtime --timestamp \
         --entitlements "$ENTITLEMENTS" \
         --sign "$IDENT" "$APP"

echo "▶︎ Verifying…"
codesign --verify --strict --deep --verbose=2 "$APP"
echo "✅  All done"
