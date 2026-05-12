#!/usr/bin/env bash
# Usage: scripts/package_release.sh v0.1.0 ["Release notes here"]
# Builds the plugin in Release mode, zips artifacts, and creates a GitHub release.
set -euo pipefail

VERSION="${1:-}"
NOTES="${2:-}"

if [[ -z "$VERSION" ]]; then
    echo "Usage: $0 <version> [\"release notes\"]"
    echo "  e.g. $0 v0.1.0 \"First public release\""
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLUGIN_DIR="$REPO_ROOT/plugin/AIMusicPlugin"
BUILD_DIR="$PLUGIN_DIR/build"
ARTEFACTS="$BUILD_DIR/AIMusicPlugin_artefacts/Release"
PRODUCT="Mirror Mirror"
OUT_DIR="$REPO_ROOT/dist"

echo ">>> Building $PRODUCT $VERSION (Release)"
cmake -S "$PLUGIN_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --config Release -j"${PLUGIN_JOBS:-8}"

mkdir -p "$OUT_DIR"

ASSETS=()

# VST3 (all platforms)
VST3="$ARTEFACTS/VST3/$PRODUCT.vst3"
if [[ -d "$VST3" ]]; then
    ZIP="$OUT_DIR/${PRODUCT// /-}-$VERSION-VST3-macOS.zip"
    rm -f "$ZIP"
    cd "$ARTEFACTS/VST3" && zip -qr "$ZIP" "$PRODUCT.vst3"
    ASSETS+=("$ZIP")
    echo ">>> Packaged VST3: $ZIP"
fi

# AU (macOS only)
AU="$ARTEFACTS/AU/$PRODUCT.component"
if [[ -d "$AU" ]]; then
    ZIP="$OUT_DIR/${PRODUCT// /-}-$VERSION-AU-macOS.zip"
    rm -f "$ZIP"
    cd "$ARTEFACTS/AU" && zip -qr "$ZIP" "$PRODUCT.component"
    ASSETS+=("$ZIP")
    echo ">>> Packaged AU: $ZIP"
fi

if [[ ${#ASSETS[@]} -eq 0 ]]; then
    echo "ERROR: No plugin artifacts found in $ARTEFACTS"
    exit 1
fi

cd "$REPO_ROOT"

echo ">>> Creating GitHub release $VERSION"
gh release create "$VERSION" \
    "${ASSETS[@]}" \
    --title "$PRODUCT $VERSION" \
    ${NOTES:+--notes "$NOTES"} \
    ${NOTES:-"--generate-notes"}

echo ">>> Done. Release $VERSION published."
