#!/usr/bin/env bash
# Mirror Mirror — one-shot installer
# Downloads everything needed, sets up the Python environment, and builds + installs the plugin.
#
# Usage:
#   bash install.sh
#   bash install.sh --dir ~/my-mirror-mirror   # custom install location
#
set -euo pipefail

# ── colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}>>>${NC} $*"; }
ok()      { echo -e "${GREEN}OK${NC}  $*"; }
warn()    { echo -e "${YELLOW}WARN${NC} $*"; }
die()     { echo -e "${RED}ERROR${NC} $*" >&2; exit 1; }

# ── args ──────────────────────────────────────────────────────────────────────
INSTALL_DIR="$HOME/mirror-mirror"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dir) INSTALL_DIR="$2"; shift 2 ;;
        *)     die "Unknown argument: $1" ;;
    esac
done

JUCE_VERSION="8.0.3"
JUCE_DIR="$HOME/JUCE"
REPO_URL="https://github.com/skrinsky/mirror-mirror.git"

# ── OS detection ──────────────────────────────────────────────────────────────
OS="$(uname -s)"
ARCH="$(uname -m)"

if [[ "$OS" == "Darwin" ]]; then
    PLATFORM="macos"
elif [[ "$OS" == "Linux" ]]; then
    PLATFORM="linux"
else
    die "Windows is not yet supported by this installer. Please follow the manual build instructions in the README."
fi

echo ""
echo "  Mirror Mirror Installer"
echo "  ========================"
echo "  Platform : $OS $ARCH"
echo "  Install  : $INSTALL_DIR"
echo "  JUCE     : $JUCE_DIR"
echo ""

# ── macOS: Xcode CLT ──────────────────────────────────────────────────────────
if [[ "$PLATFORM" == "macos" ]]; then
    if ! xcode-select -p &>/dev/null; then
        info "Installing Xcode Command Line Tools (you may see a popup)..."
        xcode-select --install
        echo "  Re-run this installer after the Xcode CLT installation completes."
        exit 0
    fi
    ok "Xcode Command Line Tools"
fi

# ── Homebrew (macOS) ──────────────────────────────────────────────────────────
if [[ "$PLATFORM" == "macos" ]]; then
    if ! command -v brew &>/dev/null; then
        info "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || /usr/local/bin/brew shellenv)"
    fi
    ok "Homebrew"
fi

# ── cmake ─────────────────────────────────────────────────────────────────────
if ! command -v cmake &>/dev/null; then
    info "Installing cmake..."
    if [[ "$PLATFORM" == "macos" ]]; then
        brew install cmake
    else
        sudo apt-get install -y cmake || sudo dnf install -y cmake || die "Could not install cmake — please install it manually."
    fi
fi
ok "cmake $(cmake --version | head -1 | awk '{print $3}')"

# ── git ───────────────────────────────────────────────────────────────────────
if ! command -v git &>/dev/null; then
    die "git not found. Install git and re-run."
fi
ok "git $(git --version | awk '{print $3}')"

# ── Python 3.10 ───────────────────────────────────────────────────────────────
PYTHON_BIN=""
for candidate in python3.10 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        ver="$("$candidate" -c 'import sys; print(sys.version_info[:2])')"
        if [[ "$ver" == "(3, 10)" ]]; then
            PYTHON_BIN="$candidate"
            break
        fi
    fi
done

if [[ -z "$PYTHON_BIN" ]]; then
    info "Python 3.10 not found — installing via Homebrew..."
    if [[ "$PLATFORM" == "macos" ]]; then
        brew install python@3.10
        PYTHON_BIN="$(brew --prefix)/bin/python3.10"
    else
        sudo apt-get install -y python3.10 python3.10-venv || \
            die "Could not install Python 3.10 — please install it manually."
        PYTHON_BIN="python3.10"
    fi
fi
ok "Python $($PYTHON_BIN --version)"

# ── JUCE ──────────────────────────────────────────────────────────────────────
if [[ ! -d "$JUCE_DIR" ]]; then
    info "Downloading JUCE $JUCE_VERSION to $JUCE_DIR..."
    TMP_ZIP="$(mktemp /tmp/juce-XXXXXX.zip)"
    if [[ "$PLATFORM" == "macos" ]]; then
        JUCE_URL="https://github.com/juce-framework/JUCE/releases/download/${JUCE_VERSION}/juce-${JUCE_VERSION}-osx.zip"
    else
        JUCE_URL="https://github.com/juce-framework/JUCE/releases/download/${JUCE_VERSION}/juce-${JUCE_VERSION}-linux.zip"
    fi
    curl -fsSL "$JUCE_URL" -o "$TMP_ZIP"
    TMP_DIR="$(mktemp -d)"
    unzip -q "$TMP_ZIP" -d "$TMP_DIR"
    mv "$TMP_DIR/JUCE" "$JUCE_DIR"
    rm -f "$TMP_ZIP"
    rm -rf "$TMP_DIR"
fi
ok "JUCE $JUCE_VERSION at $JUCE_DIR"

# ── Clone repo ────────────────────────────────────────────────────────────────
if [[ -d "$INSTALL_DIR/.git" ]]; then
    info "Repo already exists at $INSTALL_DIR — pulling latest..."
    git -C "$INSTALL_DIR" pull --ff-only
    git -C "$INSTALL_DIR" submodule update --init --recursive
else
    info "Cloning Mirror Mirror into $INSTALL_DIR..."
    git clone --recurse-submodules "$REPO_URL" "$INSTALL_DIR"
fi
ok "Repo at $INSTALL_DIR"

# ── Python venv + dependencies ────────────────────────────────────────────────
info "Setting up Python environment (this may take a few minutes)..."
PYTHON_BIN="$PYTHON_BIN" bash "$INSTALL_DIR/scripts/setup_venv.sh"
ok "Python environment ready"

# ── Build + install plugin ────────────────────────────────────────────────────
info "Building plugin (Release)..."
PLUGIN_DIR="$INSTALL_DIR/plugin/AIMusicPlugin"
BUILD_DIR="$PLUGIN_DIR/build"
cmake -S "$PLUGIN_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --config Release -j"$(sysctl -n hw.logicalcpu 2>/dev/null || nproc 2>/dev/null || echo 4)"
ok "Plugin built and installed"

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Mirror Mirror installed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
if [[ "$PLATFORM" == "macos" ]]; then
    echo "  Plugin installed to:"
    echo "    ~/Library/Audio/Plug-Ins/VST3/Mirror Mirror.vst3"
    echo "    ~/Library/Audio/Plug-Ins/Components/Mirror Mirror.component"
fi
echo ""
echo "  Repo location: $INSTALL_DIR"
echo ""
echo "  Next steps:"
echo "    1. Open your DAW and scan for new plugins"
echo "    2. Add Mirror Mirror to a MIDI track"
echo "    3. Drop audio files into $INSTALL_DIR/data/raw/"
echo "    4. Hit Process in the plugin to begin"
echo ""
