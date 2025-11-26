#!/bin/zsh

# This script automates the build, installation, and code-signing process for the DynamicMusicVst plugin.

# --- Configuration ---
PROJECT_ROOT="/Users/sarelduanis/Projects/dynamic_music_vst"
BUILD_DIR="${PROJECT_ROOT}/build"
VST3_SRC_PATH="${BUILD_DIR}/DynamicMusicVst_artefacts/VST3/DynamicMusicVst.vst3"
VST3_DEST_PATH="/Library/Audio/Plug-Ins/VST3/"

# --- Main Script ---

echo " "
echo ">>> File change detected, starting plugin update..."

# 1. Navigate to the build directory
cd "${BUILD_DIR}" || exit

# 2. Build the plugin
echo ">>> [1/3] Building plugin..."
if ! cmake --build . --config Release; then
    echo " "
    echo ">>> BUILD FAILED. Please check the compiler errors above."
    exit 1
fi
echo ">>> Build successful."

# 3. Copy the new plugin to the system folder
echo ">>> [2/3] Installing plugin..."
sudo cp -R "${VST3_SRC_PATH}" "${VST3_DEST_PATH}"
echo ">>> Installation successful."

# 4. Clean attributes and re-sign the plugin to satisfy macOS Gatekeeper
echo ">>> [3/3] Signing plugin..."
sudo xattr -cr "${VST3_DEST_PATH}/DynamicMusicVst.vst3"
sudo codesign --force --deep --sign - "${VST3_DEST_PATH}/DynamicMusicVst.vst3"
echo ">>> Signing successful."
echo " "
echo "✅ All done! You can now rescan for plugins in your DAW."
echo " "
