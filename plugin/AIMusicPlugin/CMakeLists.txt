cmake_minimum_required(VERSION 3.22)
project(AIMusicPlugin VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory($ENV{HOME}/JUCE juce_build)

juce_add_plugin(AIMusicPlugin
    COMPANY_NAME            "SummerKrinsky"
    PLUGIN_MANUFACTURER_CODE Smkr
    PLUGIN_CODE             Aimp
    FORMATS                 VST3 AU
    PRODUCT_NAME            "Mirror Mirror"
    IS_SYNTH                TRUE
    NEEDS_MIDI_INPUT        FALSE
    NEEDS_MIDI_OUTPUT       TRUE
    IS_MIDI_EFFECT          FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    VST3_CATEGORIES         "Instrument" "Tools"
    AU_MAIN_TYPE            kAudioUnitType_MusicDevice
    HARDENED_RUNTIME_ENABLED FALSE
    APP_SANDBOX_ENABLED      FALSE
    COPY_PLUGIN_AFTER_BUILD  TRUE
)

target_sources(AIMusicPlugin PRIVATE
    Source/PluginProcessor.cpp
    Source/PluginEditor.cpp
    Source/PipelineClient.cpp
)

get_filename_component(AI_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

target_compile_definitions(AIMusicPlugin PUBLIC
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
)

target_compile_definitions(AIMusicPlugin PRIVATE
    AI_REPO_ROOT="${AI_REPO_ROOT}"
)

target_link_libraries(AIMusicPlugin PRIVATE
    juce::juce_audio_utils
    juce::juce_gui_basics
    juce::juce_audio_processors
    PUBLIC
    juce::juce_recommended_config_flags
    juce::juce_recommended_lto_flags
    juce::juce_recommended_warning_flags
)

# Post-build: patch Info.plist to replace resourceUsage with sandboxSafe,
# then remove the old copy and re-sign
set(COMPONENT_DIR "/Library/Audio/Plug-Ins/Components/Mirror Mirror.component")
set(BUILD_COMPONENT "${CMAKE_CURRENT_BINARY_DIR}/AIMusicPlugin_artefacts/Debug/AU/Mirror Mirror.component")

set(USER_COMPONENT_DIR "$ENV{HOME}/Library/Audio/Plug-Ins/Components/Mirror Mirror.component")

add_custom_command(TARGET AIMusicPlugin_AU POST_BUILD
    COMMAND python3 -c
        "import plistlib; path='${BUILD_COMPONENT}/Contents/Info.plist';\
f=open(path,'rb'); pl=plistlib.load(f); f.close();\
[c.update({'sandboxSafe':True}) or c.pop('resourceUsage',None) for c in pl.get('AudioComponents',[])];\
pl.setdefault('CFBundleSupportedPlatforms',['MacOSX']);\
pl.setdefault('LSMinimumSystemVersion','10.13');\
f=open(path,'wb'); plistlib.dump(pl,f); f.close(); print('Info.plist patched')"
    COMMAND codesign --force --sign - "${BUILD_COMPONENT}"
    COMMAND rm -rf "${USER_COMPONENT_DIR}"
    COMMAND cp -r "${BUILD_COMPONENT}" "$ENV{HOME}/Library/Audio/Plug-Ins/Components/"
    COMMENT "Patching, signing, and installing AU component to user Library"
    VERBATIM
)

# Post-build: install VST3 to user library (keeps in sync after every rebuild)
set(BUILD_VST3 "${CMAKE_CURRENT_BINARY_DIR}/AIMusicPlugin_artefacts/Debug/VST3/Mirror Mirror.vst3")
set(USER_VST3_DIR "$ENV{HOME}/Library/Audio/Plug-Ins/VST3")

add_custom_command(TARGET AIMusicPlugin_VST3 POST_BUILD
    COMMAND rm -rf "${USER_VST3_DIR}/Mirror Mirror.vst3"
    COMMAND cp -r "${BUILD_VST3}" "${USER_VST3_DIR}/"
    COMMENT "Installing VST3 to user Library"
    VERBATIM
)
