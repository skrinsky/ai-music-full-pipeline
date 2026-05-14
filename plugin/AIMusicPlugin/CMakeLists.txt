cmake_minimum_required(VERSION 3.22)
project(AIMusicPlugin VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)

# Shared CPM cache across all projects that pin the same JUCE tag.
# Override by exporting CPM_SOURCE_CACHE before invoking cmake.
if(NOT DEFINED ENV{CPM_SOURCE_CACHE})
    if(WIN32)
        set(ENV{CPM_SOURCE_CACHE} "$ENV{LOCALAPPDATA}/CPM")
    else()
        set(ENV{CPM_SOURCE_CACHE} "$ENV{HOME}/.cache/CPM")
    endif()
endif()

include(${CMAKE_CURRENT_LIST_DIR}/cmake/CPM.cmake)

CPMAddPackage(
    NAME JUCE
    GITHUB_REPOSITORY juce-framework/JUCE
    GIT_TAG 8.0.3
)

# AU is macOS-only; VST3 is cross-platform
if(APPLE)
    set(PLUGIN_FORMATS VST3 AU)
else()
    set(PLUGIN_FORMATS VST3)
endif()

juce_add_plugin(AIMusicPlugin
    COMPANY_NAME            "SummerKrinsky"
    PLUGIN_MANUFACTURER_CODE Smkr
    PLUGIN_CODE             Aimp
    FORMATS                 ${PLUGIN_FORMATS}
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

# Platform-specific VST3 install location
if(APPLE)
    set(USER_VST3_DIR "$ENV{HOME}/Library/Audio/Plug-Ins/VST3")
elseif(WIN32)
    set(USER_VST3_DIR "$ENV{LOCALAPPDATA}/Programs/Common/VST3")
else()  # Linux
    set(USER_VST3_DIR "$ENV{HOME}/.vst3")
endif()

set(BUILD_VST3 "${CMAKE_CURRENT_BINARY_DIR}/AIMusicPlugin_artefacts/$<CONFIG>/VST3/Mirror Mirror.vst3")

add_custom_command(TARGET AIMusicPlugin_VST3 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${USER_VST3_DIR}/Mirror Mirror.vst3"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${BUILD_VST3}" "${USER_VST3_DIR}/Mirror Mirror.vst3"
    COMMENT "Installing VST3 to ${USER_VST3_DIR}"
    VERBATIM
)

# AU: macOS only — patch Info.plist, re-sign, install to user Library
if(APPLE)
    set(BUILD_COMPONENT "${CMAKE_CURRENT_BINARY_DIR}/AIMusicPlugin_artefacts/$<CONFIG>/AU/Mirror Mirror.component")
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
        COMMAND ${CMAKE_COMMAND} -E remove_directory "${USER_COMPONENT_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${BUILD_COMPONENT}" "${USER_COMPONENT_DIR}"
        COMMENT "Patching, signing, and installing AU component to user Library"
        VERBATIM
    )
endif()
