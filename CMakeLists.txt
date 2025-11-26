cmake_minimum_required(VERSION 3.15)
project(DynamicMusicVst VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(JUCE)

# 1. Create the plugin target using the main JUCE command
juce_add_plugin(DynamicMusicVst
    COMPANY_NAME "MyCompany"
    PLUGIN_NAME "Dynamic Music VST"
    PLUGIN_VERSION "1.0.0"
    PLUGIN_DESC "Smart music retargeting"
    PLUGIN_MANUFACTURER_CODE "MyCo"
    PLUGIN_CODE "Dmvs"
    PLUGIN_VST3_CATEGORY "Fx"
    FORMATS VST3 Standalone)

# 2. Add our source files to the target that was just created
target_sources(DynamicMusicVst PRIVATE
    Source/PluginProcessor.cpp
    Source/PluginProcessor.h
    Source/PluginEditor.cpp
    Source/PluginEditor.h
    Source/AudioAnalysis.cpp
    Source/AudioAnalysis.h
    Source/SimilarityMatrixComponent.cpp
    Source/SimilarityMatrixComponent.h)

# 3. Add any necessary compile definitions to the target
target_compile_definitions(DynamicMusicVst PRIVATE
    JUCE_FORCE_USE_LEGACY_PARAM_IDS=1
    JUCE_LOG_TO_STDERR=1)

# 4. Link the target against the JUCE modules it needs
target_link_libraries(DynamicMusicVst PRIVATE
    juce::juce_audio_basics
    juce::juce_audio_devices
    juce::juce_audio_formats
    juce::juce_audio_plugin_client
    juce::juce_audio_processors
    juce::juce_audio_utils
    juce::juce_core
    juce::juce_data_structures
    juce::juce_dsp
    juce::juce_events
    juce::juce_graphics
    juce::juce_gui_basics
    juce::juce_gui_extra)
