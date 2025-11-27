#pragma once

#include <juce_dsp/juce_dsp.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include "AudioAnalysis.h"

class DynamicMusicVstAudioProcessor  : public juce::AudioProcessor,
                                       public juce::ChangeListener
{
public:
    DynamicMusicVstAudioProcessor();
    ~DynamicMusicVstAudioProcessor() override;

    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

   #ifndef JucePlugin_PreferredChannelConfigurations
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
   #endif

    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const juce::String getProgramName (int index) override;
    void changeProgramName (int index, const juce::String& newName) override;

    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;
    void changeListenerCallback (juce::ChangeBroadcaster* source) override;

    juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();
    void loadAudioFile(const juce::File& file);
    
    // Playback control
    void startPlayback();
    void stopPlayback();
    void setPlaybackPosition(double newPositionSecs);
    
    // Getters for UI
    juce::AudioThumbnail& getAudioThumbnail() { return audioThumbnail; }
    bool isPlaying() const;
    double getCurrentPositionSecs() const;
    double getTotalLengthSecs() const;

    enum class AnalysisState
    {
        Ready,
        AnalysisNeeded,
        Analyzing,
        AnalysisComplete
    };
    std::atomic<AnalysisState> analysisState { AnalysisState::Ready };
    float estimatedBPM { 120.0f };
    std::vector<double> beatTimestamps;
    std::vector<std::vector<float>> mfccs;
    std::vector<std::vector<float>> similarityMatrix;
    std::vector<std::vector<float>> tempogram;
    std::vector<float> globalAcf;

private:
    // --- Metronome Click Synthesis ---
    double clickFrequency { 1000.0 };
    double clickDurationSecs { 0.05 };
    int clickSamplesRemaining = 0;
    double clickAngle = 0.0;
    double clickAngleDelta = 0.0;
    int nextBeatToPlay = 0;
    juce::CriticalSection clickLock;

    juce::AudioProcessorValueTreeState parameters;
    
    AudioAnalysis audioAnalyser;
    juce::AudioThumbnailCache thumbnailCache { 5 };
    juce::AudioThumbnail audioThumbnail;

    juce::AudioBuffer<float> sourceAudioBuffer;
    juce::AudioFormatManager formatManager;
    std::unique_ptr<juce::AudioFormatReaderSource> fileReaderSource;
    juce::AudioTransportSource fileTransportSource;
    bool isPlayingFile {false};
    juce::CriticalSection transportSourceLock;

    // This will hold our "remix map"
    // For now, it's a simple vector of sample ranges from the source buffer
    struct AudioSegment {
        int startSample;
        int numSamples;
    };
    std::vector<AudioSegment> remixMap;
    int remixPlaybackPosition = 0;

    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DynamicMusicVstAudioProcessor)
};
