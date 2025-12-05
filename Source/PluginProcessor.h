#pragma once

#include <juce_dsp/juce_dsp.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include "AudioAnalysis.h"
#include "AudioRetarget.h"
#include <vector>

// Forward-declare the Handle struct
struct Handle;

// Shared constraint point structure for interactive retargeting
struct ConstraintPoint
{
    int id;
    float sourceTime;  // Y-axis - time in original audio
    float targetTime;  // X-axis - time in retargeted audio
};

class DynamicMusicVstAudioProcessor  : public juce::AudioProcessor,
                                       public juce::ChangeListener,
                                       public juce::ChangeBroadcaster
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
    bool isPlaying() const;
    double getCurrentPlaybackPosition() const;
    void setPlaybackPosition(double newPositionSecs);
    void setPlaybackPositionFromSource(double sourcePositionSecs);
    bool isInSourcePlaybackMode() const { return isPlayingFromSource.load(); }
    
    // Scrubbing Controls
    void startScrubbing();
    void stopScrubbing();
    void triggerScrubSnippet(double scrubPositionSecs);
    void triggerScrubSnippetFromSource(double scrubPositionSecs);
    
    // Getters for UI
    juce::AudioThumbnail& getAudioThumbnail() { return audioThumbnail; }
    double getCurrentPositionSecs() const;
    double getTotalLengthSecs() const;
    double getOriginalTotalLengthSecs() const;
    double getFileSampleRate() const { return fileSampleRate; }

    // Cursor synchronization methods
    double getSourceTimeAt(double targetTime) const;
    std::vector<double> getTargetTimesAt(double sourceTime) const;

    void retargetWithHandles(const std::vector<Handle>& handles);

    const juce::AudioBuffer<float>& getAudioBuffer() const { return sourceAudioBuffer; }
    const juce::AudioBuffer<float>& getRetargetedAudioBuffer() const { return retargetedAudioBuffer; }

    enum class AnalysisState
    {
        Ready,
        AnalysisNeeded,
        Analyzing,
        AnalysisComplete
    };
    std::atomic<AnalysisState> analysisState { AnalysisState::Ready };
    
    enum class RetargetState
    {
        Ready,
        RetargetingNeeded,
        Retargeting,
        RetargetingComplete
    };
    std::atomic<RetargetState> retargetState { RetargetState::Ready };

    juce::AudioBuffer<float> sourceAudioBuffer;
    float estimatedBPM { 120.0f };
    std::vector<double> beatTimestamps;
    std::vector<std::vector<float>> mfccs;
    std::vector<std::vector<float>> similarityMatrix;
    std::vector<std::vector<float>> tempogram;
    std::vector<float> globalAcf;
    std::vector<float> onsetEnvelope;
    std::vector<int> retargetedBeatPath;
    juce::AudioBuffer<float> retargetedAudioBuffer;
    std::atomic<bool> isRetargeted { false };
    
    // Constraint management - the processor owns all constraint state
    std::vector<ConstraintPoint> userConstraints;
    int nextConstraintId { 0 };
    
    // Constraint manipulation methods - called from UI
    int addConstraint(float sourceTime, float targetTime);
    void moveConstraint(int id, float newSourceTime, float newTargetTime);
    void removeConstraint(int id);
    void initializeDefaultConstraints();
    void performFullRetarget();
    
    // Thread-safe constraint access
    juce::CriticalSection constraintLock;

    // Background Retargeting
    class RetargetThread : public juce::Thread
    {
    public:
        RetargetThread(DynamicMusicVstAudioProcessor& p) 
            : Thread("RetargetThread"), processor(p) {}
        
        void run() override
        {
            while (!threadShouldExit())
            {
                if (processor.retargetNeeded.load())
                {
                    processor.retargetNeeded.store(false);
                    // Perform retargeting
                    // We need a lock to copy constraints safely, then run algo
                    processor.performBackgroundRetarget();
                }
                wait(15); // Check every ~60fps
            }
        }
    private:
        DynamicMusicVstAudioProcessor& processor;
    };
    
    std::unique_ptr<RetargetThread> retargetThread;
    std::atomic<bool> retargetNeeded { false };
    std::atomic<int> lastChangedConstraintId { -1 };
    std::atomic<bool> isFullRetargetNeeded { false };
    
    void performBackgroundRetarget(); // The actual worker method
    
    // Get sorted constraints for UI display
    const std::vector<ConstraintPoint>& getConstraints() const { return userConstraints; }
    
    // Visual feedback structures
    struct CutInfo
    {
        double targetTime;      // Where the cut appears in the timeline
        double sourceTimeFrom;  // Source time before the cut
        double sourceTimeTo;    // Source time after the cut
        float quality;          // 0.0 (bad) to 1.0 (good) - based on similarity
    };
    
    // Get regions of source audio that are not used in the retargeting
    std::vector<std::pair<double, double>> getUnusedSourceRegions() const;
    
    // Get information about all cuts/discontinuities in the retargeted audio
    std::vector<CutInfo> getCuts() const;

private:
    void createRetargetedAudio(const std::vector<int>& path);
    void performSegmentedRetarget(int changedConstraintId);
    void performRetarget(bool isFullRetarget, int changedConstraintId, const std::vector<ConstraintPoint>& activeConstraints);
    
    // --- Scrubbing State ---
    juce::AudioBuffer<float> scrubSnippetBuffer;
    std::atomic<int> scrubSnippetReadPos { 0 };
    std::atomic<bool> isScrubbing { false };
    juce::CriticalSection snippetLock;
    
    // --- Source Playback State (for gray areas) ---
    std::atomic<bool> isPlayingFromSource { false };
    std::atomic<double> sourcePlaybackPosition { 0.0 };
    juce::CriticalSection sourcePlaybackLock;

    // --- Host settings ---
    int hostSamplesPerBlock { 512 };
    double hostSampleRate { 44100.0 };

    double fileSampleRate = 44100.0; // Default, will be updated on file load
    float crossfadeMs { 20.0f };
    float currentSecondsPerBeat { 0.5f }; // Store this for mapping
    
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
    AudioRetargeter audioRetargeter;
    juce::AudioThumbnailCache thumbnailCache { 5 };
    juce::AudioThumbnail audioThumbnail;

    juce::AudioFormatManager formatManager;
    std::unique_ptr<juce::AudioFormatReaderSource> fileReaderSource;
    juce::AudioTransportSource fileTransportSource;
    std::unique_ptr<juce::MemoryAudioSource> retargetedMemorySource;
    juce::AudioTransportSource retargetedTransportSource;
    bool isPlayingFile {false};
    juce::CriticalSection transportSourceLock;

    // --- Retargeted Playback State ---
    std::atomic<double> currentPlaybackPositionSecs { 0.0 };
    int currentRetargetBeatIndex = 0;
    int samplesIntoCurrentBeat = 0;


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
