#pragma once

#include "PluginProcessor.h"
#include "TimelineComponent.h"
#include "SourceOverviewComponent.h"
#include <juce_gui_basics/juce_gui_basics.h>
#include <vector>

//==============================================================================
struct Handle
                {
    int id;
    double sourceTime;      // Time in the source audio (blue view)
    double destinationTime; // Time in the manipulated timeline (orange view)
};

class DynamicMusicVstAudioProcessorEditor  : public juce::AudioProcessorEditor,
                                           public juce::Timer,
                                           private juce::ChangeListener
{
public:
    explicit DynamicMusicVstAudioProcessorEditor (DynamicMusicVstAudioProcessor&, juce::AudioProcessorValueTreeState&);
    ~DynamicMusicVstAudioProcessorEditor() override;

    void paint (juce::Graphics&) override;
    void paintOverChildren(juce::Graphics& g) override;
    void resized() override;
    void timerCallback() override;

    bool keyPressed(const juce::KeyPress& key) override;
    
    void addHandle(double sourceTime, double destinationTime);
    void moveHandle(int handleId, double newSourceTime, double newDestinationTime);
    void deleteHandle(int handleId);

private:
    void openAudioFile();
    void changeListenerCallback (juce::ChangeBroadcaster* source) override;

    DynamicMusicVstAudioProcessor& audioProcessor;
    juce::AudioProcessorValueTreeState& valueTreeState;
    
    SourceOverviewComponent sourceOverviewComponent;
    TimelineComponent timelineComponent;

    // UI Controls
    juce::TextButton openFileButton;
    juce::TextButton playButton;
    juce::TextButton stopButton;
    juce::Label statusLabel;

    std::unique_ptr<juce::FileChooser> fileChooser;

    std::vector<Handle> handles;
    int nextHandleId { 0 };
    int hoveredHandleId { -1 };
    bool isScrubbing { false };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DynamicMusicVstAudioProcessorEditor)
};
