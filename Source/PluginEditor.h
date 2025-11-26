#pragma once

#include "PluginProcessor.h"
#include "SimilarityMatrixComponent.h"
#include <juce_gui_basics/juce_gui_basics.h>

//==============================================================================
class WaveformDisplay : public juce::Component,
                        private juce::ChangeListener
{
public:
    WaveformDisplay(DynamicMusicVstAudioProcessor& p) : audioProcessor(p)
    {
        audioProcessor.getAudioThumbnail().addChangeListener(this);
    }

    ~WaveformDisplay() override
    {
        audioProcessor.getAudioThumbnail().removeChangeListener(this);
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colours::darkgrey);
        g.setColour(juce::Colours::grey);
        g.drawRect(getLocalBounds(), 1);
        
        g.setColour(juce::Colours::orange);

        if (audioProcessor.getTotalLengthSecs() > 0)
        {
            audioProcessor.getAudioThumbnail().drawChannels(g, getLocalBounds(), 0.0, audioProcessor.getTotalLengthSecs(), 1.0f);

            auto playheadPos = (float)audioProcessor.getCurrentPositionSecs() / (float)audioProcessor.getTotalLengthSecs();
            g.setColour(juce::Colours::whitesmoke);
            g.drawVerticalLine(juce::roundToInt(playheadPos * getWidth()), 0.0f, (float)getHeight());
        }
    }
    
    void mouseDown(const juce::MouseEvent& event) override
    {
        if (audioProcessor.getTotalLengthSecs() > 0)
        {
            auto newPos = (double)event.x / (double)getWidth();
            auto newPosSecs = newPos * audioProcessor.getTotalLengthSecs();
            audioProcessor.setPlaybackPosition(newPosSecs);
        }
    }

    void changeListenerCallback(juce::ChangeBroadcaster* source) override
    {
        if (source == &audioProcessor.getAudioThumbnail())
        {
            repaint();
        }
    }
private:
    DynamicMusicVstAudioProcessor& audioProcessor;
};


//==============================================================================
class DynamicMusicVstAudioProcessorEditor  : public juce::AudioProcessorEditor,
                                           public juce::Timer
{
public:
    explicit DynamicMusicVstAudioProcessorEditor (DynamicMusicVstAudioProcessor&, juce::AudioProcessorValueTreeState&);
    ~DynamicMusicVstAudioProcessorEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;
    void timerCallback() override;
    void openAudioFile();

private:
    // Simple component to act as a flickering dot
    class BeatIndicator : public juce::Component
    {
    public:
        void paint(juce::Graphics& g) override
        {
            g.setColour(juce::Colours::orange.withAlpha(0.8f));
            g.fillEllipse(getLocalBounds().toFloat());
        }
    };

    DynamicMusicVstAudioProcessor& audioProcessor;
    juce::AudioProcessorValueTreeState& valueTreeState;
    
    WaveformDisplay waveformDisplay;
    SimilarityMatrixComponent similarityMatrixDisplay;
    juce::Slider targetDurationSlider;
    juce::Slider beatTightnessSlider;
    juce::Slider tempoInertiaSlider;
    juce::Slider onsetFollowingSlider;
    juce::Slider trimStartSlider;
    juce::Slider trimEndSlider;
    juce::TextButton analyzeButton;
    juce::TextButton openFileButton;
    juce::TextButton playButton;
    juce::TextButton stopButton;
    juce::Label statusLabel;
    juce::Label tempoLabel;
    juce::Label targetDurationLabel;
    juce::Label beatTightnessLabel;
    juce::Label tempoInertiaLabel;
    juce::Label onsetFollowingLabel;
    juce::Label trimStartLabel;
    juce::Label trimEndLabel;
    BeatIndicator beatIndicator;

    juce::uint32 lastBeatFlashTime = 0;

    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> targetDurationAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> beatTightnessAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> tempoInertiaAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> onsetFollowingAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> trimStartAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> trimEndAttachment;
    std::unique_ptr<juce::FileChooser> fileChooser;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DynamicMusicVstAudioProcessorEditor)
};
