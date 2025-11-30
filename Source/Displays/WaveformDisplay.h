#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include "PluginProcessor.h"

class WaveformDisplay : public juce::Component
{
public:
    WaveformDisplay(DynamicMusicVstAudioProcessor& p) : audioProcessor(p) {}
    
    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colours::darkgrey);
        g.setColour(juce::Colours::white);
        g.drawFittedText("Waveform Display", getLocalBounds(), juce::Justification::centred, 1);
    }
    
private:
    DynamicMusicVstAudioProcessor& audioProcessor;
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(WaveformDisplay)
};
