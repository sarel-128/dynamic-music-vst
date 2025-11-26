/*
  ==============================================================================

    AudioPlayerNode.h
    Created: 24 Nov 2025 1:00:00pm
    Author:  Your Name

  ==============================================================================
*/

#pragma once

#include "../JuceLibraryCode/JuceHeader.h"
#include "Graph/FilterGraph.h"

class AudioPlayerNode   : public juce::Component
{
public:
    AudioPlayerNode (FilterGraph& graph)
        : filterGraph (graph)
    {
        addAndMakeVisible (openButton);
        openButton.setButtonText ("Open...");
        openButton.onClick = [this] { openButtonClicked(); };

        addAndMakeVisible (playButton);
        playButton.setButtonText ("Play");
        playButton.onClick = [this] { playButtonClicked(); };
        playButton.setColour (juce::TextButton::buttonColourId, juce::Colours::green);
        playButton.setEnabled (false);

        addAndMakeVisible (stopButton);
        stopButton.setButtonText ("Stop");
        stopButton.onClick = [this] { stopButtonClicked(); };
        stopButton.setColour (juce::TextButton::buttonColourId, juce::Colours::red);
        stopButton.setEnabled (false);
        
        setSize (200, 100);
        
        formatManager.registerBasicFormats();
    }

    void paint (juce::Graphics& g) override
    {
        g.fillAll (getLookAndFeel().findColour (juce::ResizableWindow::backgroundColourId).darker());
    }

    void resized() override
    {
        juce::FlexBox fb;
        fb.flexDirection = juce::FlexBox::Direction::row;
        fb.items.add (juce::FlexItem (openButton).withFlex (1.0f));
        fb.items.add (juce::FlexItem (playButton).withFlex (1.0f));
        fb.items.add (juce::FlexItem (stopButton).withFlex (1.0f));
        fb.performLayout (getLocalBounds().reduced (10));
    }

private:
    void openButtonClicked()
    {
        chooser = std::make_unique<juce::FileChooser> ("Select a Wave or AIFF file to play...",
                                                     juce::File{},
                                                     "*.wav;*.aiff;*.mp3");
        auto chooserFlags = juce::FileBrowserComponent::openMode
                          | juce::FileBrowserComponent::canSelectFiles;

        chooser->launchAsync (chooserFlags, [this] (const juce::FileChooser& fc)
        {
            auto file (fc.getResult());
            if (file != juce::File{})
            {
                filterGraph.addAudioPlayer (file);
                playButton.setEnabled (true);
            }
        });
    }

    void playButtonClicked()
    {
        playButton.setEnabled (false);
        stopButton.setEnabled (true);
        // This is a simplified way to control playback; a real implementation would be more robust.
        // We'll tell the graph to start playing all audio file players.
    }

    void stopButtonClicked()
    {
        playButton.setEnabled (true);
        stopButton.setEnabled (false);
        // We'll tell the graph to stop playing all audio file players.
    }

    FilterGraph& filterGraph;

    juce::TextButton openButton;
    juce::TextButton playButton;
    juce::TextButton stopButton;
    
    juce::AudioFormatManager formatManager;
    std::unique_ptr<juce::FileChooser> chooser;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AudioPlayerNode)
};
