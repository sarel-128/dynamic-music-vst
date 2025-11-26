/*
  ==============================================================================

    AudioPlayer.h
    Created: 24 Nov 2025 10:00:00am
    Author:  Your Name

  ==============================================================================
*/

#pragma once

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_formats/juce_audio_formats.h>

class AudioPlayer  : public juce::Component,
                     public juce::Timer
{
public:
    AudioPlayer()
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

        setSize (300, 200);

        formatManager.registerBasicFormats();
        transportSource.addChangeListener (this);
        
        startTimer (20);
    }

    ~AudioPlayer() override
    {
        shutdownAudio();
    }

    void resized() override
    {
        juce::FlexBox flex;
        flex.flexDirection = juce::FlexBox::Direction::row;
        flex.justifyContent = juce::FlexBox::JustifyContent::spaceAround;
        flex.alignItems = juce::FlexBox::AlignItems::center;

        flex.items.add(juce::FlexItem(openButton).withFlex(1.0f));
        flex.items.add(juce::FlexItem(playButton).withFlex(1.0f));
        flex.items.add(juce::FlexItem(stopButton).withFlex(1.0f));
        
        flex.performLayout(getLocalBounds().reduced(10));
    }

    void openButtonClicked()
    {
        chooser = std::make_unique<juce::FileChooser> ("Select an audio file...",
                                                       juce::File{},
                                                       "*.wav;*.mp3;*.aiff");
        auto chooserFlags = juce::FileBrowserComponent::openMode
                          | juce::FileBrowserComponent::canSelectFiles;

        chooser->launchAsync (chooserFlags, [this] (const juce::FileChooser& fc)
        {
            auto file = fc.getResult();

            if (file != juce::File{})
            {
                auto* reader = formatManager.createReaderFor (file);

                if (reader != nullptr)
                {
                    auto newSource = std::make_unique<juce::AudioFormatReaderSource> (reader, true);
                    transportSource.setSource (newSource.get(), 0, nullptr, reader->sampleRate);
                    playButton.setEnabled (true);
                    readerSource.reset (newSource.release());
                }
            }
        });
    }

    void playButtonClicked()
    {
        transportSource.start();
        playButton.setEnabled(false);
        stopButton.setEnabled(true);
    }

    void stopButtonClicked()
    {
        transportSource.stop();
        stopButton.setEnabled(false);
        playButton.setEnabled(true);
    }
    
    juce::AudioSource& getAudioSource() { return transportSource; }

private:
    juce::TextButton openButton;
    juce::TextButton playButton;
    juce::TextButton stopButton;

    juce::AudioFormatManager formatManager;
    std::unique_ptr<juce::AudioFormatReaderSource> readerSource;
    juce::AudioTransportSource transportSource;
    
    std::unique_ptr<juce::FileChooser> chooser;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AudioPlayer)
};
