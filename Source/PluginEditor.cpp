#include "PluginProcessor.h"
#include "PluginEditor.h"

DynamicMusicVstAudioProcessorEditor::DynamicMusicVstAudioProcessorEditor (DynamicMusicVstAudioProcessor& p, juce::AudioProcessorValueTreeState& vts)
    : AudioProcessorEditor (&p), audioProcessor (p), valueTreeState(vts), waveformDisplay(p)
{
    setSize (500, 800);

    // Waveform Display
    addAndMakeVisible(waveformDisplay);

    // Similarity Matrix Display
    addAndMakeVisible(similarityMatrixDisplay);

    // Target Duration Slider
    targetDurationAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(valueTreeState, "targetDuration", targetDurationSlider);
    targetDurationSlider.setSliderStyle(juce::Slider::Rotary);
    targetDurationSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 90, 20);
    targetDurationSlider.setPopupDisplayEnabled(true, false, this);
    targetDurationSlider.setTextValueSuffix(" s");
    addAndMakeVisible(targetDurationSlider);
    targetDurationLabel.setText("Target Duration", juce::dontSendNotification);
    targetDurationLabel.setJustificationType(juce::Justification::centredRight);
    addAndMakeVisible(targetDurationLabel);

    // Beat Tightness Slider
    beatTightnessAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(valueTreeState, "beatTightness", beatTightnessSlider);
    beatTightnessSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    beatTightnessSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 90, 20);
    beatTightnessSlider.setPopupDisplayEnabled(true, false, this);
    addAndMakeVisible(beatTightnessSlider);
    beatTightnessLabel.setText("Beat Tightness", juce::dontSendNotification);
    beatTightnessLabel.setJustificationType(juce::Justification::centredRight);
    addAndMakeVisible(beatTightnessLabel);

    // Tempo Inertia Slider
    tempoInertiaAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(valueTreeState, "tempoInertia", tempoInertiaSlider);
    tempoInertiaSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    tempoInertiaSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 90, 20);
    tempoInertiaSlider.setPopupDisplayEnabled(true, false, this);
    addAndMakeVisible(tempoInertiaSlider);
    tempoInertiaLabel.setText("Tempo Inertia", juce::dontSendNotification);
    tempoInertiaLabel.setJustificationType(juce::Justification::centredRight);
    addAndMakeVisible(tempoInertiaLabel);

    // Onset Following Slider
    onsetFollowingAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(valueTreeState, "onsetFollowing", onsetFollowingSlider);
    onsetFollowingSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    onsetFollowingSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 90, 20);
    onsetFollowingSlider.setPopupDisplayEnabled(true, false, this);
    addAndMakeVisible(onsetFollowingSlider);
    onsetFollowingLabel.setText("Onset Following", juce::dontSendNotification);
    onsetFollowingLabel.setJustificationType(juce::Justification::centredRight);
    addAndMakeVisible(onsetFollowingLabel);

    // Trim Sliders
    trimStartAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(valueTreeState, "trimStart", trimStartSlider);
    trimStartSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    trimStartSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 90, 20);
    trimStartSlider.setPopupDisplayEnabled(true, false, this);
    addAndMakeVisible(trimStartSlider);
    trimStartLabel.setText("Trim Start", juce::dontSendNotification);
    trimStartLabel.setJustificationType(juce::Justification::centredRight);
    addAndMakeVisible(trimStartLabel);

    trimEndAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(valueTreeState, "trimEnd", trimEndSlider);
    trimEndSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    trimEndSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 90, 20);
    trimEndSlider.setPopupDisplayEnabled(true, false, this);
    addAndMakeVisible(trimEndSlider);
    trimEndLabel.setText("Trim End", juce::dontSendNotification);
    trimEndLabel.setJustificationType(juce::Justification::centredRight);
    addAndMakeVisible(trimEndLabel);

    // Open File Button
    openFileButton.setButtonText("Open Audio File");
    openFileButton.onClick = [this] { openAudioFile(); };
    addAndMakeVisible(openFileButton);
    
    // Play/Stop Buttons
    playButton.setButtonText("Play");
    playButton.onClick = [this] { audioProcessor.startPlayback(); };
    addAndMakeVisible(playButton);

    stopButton.setButtonText("Stop");
    stopButton.onClick = [this] { audioProcessor.stopPlayback(); };
    addAndMakeVisible(stopButton);

    // Analyze Button
    analyzeButton.setButtonText("Analyze!");
    analyzeButton.onClick = [this] {
        audioProcessor.analysisState = DynamicMusicVstAudioProcessor::AnalysisState::AnalysisNeeded;
    };
    addAndMakeVisible(analyzeButton);

    // Status Label
    statusLabel.setText("Ready", juce::dontSendNotification);
    statusLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(statusLabel);
    
    // Tempo Label
    tempoLabel.setText("BPM: -", juce::dontSendNotification);
    tempoLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(tempoLabel);

    // Beat Indicator
    beatIndicator.setVisible(false);
    addAndMakeVisible(beatIndicator);

    startTimerHz(60); // Poll 60 times per second for smoother animation
}

DynamicMusicVstAudioProcessorEditor::~DynamicMusicVstAudioProcessorEditor()
{
    stopTimer();
}

void DynamicMusicVstAudioProcessorEditor::paint (juce::Graphics& g)
{
    g.fillAll (getLookAndFeel().findColour (juce::ResizableWindow::backgroundColourId));

    g.setColour (juce::Colours::white);
    g.setFont (15.0f);
    g.drawFittedText ("Dynamic Music VST!!!", getLocalBounds().removeFromTop(25), juce::Justification::centred, 1);
}

void DynamicMusicVstAudioProcessorEditor::openAudioFile()
{
    fileChooser = std::make_unique<juce::FileChooser>("Select an audio file...",
                                                       juce::File(),
                                                       "*.wav;*.mp3;*.aiff;*.flac");
    
    auto flags = juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles;
    
    fileChooser->launchAsync(flags, [this](const juce::FileChooser& fc)
    {
        auto file = fc.getResult();
        if (file.existsAsFile())
        {
            audioProcessor.loadAudioFile(file);
            statusLabel.setText("Loaded: " + file.getFileNameWithoutExtension(), juce::dontSendNotification);
        }
    });
}

void DynamicMusicVstAudioProcessorEditor::timerCallback()
{
    waveformDisplay.repaint();

    auto playheadRatio = audioProcessor.getTotalLengthSecs() > 0 ? (float)audioProcessor.getCurrentPositionSecs() / (float)audioProcessor.getTotalLengthSecs() : 0.0f;
    similarityMatrixDisplay.setPlayheadPosition(playheadRatio);

    if (audioProcessor.getTotalLengthSecs() > 0)
    {
        double totalSecs = audioProcessor.getTotalLengthSecs();
        trimStartSlider.setTextValueSuffix(" s (" + juce::String(trimStartSlider.getValue() * totalSecs, 2) + ")");
        trimEndSlider.setTextValueSuffix(" s (" + juce::String(trimEndSlider.getValue() * totalSecs, 2) + ")");
    }
    else
    {
        trimStartSlider.setTextValueSuffix(" s");
        trimEndSlider.setTextValueSuffix(" s");
    }

    // Beat flickering logic - a more robust version
    if (audioProcessor.isPlaying() && !audioProcessor.beatTimestamps.empty())
    {
        // Check if a beat has occurred since the last timer tick
        double currentTime = audioProcessor.getCurrentPositionSecs();
        double timeSinceLastTick = 1.0 / 60.0; // Corresponds to timer frequency (60Hz)
        double previousTime = currentTime - timeSinceLastTick;

        for(const auto& beatTime : audioProcessor.beatTimestamps)
        {
            if (beatTime > previousTime && beatTime <= currentTime)
            {
                lastBeatFlashTime = juce::Time::getMillisecondCounter();
                break; // Only need to flash once per tick
            }
        }
    }

    const juce::uint32 flashDurationMs = 100;
    if (juce::Time::getMillisecondCounter() - lastBeatFlashTime < flashDurationMs)
    {
        beatIndicator.setVisible(true);
    }
    else
    {
        beatIndicator.setVisible(false);
    }

    switch (audioProcessor.analysisState.load())
    {
        case DynamicMusicVstAudioProcessor::AnalysisState::Ready:
            statusLabel.setText("Ready!", juce::dontSendNotification);
            analyzeButton.setEnabled(true);
            break;
        case DynamicMusicVstAudioProcessor::AnalysisState::AnalysisNeeded:
            statusLabel.setText("Analysis required", juce::dontSendNotification);
            analyzeButton.setEnabled(true);
            break;
        case DynamicMusicVstAudioProcessor::AnalysisState::Analyzing:
            statusLabel.setText("Analyzing...", juce::dontSendNotification);
            analyzeButton.setEnabled(false); // Disable button while analyzing
            break;
        case DynamicMusicVstAudioProcessor::AnalysisState::AnalysisComplete:
            statusLabel.setText("Analysis Complete!", juce::dontSendNotification);
            tempoLabel.setText("BPM: " + juce::String(audioProcessor.estimatedBPM, 1), juce::dontSendNotification);
            analyzeButton.setEnabled(true);
            similarityMatrixDisplay.updateBeatInfo(audioProcessor.beatTimestamps, audioProcessor.getTotalLengthSecs());
            similarityMatrixDisplay.updateMatrix(audioProcessor.similarityMatrix);
            audioProcessor.analysisState = DynamicMusicVstAudioProcessor::AnalysisState::Ready; // Reset state
            break;
    }
}

void DynamicMusicVstAudioProcessorEditor::resized()
{
    auto bounds = getLocalBounds();
    bounds.removeFromTop(25); // Make space for title

    auto usableBounds = bounds.reduced(10);
    auto matrixWidth = usableBounds.getWidth();

    juce::FlexBox mainFlexbox;
    mainFlexbox.flexDirection = juce::FlexBox::Direction::column;

    // The similarity matrix display with a 1:1 aspect ratio
    mainFlexbox.items.add(juce::FlexItem(similarityMatrixDisplay).withHeight(matrixWidth));
    
    // The waveform display below the matrix
    mainFlexbox.items.add(juce::FlexItem(waveformDisplay).withHeight(80).withMargin(juce::FlexItem::Margin(5, 0, 0, 0)));

    // --- Controls Row ---
    juce::FlexBox controlsFlexbox;
    controlsFlexbox.flexDirection = juce::FlexBox::Direction::row;
    controlsFlexbox.justifyContent = juce::FlexBox::JustifyContent::spaceAround;
    controlsFlexbox.alignItems = juce::FlexBox::AlignItems::center;

    controlsFlexbox.items.add(juce::FlexItem(openFileButton).withWidth(120).withHeight(30));
    controlsFlexbox.items.add(juce::FlexItem(playButton).withWidth(60).withHeight(30));
    controlsFlexbox.items.add(juce::FlexItem(stopButton).withWidth(60).withHeight(30));
    controlsFlexbox.items.add(juce::FlexItem(analyzeButton).withWidth(80).withHeight(30));

    mainFlexbox.items.add(juce::FlexItem(controlsFlexbox).withFlex(1.0f));
    
    // --- Sliders Area ---
    auto createSliderRow = [](juce::Label& label, juce::Slider& slider)
    {
        juce::FlexBox box;
        box.flexDirection = juce::FlexBox::Direction::row;
        box.items.add(juce::FlexItem(label).withFlex(1.0f).withMargin(juce::FlexItem::Margin(0, 10, 0, 0)));
        box.items.add(juce::FlexItem(slider).withFlex(2.0f));
        return box;
    };
    
    juce::FlexBox durationRow = createSliderRow(targetDurationLabel, targetDurationSlider);
    juce::FlexBox tightnessRow = createSliderRow(beatTightnessLabel, beatTightnessSlider);
    juce::FlexBox inertiaRow = createSliderRow(tempoInertiaLabel, tempoInertiaSlider);
    juce::FlexBox onsetRow = createSliderRow(onsetFollowingLabel, onsetFollowingSlider);
    juce::FlexBox trimStartRow = createSliderRow(trimStartLabel, trimStartSlider);
    juce::FlexBox trimEndRow = createSliderRow(trimEndLabel, trimEndSlider);

    mainFlexbox.items.add(juce::FlexItem(durationRow).withFlex(1.0f));
    mainFlexbox.items.add(juce::FlexItem(tightnessRow).withFlex(1.0f));
    mainFlexbox.items.add(juce::FlexItem(inertiaRow).withFlex(1.0f));
    mainFlexbox.items.add(juce::FlexItem(onsetRow).withFlex(1.0f));
    mainFlexbox.items.add(juce::FlexItem(trimStartRow).withFlex(1.0f));
    mainFlexbox.items.add(juce::FlexItem(trimEndRow).withFlex(1.0f));

    // --- Status Area ---
    juce::FlexBox statusBox;
    statusBox.flexDirection = juce::FlexBox::Direction::row;
    statusBox.justifyContent = juce::FlexBox::JustifyContent::spaceAround;
    statusBox.items.add(juce::FlexItem(statusLabel).withFlex(1.0f));
    statusBox.items.add(juce::FlexItem(tempoLabel).withFlex(1.0f));
    
    mainFlexbox.items.add(juce::FlexItem(statusBox).withFlex(0.5f));
    
    mainFlexbox.performLayout(usableBounds);

    // Position beat indicator in the top right
    beatIndicator.setBounds(bounds.getRight() - 30, bounds.getY() + 10, 20, 20);
}
