#include "PluginProcessor.h"
#include "PluginEditor.h"

DynamicMusicVstAudioProcessorEditor::DynamicMusicVstAudioProcessorEditor (DynamicMusicVstAudioProcessor& p, juce::AudioProcessorValueTreeState& vts)
    : AudioProcessorEditor (&p), audioProcessor (p), valueTreeState(vts), waveformDisplay(p)
{
    setSize (900, 900);

    // Waveform Display
    addAndMakeVisible(waveformDisplay);

    // Similarity Matrix Display
    // addAndMakeVisible(similarityMatrixDisplay);

    // Tempogram Display
    addAndMakeVisible(tempogramDisplay);

    // Show Similarity Matrix Toggle
    // showSimilarityMatrixButton.setButtonText("Show Similarity Matrix");
    // addAndMakeVisible(showSimilarityMatrixButton);
    // showSimilarityMatrixAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(valueTreeState, "showSimilarityMatrix", showSimilarityMatrixButton);
    // valueTreeState.addParameterListener("showSimilarityMatrix", this);
    // Initial visibility for startup
    // similarityMatrixDisplay.setVisible(valueTreeState.getRawParameterValue("showSimilarityMatrix")->load());

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

    startTimerHz(60); // Poll 60 times per second for smoother animation
}

DynamicMusicVstAudioProcessorEditor::~DynamicMusicVstAudioProcessorEditor()
{
    // valueTreeState.removeParameterListener("showSimilarityMatrix", this);
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
    // similarityMatrixDisplay.setPlayheadPosition(playheadRatio);
    tempogramDisplay.setPlayheadPosition(playheadRatio);

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
            // similarityMatrixDisplay.updateBeatInfo(audioProcessor.beatTimestamps, audioProcessor.getTotalLengthSecs());
            // similarityMatrixDisplay.updateMatrix(audioProcessor.similarityMatrix);
            tempogramDisplay.updateDisplayInfo(audioProcessor.tempogram, audioProcessor.globalAcf, audioProcessor.getSampleRate(), 256);
            audioProcessor.analysisState = DynamicMusicVstAudioProcessor::AnalysisState::Ready; // Reset state
            break;
    }
}

void DynamicMusicVstAudioProcessorEditor::parameterChanged(const juce::String &parameterID, float newValue)
{
    // if (parameterID == "showSimilarityMatrix")
    // {
    //     similarityMatrixDisplay.setVisible(newValue > 0.5f);
    // }
}

void DynamicMusicVstAudioProcessorEditor::resized()
{
    auto bounds = getLocalBounds();
    bounds.removeFromTop(25); // Make space for title

    auto usableBounds = bounds.reduced(10);

    // --- Split main area into plots and controls ---
    auto controlsWidth = 300;
    auto plotsBounds = usableBounds.removeFromLeft(usableBounds.getWidth() - controlsWidth - 10);
    auto controlsBounds = usableBounds.withLeft(plotsBounds.getRight() + 10);

    // --- Layout Plots Column ---
    juce::FlexBox plotsBox;
    plotsBox.flexDirection = juce::FlexBox::Direction::column;
    // plotsBox.items.add(juce::FlexItem(similarityMatrixDisplay).withHeight(plotsBounds.getWidth()));
    plotsBox.items.add(juce::FlexItem(tempogramDisplay).withHeight(120).withMargin(juce::FlexItem::Margin(5, 0, 0, 0)));
    plotsBox.items.add(juce::FlexItem(waveformDisplay).withHeight(80).withMargin(juce::FlexItem::Margin(5, 0, 0, 0)));
    plotsBox.performLayout(plotsBounds);

    // --- Layout Controls Column ---
    juce::FlexBox controlsBox;
    controlsBox.flexDirection = juce::FlexBox::Direction::column;

    // Buttons
    juce::FlexBox buttonBox;
    buttonBox.flexDirection = juce::FlexBox::Direction::row;
    buttonBox.justifyContent = juce::FlexBox::JustifyContent::spaceAround;
    buttonBox.alignItems = juce::FlexBox::AlignItems::center;
    buttonBox.items.add(juce::FlexItem(openFileButton).withFlex(6.0f).withHeight(30));
    buttonBox.items.add(juce::FlexItem(playButton).withFlex(3.0f).withHeight(30));
    buttonBox.items.add(juce::FlexItem(stopButton).withFlex(3.0f).withHeight(30));
    buttonBox.items.add(juce::FlexItem(analyzeButton).withFlex(4.0f).withHeight(30));
    controlsBox.items.add(juce::FlexItem(buttonBox).withHeight(50));

    // Toggle
    // controlsBox.items.add(juce::FlexItem(showSimilarityMatrixButton).withHeight(30).withMargin(juce::FlexItem::Margin(10, 0, 10, 0)));

    // Sliders
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
    juce::FlexBox trimStartRow = createSliderRow(trimStartLabel, trimStartSlider);
    juce::FlexBox trimEndRow = createSliderRow(trimEndLabel, trimEndSlider);
    controlsBox.items.add(juce::FlexItem(durationRow).withFlex(1.0f));
    controlsBox.items.add(juce::FlexItem(tightnessRow).withFlex(1.0f));
    controlsBox.items.add(juce::FlexItem(trimStartRow).withFlex(1.0f));
    controlsBox.items.add(juce::FlexItem(trimEndRow).withFlex(1.0f));

    // Status
    juce::FlexBox statusBox;
    statusBox.flexDirection = juce::FlexBox::Direction::row;
    statusBox.justifyContent = juce::FlexBox::JustifyContent::spaceAround;
    statusBox.items.add(juce::FlexItem(statusLabel).withFlex(1.0f));
    statusBox.items.add(juce::FlexItem(tempoLabel).withFlex(1.0f));
    controlsBox.items.add(juce::FlexItem(statusBox).withFlex(0.5f));

    controlsBox.performLayout(controlsBounds);
}
