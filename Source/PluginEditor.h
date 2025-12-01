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

        int waveformVisibleWidth = getWidth();

        if (audioProcessor.isRetargeted.load())
        {
            auto& buffer = audioProcessor.retargetedAudioBuffer;
            auto originalLengthSecs = audioProcessor.getOriginalTotalLengthSecs();

            if (buffer.getNumSamples() > 0 && originalLengthSecs > 0)
            {
                auto retargetedLengthSecs = buffer.getNumSamples() / audioProcessor.getSampleRate();
                auto proportion = juce::jlimit(0.0, 1.0, retargetedLengthSecs / originalLengthSecs);
                waveformVisibleWidth = (int)(getWidth() * proportion);

                juce::Path path;
                
                const auto numSamples = buffer.getNumSamples();
                const auto* channelData = buffer.getReadPointer(0); // Draw mono for simplicity

                const float xScale = (float)waveformVisibleWidth / numSamples;
                const float yScale = getHeight() * 0.5f;

                path.startNewSubPath(0, yScale);

                for (int i = 0; i < numSamples; ++i)
                {
                    path.lineTo(i * xScale, yScale - channelData[i] * yScale);
                }
                
                g.strokePath(path, juce::PathStrokeType(1.0f));
            }
        }
        else if (audioProcessor.getTotalLengthSecs() > 0)
        {
            audioProcessor.getAudioThumbnail().drawChannels(g, getLocalBounds(), 0.0, audioProcessor.getTotalLengthSecs(), 1.0f);
        }
        
        if (audioProcessor.getTotalLengthSecs() > 0)
        {
            auto playheadPos = (float)audioProcessor.getCurrentPositionSecs() / (float)audioProcessor.getTotalLengthSecs();
            g.setColour(juce::Colours::whitesmoke);
            g.drawVerticalLine(juce::roundToInt(playheadPos * waveformVisibleWidth), 0.0f, (float)getHeight());
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
class OnsetEnvelopeDisplay : public juce::Component
{
public:
    OnsetEnvelopeDisplay() = default;

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colours::black);

        if (onsetEnvelope.empty())
        {
            g.setColour(juce::Colours::white);
            g.drawText("No Onset Envelope Data", getLocalBounds(), juce::Justification::centred, false);
            return;
        }

        g.setColour(juce::Colours::yellow);

        auto maxValue = *std::max_element(onsetEnvelope.begin(), onsetEnvelope.end());
        if (maxValue <= 0) maxValue = 1.0f;

        juce::Path path;
        path.startNewSubPath(0, getHeight());

        for (int i = 0; i < onsetEnvelope.size(); ++i)
        {
            auto x = (float)i / (float)(onsetEnvelope.size() - 1) * getWidth();
            auto y = getHeight() - (onsetEnvelope[i] / maxValue) * getHeight();
            path.lineTo(x, y);
        }
        
        path.lineTo(getWidth(), getHeight());
        path.closeSubPath();
        
        g.setColour(juce::Colours::yellow.withAlpha(0.5f));
        g.fillPath(path);
        
        g.setColour(juce::Colours::yellow);
        g.strokePath(path, juce::PathStrokeType(1.0f));

        // Draw beat markers
        if (totalDuration > 0 && !beatTimestamps.empty())
        {
            g.setColour(juce::Colours::cyan.withAlpha(0.7f));
            for (const auto& timestamp : beatTimestamps)
            {
                float x = (float)(timestamp / totalDuration) * getWidth();
                g.drawVerticalLine(juce::roundToInt(x), 0.0f, (float)getHeight());
            }
        }

        // Draw playhead
        g.setColour(juce::Colours::red.withAlpha(0.7f));
        float xPosition = playheadPosition * getWidth();
        g.drawVerticalLine(juce::roundToInt(xPosition), 0.0f, (float)getHeight());
    }

    void updateOnsetEnvelope(const std::vector<float>& newEnvelope)
    {
        onsetEnvelope = newEnvelope;
        repaint();
    }

    void updateBeatInfo(const std::vector<double>& newBeatTimestamps, double newTotalDuration)
    {
        beatTimestamps = newBeatTimestamps;
        totalDuration = newTotalDuration;
        repaint();
    }

    void setPlayheadPosition(float newPosition)
    {
        playheadPosition = juce::jlimit(0.0f, 1.0f, newPosition);
        repaint();
    }

private:
    std::vector<float> onsetEnvelope;
    std::vector<double> beatTimestamps;
    double totalDuration = 0.0;
    float playheadPosition = 0.0f;
};

//==============================================================================
class TempogramDisplayComponent : public juce::Component
{
public:
    TempogramDisplayComponent() = default;

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colours::black);

        if (tempogram.empty() || tempogram[0].empty())
        {
            g.setColour(juce::Colours::white);
            g.drawText("No data", getLocalBounds(), juce::Justification::centred, false);
            return;
        }

        int numFrames = tempogram.size();
        int numLagsTotal = tempogram[0].size();
        
        // --- Y-axis cropping to 55-200 BPM range ---
        const int minBPM = 55;
        const int maxBPM = 200;
        int minLag = static_cast<int>(60.0 * sampleRate / (hopLength * maxBPM));
        int maxLag = static_cast<int>(60.0 * sampleRate / (hopLength * minBPM));
        minLag = juce::jmax(0, minLag);
        maxLag = juce::jmin(numLagsTotal - 1, maxLag);

        if (minLag >= maxLag)
        {
            g.setColour(juce::Colours::white);
            g.drawText("Tempo range not available", getLocalBounds(), juce::Justification::centred, false);
            return;
        }

        int numLagsToDisplay = maxLag - minLag + 1;
        
        int numFramesToDraw = (numFrames + resolution - 1) / resolution;
        float cellWidth = (float)getWidth() / numFramesToDraw;
        float cellHeight = (float)getHeight() / numLagsToDisplay;

        // --- Normalize the tempogram based on the visible range for full contrast ---
        float minVal = std::numeric_limits<float>::max();
        float maxVal = std::numeric_limits<float>::min();
        for (int i=0; i < numFrames; ++i)
        {
            for (int j=minLag; j <= maxLag; ++j)
            {
                minVal = std::min(minVal, tempogram[i][j]);
                maxVal = std::max(maxVal, tempogram[i][j]);
            }
        }

        // Create a viridis-like color gradient
        juce::ColourGradient gradient;
        gradient.addColour(0.0, juce::Colour::fromString("0xff440154")); // Dark Purple
        gradient.addColour(0.25, juce::Colour::fromString("0xff3b528b"));// Blue
        gradient.addColour(0.5, juce::Colour::fromString("0xff21918c")); // Green
        gradient.addColour(0.75, juce::Colour::fromString("0xff5ec962"));// Light Green
        gradient.addColour(1.0, juce::Colour::fromString("0xfffde725")); // Yellow

        for (int i = 0; i < numFramesToDraw; ++i)
        {
            for (int j = minLag; j <= maxLag; ++j)
            {
                float avgValue = 0.0f;
                int count = 0;
                for (int k = 0; k < resolution; ++k)
                {
                    int frameIndex = i * resolution + k;
                    if (frameIndex < numFrames)
                    {
                        avgValue += tempogram[frameIndex][j];
                        count++;
                    }
                }

                if (count > 0)
                {
                    avgValue /= count;
                }
                
                double proportion = 0.0;
                if (maxVal > minVal)
                {
                    proportion = (avgValue - minVal) / (maxVal - minVal);
                }
                g.setColour(gradient.getColourAtPosition(proportion));
                
                // Map lag 'j' to the display rect
                int displayRow = j - minLag;
                g.fillRect((float)i * cellWidth, (float)(numLagsToDisplay - 1 - displayRow) * cellHeight, cellWidth, cellHeight);
            }
        }

        // --- Overlay the aggregated tempo profile ---
        if (!aggregatedProfile.empty())
        {
            float acfNorm = aggregatedProfile[0];
            if (acfNorm <= 0) acfNorm = 1.0f;

            float maxAcfVal = 0.0f;
            for(int i = minLag; i <= maxLag; ++i)
            {
                if (i < aggregatedProfile.size())
                    maxAcfVal = std::max(maxAcfVal, aggregatedProfile[i] / acfNorm);
            }
            
            g.setColour(juce::Colours::red.withAlpha(0.8f));
            juce::Path acfPath;

            for (int j = minLag; j <= maxLag; ++j)
            {
                if (j >= aggregatedProfile.size()) continue;

                float strength = (aggregatedProfile[j] / acfNorm);
                if (maxAcfVal > 0)
                {
                    strength /= maxAcfVal; // Normalize to fit width
                }

                int displayRow = j - minLag;
                float yPos = (float)(numLagsToDisplay - 1 - displayRow) * cellHeight + cellHeight / 2.0f;
                float xPos = strength * getWidth() * 0.5f; // Use 50% of width for plot

                if (j == minLag)
                    acfPath.startNewSubPath(xPos, yPos);
                else
                    acfPath.lineTo(xPos, yPos);
            }
            g.strokePath(acfPath, juce::PathStrokeType(2.0f));
        }

        // Draw playhead
        g.setColour(juce::Colours::red.withAlpha(0.7f));
        float xPosition = playheadPosition * getWidth();
        g.drawVerticalLine(juce::roundToInt(xPosition), 0.0f, (float)getHeight());
        
        // Draw hover label
        if (isMouseOver)
        {
            // Calculate Time (X-axis)
            float timeRatio = mousePosition.x / (float)getWidth();
            float totalTime = (float)numFrames * hopLength / sampleRate;
            float timeInSeconds = timeRatio * totalTime;

            // Calculate BPM (Y-axis)
            float yProportion = 1.0f - ((float)mousePosition.y / getHeight());
            int lagOffset = static_cast<int>(yProportion * numLagsToDisplay);
            int lag = minLag + lagOffset;
            
            float bpm = 0.0f;
            if (lag > 0)
            {
                bpm = 60.0f * (float)sampleRate / ((float)hopLength * lag);
            }

            if (bpm >= minBPM && bpm <= maxBPM)
            {
                juce::String labelText = "Time: " + juce::String(timeInSeconds, 2) + "s, BPM: " + juce::String(bpm, 1);
                
                juce::GlyphArrangement glyphs;
                glyphs.addJustifiedText(g.getCurrentFont(), labelText, 0, 0, (float)getWidth(), juce::Justification::left);
                float textWidth = glyphs.getBoundingBox(0, glyphs.getNumGlyphs(), true).getWidth();
                
                auto x = mousePosition.x + 12;
                auto y = mousePosition.y;
                
                if (x + textWidth > getWidth())
                    x = mousePosition.x - textWidth - 12;
                
                g.setColour(juce::Colours::black.withAlpha(0.7f));
                g.fillRoundedRectangle(x, y - 20, textWidth + 8, 18, 5.0f);

                g.setColour(juce::Colours::white);
                g.drawText(labelText, (int) x + 4, (int) y - 18, (int) textWidth, 14, juce::Justification::left, true);
            }
        }
    }

    void updateDisplayInfo(const std::vector<std::vector<float>>& newTempogram, const std::vector<float>& newAggregatedProfile, double sr, int hl)
    {
        tempogram = newTempogram;
        aggregatedProfile = newAggregatedProfile;
        sampleRate = sr;
        hopLength = hl;
        repaint();
    }

    void setPlayheadPosition(float newPosition)
    {
        playheadPosition = juce::jlimit(0.0f, 1.0f, newPosition);
        repaint();
    }
    
    void mouseMove(const juce::MouseEvent& event) override
    {
        mousePosition = event.getPosition();
        isMouseOver = true;
        repaint();
    }
    
    void mouseExit(const juce::MouseEvent& event) override
    {
        isMouseOver = false;
        repaint();
    }

private:
    std::vector<std::vector<float>> tempogram;
    std::vector<float> aggregatedProfile;
    float playheadPosition = 0.0f;
    double sampleRate = 44100.0;
    int hopLength = 512;
    int resolution = 256;
    juce::Point<int> mousePosition;
    bool isMouseOver = false;
};


//==============================================================================
class DynamicMusicVstAudioProcessorEditor  : public juce::AudioProcessorEditor,
                                           public juce::Timer,
                                           private juce::ChangeListener,
                                           private juce::AudioProcessorValueTreeState::Listener
{
public:
    explicit DynamicMusicVstAudioProcessorEditor (DynamicMusicVstAudioProcessor&, juce::AudioProcessorValueTreeState&);
    ~DynamicMusicVstAudioProcessorEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;
    void timerCallback() override;
    void openAudioFile();

private:
    void parameterChanged (const juce::String& parameterID, float newValue) override;
    void changeListenerCallback (juce::ChangeBroadcaster* source) override;

    DynamicMusicVstAudioProcessor& audioProcessor;
    juce::AudioProcessorValueTreeState& valueTreeState;
    
    WaveformDisplay waveformDisplay;
    SimilarityMatrixComponent similarityMatrixDisplay;
    
    //OnsetEnvelopeDisplay onsetEnvelopeDisplay;
    //TempogramDisplayComponent tempogramDisplay;
    juce::Slider targetDurationSlider;
    juce::Slider beatTightnessSlider;
    juce::Slider trimStartSlider;
    juce::Slider trimEndSlider;
    juce::TextButton analyzeButton;
    juce::TextButton openFileButton;
    juce::TextButton playButton;
    juce::TextButton stopButton;
    juce::TextButton retargetButton;
    juce::Label statusLabel;
    juce::Label tempoLabel;
    juce::Label targetDurationLabel;
    juce::Label beatTightnessLabel;
    juce::Label trimStartLabel;
    juce::Label trimEndLabel;
    juce::ToggleButton showSimilarityMatrixButton;

    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> targetDurationAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> beatTightnessAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> trimStartAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> trimEndAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> showSimilarityMatrixAttachment;
    std::unique_ptr<juce::FileChooser> fileChooser;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DynamicMusicVstAudioProcessorEditor)
};
