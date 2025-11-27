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
class TempogramDisplayComponent : public juce::Component
{
public:
    TempogramDisplayComponent() = default;

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colours::black);

        if (tempogramImage.isNull())
        {
            g.setColour(juce::Colours::white);
            g.drawText("No data", getLocalBounds(), juce::Justification::centred, false);
            return;
        }
        
        // --- Draw the cached tempogram image ---
        g.drawImage(tempogramImage, getLocalBounds().toFloat());

        // --- Y-axis info needed for overlays ---
        const int minBPM = 16;
        const int maxBPM = 300;
        int minLag = static_cast<int>(60.0 * sampleRate / (hopLength * maxBPM));
        int maxLag = static_cast<int>(60.0 * sampleRate / (hopLength * minBPM));
        minLag = juce::jmax(0, minLag);
        if (tempogram.size() > 0)
        {
            maxLag = juce::jmin((int)tempogram.size() - 1, maxLag);
        } else {
            maxLag = minLag;
        }
        
        if (minLag >= maxLag) return; // Nothing to draw for overlays
        
        int numLagsToDisplay = maxLag - minLag + 1;
        float cellHeight = (float)getHeight() / numLagsToDisplay;

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
            int numFrames = tempogram.empty() ? 0 : tempogram[0].size();
            float frameIndex = (float)mousePosition.x / getWidth() * numFrames;
            float timeInSeconds = frameIndex * hopLength / sampleRate;

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
                glyphs.addJustifiedText(g.getCurrentFont(), labelText, 0, 0, getWidth(), 1.0f);
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
        renderTempogramImage();
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
    void renderTempogramImage()
    {
        if (tempogram.empty() || tempogram[0].empty())
        {
            tempogramImage = juce::Image(); // Clear image
            return;
        }
        
        // Define a fixed resolution for the image to improve performance
        const int imageWidth = 512;
        const int imageHeight = 256;
        tempogramImage = juce::Image(juce::Image::RGB, imageWidth, imageHeight, true);
        juce::Graphics g(tempogramImage);
        
        int numFrames = tempogram[0].size();
        int numLagsTotal = tempogram.size();
        
        const int minBPM = 16;
        const int maxBPM = 300;
        int minLag = static_cast<int>(60.0 * sampleRate / (hopLength * maxBPM));
        int maxLag = static_cast<int>(60.0 * sampleRate / (hopLength * minBPM));
        minLag = juce::jmax(0, minLag);
        maxLag = juce::jmin(numLagsTotal - 1, maxLag);

        if (minLag >= maxLag) return;

        int numLagsToDisplay = maxLag - minLag + 1;
        float cellWidth = (float)imageWidth / numFrames;
        float cellHeight = (float)imageHeight / numLagsToDisplay;

        float minVal = std::numeric_limits<float>::max();
        float maxVal = std::numeric_limits<float>::min();
        for (int i=0; i < numFrames; ++i)
        {
            for (int j=minLag; j <= maxLag; ++j)
            {
                minVal = std::min(minVal, tempogram[j][i]);
                maxVal = std::max(maxVal, tempogram[j][i]);
            }
        }

        juce::ColourGradient gradient;
        gradient.addColour(0.0, juce::Colour::fromString("0xff440154"));
        gradient.addColour(0.25, juce::Colour::fromString("0xff3b528b"));
        gradient.addColour(0.5, juce::Colour::fromString("0xff21918c"));
        gradient.addColour(0.75, juce::Colour::fromString("0xff5ec962"));
        gradient.addColour(1.0, juce::Colour::fromString("0xfffde725"));

        for (int i = 0; i < numFrames; ++i)
        {
            for (int j = minLag; j <= maxLag; ++j)
            {
                float value = tempogram[j][i];
                double proportion = 0.0;
                if (maxVal > minVal)
                {
                    proportion = (value - minVal) / (maxVal - minVal);
                }
                g.setColour(gradient.getColourAtPosition(proportion));
                
                int displayRow = j - minLag;
                g.fillRect((float)i * cellWidth, (float)(numLagsToDisplay - 1 - displayRow) * cellHeight, cellWidth, cellHeight);
            }
        }
    }

    std::vector<std::vector<float>> tempogram;
    std::vector<float> aggregatedProfile;
    juce::Image tempogramImage;
    float playheadPosition = 0.0f;
    double sampleRate = 44100.0;
    int hopLength = 512;
    juce::Point<int> mousePosition;
    bool isMouseOver = false;
};


//==============================================================================
class DynamicMusicVstAudioProcessorEditor  : public juce::AudioProcessorEditor,
                                           public juce::Timer,
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

    DynamicMusicVstAudioProcessor& audioProcessor;
    juce::AudioProcessorValueTreeState& valueTreeState;
    
    WaveformDisplay waveformDisplay;
    // SimilarityMatrixComponent similarityMatrixDisplay;
    TempogramDisplayComponent tempogramDisplay;
    juce::Slider targetDurationSlider;
    juce::Slider beatTightnessSlider;
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
    juce::Label trimStartLabel;
    juce::Label trimEndLabel;
    // juce::ToggleButton showSimilarityMatrixButton;

    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> targetDurationAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> beatTightnessAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> trimStartAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> trimEndAttachment;
    // std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> showSimilarityMatrixAttachment;
    std::unique_ptr<juce::FileChooser> fileChooser;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DynamicMusicVstAudioProcessorEditor)
};
