#include "SimilarityMatrixComponent.h"

SimilarityMatrixComponent::SimilarityMatrixComponent()
{
    // Set a default size or wait for resized()
}

void SimilarityMatrixComponent::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colours::black);

    if (matrix.empty())
    {
        g.setColour(juce::Colours::white);
        g.drawText("No data", getLocalBounds(), juce::Justification::centred, false);
        return;
    }

    int numCells = matrix.size();
    if (numCells == 0) return;
    
    float cellWidth = (float)getWidth() / numCells;
    float cellHeight = (float)getHeight() / numCells;

    // --- Normalize the matrix for full contrast ---
    float minVal = 1.0f, maxVal = -1.0f;
    for (int i = 0; i < numCells; ++i)
    {
        for (int j = 0; j < numCells; ++j)
        {
            minVal = std::min(minVal, matrix[i][j]);
            maxVal = std::max(maxVal, matrix[i][j]);
        }
    }

    // Create a high-contrast color gradient for better visibility.
    juce::ColourGradient gradient;
    gradient.addColour(0.0, juce::Colours::darkblue);
    gradient.addColour(0.5, juce::Colours::grey);
    gradient.addColour(1.0, juce::Colours::yellow);

    for (int i = 0; i < numCells; ++i)
    {
        for (int j = 0; j < numCells; ++j)
        {
            float value = matrix[i][j];
            // Normalize the value to the [0, 1] range based on the matrix's actual min and max.
            double proportion = 0.0;
            if (maxVal > minVal)
            {
                proportion = (value - minVal) / (maxVal - minVal);
            }
            g.setColour(gradient.getColourAtPosition(proportion));
            g.fillRect(i * cellWidth, j * cellHeight, cellWidth, cellHeight);
        }
    }

    // Draw hover label
    if (isMouseOver)
    {
        int xBeat = static_cast<int>(mousePosition.x / cellWidth);
        int yBeat = static_cast<int>(mousePosition.y / cellHeight);

        if (xBeat >= 0 && xBeat < numCells && yBeat >= 0 && yBeat < numCells)
        {
            juce::String labelText = "Beat X: " + juce::String(xBeat) + ", Beat Y: " + juce::String(yBeat);
            
            float textWidth = g.getCurrentFont().getStringWidth(labelText);

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

    // Draw playhead
    if (beats.empty() || duration <= 0.0) return;

    double currentTime = playheadPosition * duration;

    int currentBeatIndex = -1;
    if (!beats.empty())
    {
        auto it = std::upper_bound(beats.begin(), beats.end(), currentTime);
        if (it != beats.begin())
        {
            currentBeatIndex = std::distance(beats.begin(), it) - 1;
        }
    }

    if (currentBeatIndex != -1)
    {
        double startTimeOfBeat = beats[currentBeatIndex];
        double endTimeOfBeat = (currentBeatIndex + 1 < (int)beats.size()) ? beats[currentBeatIndex + 1] : duration;
        double beatDuration = endTimeOfBeat - startTimeOfBeat;
        
        double fractionalPosition = 0.0;
        if (beatDuration > 0)
        {
            fractionalPosition = (currentTime - startTimeOfBeat) / beatDuration;
        }

        float xPosition = (currentBeatIndex + fractionalPosition) * cellWidth;

        g.setColour(juce::Colours::red.withAlpha(0.7f));
        g.drawVerticalLine(juce::roundToInt(xPosition), 0.0f, (float)getHeight());
    }
}

void SimilarityMatrixComponent::mouseMove(const juce::MouseEvent& event)
{
    mousePosition = event.getPosition();
    isMouseOver = true;
    repaint();
}

void SimilarityMatrixComponent::mouseExit(const juce::MouseEvent& event)
{
    isMouseOver = false;
    repaint();
}

void SimilarityMatrixComponent::updateMatrix(const std::vector<std::vector<float>>& newMatrix)
{
    matrix = newMatrix;
    repaint();
}

void SimilarityMatrixComponent::updateBeatInfo(const std::vector<double>& beatTimestamps, double totalDuration)
{
    beats = beatTimestamps;
    duration = totalDuration;
}

void SimilarityMatrixComponent::setPlayheadPosition(float newPosition)
{
    playheadPosition = juce::jlimit(0.0f, 1.0f, newPosition);
    repaint();
}
