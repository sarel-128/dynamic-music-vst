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

    for (int i = 0; i < numCells; ++i)
    {
        for (int j = 0; j < numCells; ++j)
        {
            float value = matrix[i][j]; // Should be in [-1, 1] for correlation
            // Map value from [-1, 1] to a color. For example, a grayscale.
            // Or a more complex color map. Let's use a simple grayscale.
            // Remap [-1, 1] to [0, 1] for color intensity.
            float intensity = (value + 1.0f) / 2.0f;
            g.setColour(juce::Colour::fromFloatRGBA(intensity, intensity, intensity, 1.0f));
            g.fillRect(i * cellWidth, j * cellHeight, cellWidth, cellHeight);
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
