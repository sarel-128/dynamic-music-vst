#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <vector>

class SimilarityMatrixComponent : public juce::Component
{
public:
    SimilarityMatrixComponent();

    void paint(juce::Graphics& g) override;

    void mouseMove(const juce::MouseEvent& event) override;
    void mouseExit(const juce::MouseEvent& event) override;

    void updateMatrix(const std::vector<std::vector<float>>& newMatrix);
    void updateBeatInfo(const std::vector<double>& beatTimestamps, double totalDuration);
    void setPlayheadPosition(float newPosition);

private:
    std::vector<std::vector<float>> matrix;
    std::vector<double> beats;
    double duration = 0.0;
    float playheadPosition = 0.0f;

    juce::Point<int> mousePosition;
    bool isMouseOver = false;
};
