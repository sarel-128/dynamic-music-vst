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
    void updatePath(const std::vector<int>& newPath);
    void setPlayheadPosition(float newPosition);
    void setRetargetedMode(bool isRetargeted);
    void setTargetDuration(double targetDur);
    void clearConstraints();
    
    std::function<void(const std::vector<std::pair<float, float>>&)> onConstraintsChanged;

private:
    void mouseDown(const juce::MouseEvent& event) override;
    void resized() override;
    void updateMatrixImage();
    void updatePathImage();

    std::vector<std::vector<float>> matrix;
    juce::Image matrixImage;
    juce::Image pathImage;
    std::vector<double> beats;
    std::vector<int> path;
    std::vector<std::pair<float, float>> constraints;
    double duration = 0.0;
    double targetDuration = 0.0;
    float playheadPosition = 0.0f;
    bool inRetargetedMode = false;

    juce::Point<int> mousePosition;
    bool isMouseOver = false;
};
