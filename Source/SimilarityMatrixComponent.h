#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <vector>

// Forward declaration
struct ConstraintPoint;

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
    
    // Update constraints from processor (for display)
    void updateConstraints(const std::vector<ConstraintPoint>& newConstraints);
    
    // Event callbacks for discrete constraint changes
    std::function<void(float sourceTime, float targetTime)> onConstraintAdded;
    std::function<void(int id, float newSourceTime, float newTargetTime)> onConstraintMoved;
    std::function<void(int id)> onConstraintRemoved;

private:
    void mouseDown(const juce::MouseEvent& event) override;
    void mouseDrag(const juce::MouseEvent& event) override;
    void mouseUp(const juce::MouseEvent& event) override;
    void resized() override;
    void updateMatrixImage();
    void updatePathImage();
    
    int findConstraintAtPosition(juce::Point<int> pos, float hitRadius = 10.0f) const;

    std::vector<std::vector<float>> matrix;
    juce::Image matrixImage;
    juce::Image pathImage;
    std::vector<double> beats;
    std::vector<int> path;
    
    // Constraint storage - mirrors the processor's constraints for display
    struct DisplayConstraint
    {
        int id;
        float sourceTime;
        float targetTime;
        bool isStartAnchor = false;  // ID 0 - completely fixed
        bool isEndAnchor = false;    // ID 1 - only X-axis movable
    };
    std::vector<DisplayConstraint> constraints;
    
    // Interaction state
    int selectedConstraintId = -1;
    bool isDragging = false;
    juce::Point<int> dragStartPosition;
    
    double duration = 0.0;
    double targetDuration = 0.0;
    float playheadPosition = 0.0f;
    bool inRetargetedMode = false;

    juce::Point<int> mousePosition;
    bool isMouseOver = false;
};
