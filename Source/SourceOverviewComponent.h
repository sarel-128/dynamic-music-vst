#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_graphics/juce_graphics.h>
#include <vector>
#include <functional>

// Forward declare the Handle struct from PluginEditor.h
struct Handle;
class DynamicMusicVstAudioProcessorEditor;

class SourceOverviewComponent : public juce::Component
{
public:
    SourceOverviewComponent();
    ~SourceOverviewComponent() override;

    void paint(juce::Graphics&) override;
    void resized() override;
    void mouseWheelMove(const juce::MouseEvent& event, const juce::MouseWheelDetails& wheel) override;
    void mouseMagnify(const juce::MouseEvent& event, float scaleFactor) override;
    void mouseEnter(const juce::MouseEvent& event) override;
    void mouseExit(const juce::MouseEvent& event) override;
    void mouseDown(const juce::MouseEvent& event) override;
    void mouseDoubleClick(const juce::MouseEvent& event) override;
    void mouseDrag(const juce::MouseEvent& event) override;
    void mouseUp(const juce::MouseEvent& event) override;
    void mouseMove(const juce::MouseEvent& event) override;

    void setSourceAudio(const juce::AudioBuffer<float>& newSource, double newSampleRate);
    void setHandles(const std::vector<Handle>* handlesToDisplay);
    void setUnusedRegions(const std::vector<std::pair<double, double>>& regions);

    juce::Point<float> getScreenPositionForHandle(int handleId) const;
    int getHandleIdAt(juce::Point<float> localPoint) const;

    std::function<void(double time)> onHandleAddRequest;
    std::function<void(int handleId, double newTime)> onHandleMoved;
    std::function<void(int handleId)> onHandleDeleteRequest;
    std::function<void()> onViewChanged;
    std::function<void(double time)> onCursorMoved;
    std::function<void(double time)> onSeekRequest;
    std::function<void()> onScrubStart;
    std::function<void(double time)> onScrubRequest;
    std::function<void()> onScrubEnd;
    std::function<void(int handleId)> onHandleHoverChanged;

    void setHoveredHandle(int handleId);

    void setCursorPosition(const std::vector<double>& times);
    void setPlayheadPosition(double time);
    void setPlayheadIsLeading(bool isLeading);

private:
    void generateWaveformPath();
    void updateBackgroundCache();

    // Coordinate mapping functions
    float timeToX(double time) const;
    double xToTime(float x) const;
    double getTotalSourceDuration() const;
    double getVisibleDuration() const;

    // View State
    double zoomFactor { 1.0 };
    double viewStartSeconds { 0.0 };
    double sampleRate { 44100.0 };

    juce::AudioBuffer<float> sourceAudioBuffer;
    juce::Path sourceWaveformPath;
    juce::Image backgroundCache;

    int activeHandleId { -1 };
    int hoveredHandleId { -1 };
    bool hasStartedScrubbing { false };
    juce::Point<double> currentDragPoint;

    const std::vector<Handle>* handles = nullptr;

    std::vector<double> cursorPositions;
    std::vector<std::pair<double, double>> unusedRegions;
    double playheadPositionSeconds { -1.0 };
    bool playheadIsLeading { false };  // Source is following by default

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SourceOverviewComponent)
};
