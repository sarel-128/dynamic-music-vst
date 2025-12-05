#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_graphics/juce_graphics.h>
#include <vector>
#include <functional>

struct Handle;
class DynamicMusicVstAudioProcessorEditor;

class TimelineComponent : public juce::Component
{
public:
    TimelineComponent();
    ~TimelineComponent() override;

    void paint(juce::Graphics&) override;
    void resized() override;
    void mouseWheelMove(const juce::MouseEvent& event, const juce::MouseWheelDetails& wheel) override;
    void mouseMagnify(const juce::MouseEvent& event, float scaleFactor) override;
    void mouseEnter(const juce::MouseEvent& event) override;
    void mouseExit(const juce::MouseEvent& event) override;
    void mouseDown(const juce::MouseEvent& event) override;
    void mouseDrag(const juce::MouseEvent& event) override;
    void mouseUp(const juce::MouseEvent& event) override;
    void mouseDoubleClick(const juce::MouseEvent& event) override;
    void mouseMove(const juce::MouseEvent& event) override;

    void setSourceAudio(const juce::AudioBuffer<float>& newSource, double newSampleRate);
    void setHandles(const std::vector<Handle>* handlesToDisplay);
    
    struct CutInfo
    {
        double targetTime;
        double sourceTimeFrom;
        double sourceTimeTo;
        float quality;
    };
    void setCuts(const std::vector<CutInfo>& cuts);

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

    void updateManipulatedAudio(const juce::AudioBuffer<float>& newAudio)
    {
        manipulatedAudioBuffer = newAudio;
        generateWaveformPath(manipulatedAudioBuffer, manipulatedWaveformPath);
        outdatedRange = { 0.0, 0.0 }; // New data arrived, clear range
        repaint();
    }

    void setCursorPosition(const std::vector<double>& times);
    void setPlayheadPosition(double time);
    void setPlayheadIsLeading(bool isLeading);
    void setOutdatedRange(juce::Range<double> range);

private:
    void generateWaveformPath(const juce::AudioBuffer<float>& buffer, juce::Path& path);
    void updateBackgroundCache();

    float timeToX(double time) const;
    double xToTime(float x) const;
    double getTotalSourceDuration() const;
    double getVisibleDuration() const;

    // View State
    double zoomFactor { 1.0 };
    double viewStartSeconds { 0.0 };
    double sampleRate { 44100.0 };
    juce::Range<double> outdatedRange { 0.0, 0.0 };

    juce::AudioBuffer<float> sourceAudioBuffer;
    juce::AudioBuffer<float> manipulatedAudioBuffer;

    struct Handle_Internal
    {
        int id;
        double sourceTimeInSeconds;
        double destinationTimeInSeconds;
    };
    std::vector<Handle_Internal> handles_internal;

    struct Stitch
    {
        double timeInSeconds;
        float quality;
    };
    std::vector<Stitch> stitches;

    juce::Path sourceWaveformPath;
    juce::Path manipulatedWaveformPath;

    juce::Image backgroundCache;

    enum class UIState
    {
        Idle,
        DraggingHandle,
        Scrubbing,
        Processing
    };
    UIState currentState { UIState::Idle };

    int activeHandleId { -1 };
    int hoveredHandleId { -1 };
    bool hasStartedScrubbing { false };

    juce::Point<double> dragStartPoint, currentDragPoint;

    const std::vector<Handle>* handles = nullptr;

    std::vector<double> cursorPositions;
    std::vector<CutInfo> cuts;
    double playheadPositionSeconds { -1.0 };
    bool playheadIsLeading { true };  // Timeline is leading by default

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(TimelineComponent)
};
