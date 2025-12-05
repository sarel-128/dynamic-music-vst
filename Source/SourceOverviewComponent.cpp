/* SourceOverviewComponent.cpp - Restored */
#include "SourceOverviewComponent.h"
#include "PluginEditor.h"

SourceOverviewComponent::SourceOverviewComponent()
{
    setWantsKeyboardFocus(true);
}

SourceOverviewComponent::~SourceOverviewComponent() {}

void SourceOverviewComponent::setSourceAudio(const juce::AudioBuffer<float>& newSource, double newSampleRate)
{
    sourceAudioBuffer = newSource;
    sampleRate = newSampleRate;
    zoomFactor = 1.0;
    viewStartSeconds = 0.0;
    generateWaveformPath();
    updateBackgroundCache();
    repaint();
}

void SourceOverviewComponent::setHandles(const std::vector<Handle>* handlesToDisplay)
{
    handles = handlesToDisplay;
}

void SourceOverviewComponent::setUnusedRegions(const std::vector<std::pair<double, double>>& regions)
{
    unusedRegions = regions;
    repaint();
}

double SourceOverviewComponent::getTotalSourceDuration() const
{
    if (sourceAudioBuffer.getNumSamples() > 0 && sampleRate > 0)
        return (double)sourceAudioBuffer.getNumSamples() / sampleRate;
    return 1.0;
}

double SourceOverviewComponent::getVisibleDuration() const
{
    return getTotalSourceDuration() / zoomFactor;
}

float SourceOverviewComponent::timeToX(double time) const
{
    return (float)getWidth() * (float)((time - viewStartSeconds) / getVisibleDuration());
}

double SourceOverviewComponent::xToTime(float x) const
{
    return (x / (float)getWidth()) * getVisibleDuration() + viewStartSeconds;
}

void SourceOverviewComponent::mouseEnter(const juce::MouseEvent& event) { grabKeyboardFocus(); }
void SourceOverviewComponent::mouseExit(const juce::MouseEvent& event) 
{ 
    giveAwayKeyboardFocus(); 
    if (hoveredHandleId != -1)
    {
        if (onHandleHoverChanged)
            onHandleHoverChanged(-1);
    }
    cursorPositions.clear();
    setMouseCursor(juce::MouseCursor::NormalCursor);
    repaint();
    if (onCursorMoved) onCursorMoved(-1.0);
}
void SourceOverviewComponent::mouseDown(const juce::MouseEvent& event)
{
    grabKeyboardFocus();
    
    // Clear hover cursor when starting any drag operation
    cursorPositions.clear();
    if (onCursorMoved)
        onCursorMoved(-1.0);
    
    int handleId = getHandleIdAt(event.getPosition().toFloat());
    
    // Right-click on a handle: delete it
    if (event.mods.isRightButtonDown() && handleId != -1)
    {
        if (onHandleDeleteRequest)
            onHandleDeleteRequest(handleId);
        return;
    }
    
    if (handleId != -1)
    {
        activeHandleId = handleId;
        // Don't change cursor - let it stay as DraggingHandCursor
    }
    else
    {
        // Ready to scrub/seek from wherever we clicked
        activeHandleId = -2; // Use -2 to indicate scrubbing playhead
        hasStartedScrubbing = false;  // Will be set to true on first drag
        // Don't call onScrubStart() yet - wait until actual drag to stop playback
        // Just update position for visual feedback
        if (onScrubRequest)
            onScrubRequest(xToTime(event.x));
    }
    repaint();
}

void SourceOverviewComponent::mouseDrag(const juce::MouseEvent& event)
{
    if (activeHandleId == -2) // Scrubbing the playhead
    {
        // On first actual drag, call onScrubStart to stop playback
        if (!hasStartedScrubbing)
        {
            hasStartedScrubbing = true;
            if (onScrubStart)
                onScrubStart();
        }
        if (onScrubRequest)
            onScrubRequest(xToTime(event.x));
    }
    else if (activeHandleId != -1) // Dragging a handle
    {
        double newTime = xToTime(event.x);
        currentDragPoint = { newTime, (double)event.y };
        
        // Trigger real-time retargeting during drag
        if (onHandleMoved)
        {
            onHandleMoved(activeHandleId, newTime);
        }
        
        repaint();
        if (onViewChanged) onViewChanged(); // Notify parent to update connection lines
    }
}

void SourceOverviewComponent::mouseMove(const juce::MouseEvent& event)
{
    if (sourceAudioBuffer.getNumSamples() == 0) return;

    // Check if hovering over a handle
    int handleId = getHandleIdAt(event.getPosition().toFloat());
    if (handleId != hoveredHandleId)
    {
        if (onHandleHoverChanged)
            onHandleHoverChanged(handleId);
    }
    
    // Change cursor to hand when over a handle
    if (handleId != -1)
    {
        setMouseCursor(juce::MouseCursor::DraggingHandCursor); // Open hand with 5 fingers
    }
    else
    {
        setMouseCursor(juce::MouseCursor::NormalCursor);
    }

    // Don't show hover cursor while dragging/scrubbing
    if (activeHandleId != -1)
        return;

    // Update cursor
    double time = xToTime(event.x);
    
    // Notify parent
    if (onCursorMoved)
        onCursorMoved(time);
        
    // Self-update (show local cursor)
    cursorPositions.clear();
    cursorPositions.push_back(time);
    repaint();
}

void SourceOverviewComponent::setCursorPosition(const std::vector<double>& times)
{
    cursorPositions = times;
    repaint();
}

void SourceOverviewComponent::setPlayheadPosition(double time)
{
    playheadPositionSeconds = time;
    repaint();
}

void SourceOverviewComponent::setPlayheadIsLeading(bool isLeading)
{
    if (playheadIsLeading != isLeading)
    {
        playheadIsLeading = isLeading;
        repaint();
    }
}

void SourceOverviewComponent::setHoveredHandle(int handleId)
{
    if (hoveredHandleId != handleId)
    {
        hoveredHandleId = handleId;
        repaint();
    }
}

void SourceOverviewComponent::mouseUp(const juce::MouseEvent& event)
{
    if (activeHandleId == -2) // Was scrubbing the playhead
    {
        // After scrubbing, seek the main transport to the final position
        if (onSeekRequest)
            onSeekRequest(xToTime(event.x));
        
        // Only call onScrubEnd if we actually started scrubbing (dragged)
        if (hasStartedScrubbing && onScrubEnd)
            onScrubEnd();
    }
    else if (activeHandleId != -1) // Was dragging a handle
    {
        if (onHandleMoved)
        {
            onHandleMoved(activeHandleId, xToTime(event.x));
        }
    }
    
    activeHandleId = -1;
    hasStartedScrubbing = false;
    
    // Update cursor based on what's under the mouse after release
    int handleId = getHandleIdAt(event.getPosition().toFloat());
    if (handleId != -1)
        setMouseCursor(juce::MouseCursor::DraggingHandCursor); // Open hand when hovering
    else
        setMouseCursor(juce::MouseCursor::NormalCursor);
    
    repaint();
}

void SourceOverviewComponent::mouseDoubleClick(const juce::MouseEvent& event)
{
    if (onHandleAddRequest)
    {
        onHandleAddRequest(xToTime(event.x));
    }
}

void SourceOverviewComponent::mouseWheelMove(const juce::MouseEvent& event, const juce::MouseWheelDetails& wheel)
{
    if (sourceAudioBuffer.getNumSamples() == 0) return;

    double scrollAmount = -wheel.deltaX;
    if (std::abs(wheel.deltaY) > std::abs(wheel.deltaX))
        scrollAmount = -wheel.deltaY;
    
    viewStartSeconds += scrollAmount * getVisibleDuration() * 0.2;

    double maxViewStart = getTotalSourceDuration() - getVisibleDuration();
    if (maxViewStart < 0) maxViewStart = 0;
    viewStartSeconds = juce::jlimit(0.0, maxViewStart, viewStartSeconds);

    generateWaveformPath();
    updateBackgroundCache();
    repaint();
    if (onViewChanged) onViewChanged(); // Notify parent to update connection lines
}

juce::Point<float> SourceOverviewComponent::getScreenPositionForHandle(int handleId) const
{
    if (handles)
    {
        for (const auto& handle : *handles)
        {
            if (handle.id == handleId)
            {
                return { timeToX(handle.sourceTime), (float)getHeight() / 2.0f };
            }
        }
    }
    return {};
}

int SourceOverviewComponent::getHandleIdAt(juce::Point<float> localPoint) const
{
    if (handles)
    {
        for (const auto& handle : *handles)
        {
            float handleX = timeToX(handle.sourceTime);
            if (std::abs(localPoint.x - handleX) < 8.0f) // 8-pixel click radius
            {
                return handle.id;
            }
        }
    }
    return -1;
}


void SourceOverviewComponent::mouseMagnify(const juce::MouseEvent& event, float scaleFactor)
{
    if (sourceAudioBuffer.getNumSamples() == 0) return;

    auto timeAtMouseBeforeZoom = xToTime(event.x);
    zoomFactor *= scaleFactor;
    zoomFactor = juce::jlimit(1.0, 200.0, zoomFactor);
    auto timeAtMouseAfterZoom = xToTime(event.x);
    viewStartSeconds += timeAtMouseBeforeZoom - timeAtMouseAfterZoom;

    double maxViewStart = getTotalSourceDuration() - getVisibleDuration();
    if (maxViewStart < 0) maxViewStart = 0;
    viewStartSeconds = juce::jlimit(0.0, maxViewStart, viewStartSeconds);

    generateWaveformPath();
    updateBackgroundCache();
    repaint();
    if (onViewChanged) onViewChanged(); // Notify parent to update connection lines
}

void SourceOverviewComponent::generateWaveformPath()
{
    sourceWaveformPath.clear();
    if (sourceAudioBuffer.getNumSamples() == 0) return;

    auto bounds = getLocalBounds().toFloat();
    auto numSamples = sourceAudioBuffer.getNumSamples();
    auto const* data = sourceAudioBuffer.getReadPointer(0);
    
    auto pathBounds = bounds.reduced(5.0f);

    for (int x = 0; x < pathBounds.getWidth(); ++x)
    {
        auto t1 = xToTime(pathBounds.getX() + x);
        auto t2 = xToTime(pathBounds.getX() + x + 1);
        auto s1 = juce::jlimit(0, numSamples, (int)(t1 * sampleRate));
        auto s2 = juce::jlimit(0, numSamples, (int)(t2 * sampleRate));
        
        float min = 0.0f, max = 0.0f;
        if (s1 < s2)
        {
            min = juce::FloatVectorOperations::findMinimum(data + s1, s2 - s1);
            max = juce::FloatVectorOperations::findMaximum(data + s1, s2 - s1);
        }

        auto y1 = juce::jmap(max, -1.0f, 1.0f, pathBounds.getBottom(), pathBounds.getY());
        auto y2 = juce::jmap(min, -1.0f, 1.0f, pathBounds.getBottom(), pathBounds.getY());

        if (x == 0) sourceWaveformPath.startNewSubPath(pathBounds.getX() + x, y1);
        else sourceWaveformPath.lineTo(pathBounds.getX() + x, y1);
        sourceWaveformPath.lineTo(pathBounds.getX() + x, y2);
    }
}

void SourceOverviewComponent::updateBackgroundCache()
{
    backgroundCache = juce::Image(juce::Image::ARGB, getWidth(), getHeight(), true);
    juce::Graphics g(backgroundCache);

    g.fillAll(juce::Colours::darkslategrey.darker());
    g.setColour(juce::Colours::black.withAlpha(0.5f));
    
    double interval = 1.0;
    double visible = getVisibleDuration();
    if (visible > 600) interval = 60.0; else if (visible > 120) interval = 10.0;
    else if (visible > 30) interval = 5.0; else if (visible < 5) interval = 0.1;
    else if (visible < 1) interval = 0.01;

    for (double t = std::ceil(viewStartSeconds/interval)*interval; t < viewStartSeconds + visible; t += interval)
        g.drawVerticalLine((int)timeToX(t), 0.0f, (float)getHeight());
}

void SourceOverviewComponent::paint(juce::Graphics& g)
{
    if (backgroundCache.isValid())
        g.drawImageAt(backgroundCache, 0, 0);
    else
    {
        g.fillAll(juce::Colours::darkgrey);
        g.setColour(juce::Colours::white);
        g.drawFittedText("Source Overview", getLocalBounds(), juce::Justification::centred, 1);
    }

    // Draw waveform with color coding for used/unused regions
    // First, draw the entire waveform in blue (used color)
    g.setColour(juce::Colours::cornflowerblue);
    g.strokePath(sourceWaveformPath, juce::PathStrokeType(1.5f));
    
    // Then, overdraw unused regions in gray
    if (!unusedRegions.empty())
    {
        for (const auto& region : unusedRegions)
        {
            float x1 = timeToX(region.first);
            float x2 = timeToX(region.second);
            
            // Create a clip region for this unused section
            g.saveState();
            g.reduceClipRegion(juce::Rectangle<int>((int)x1, 0, (int)(x2 - x1 + 1), getHeight()));
            
            // Draw the waveform in gray within this clip region
            g.setColour(juce::Colours::grey.withAlpha(0.5f));
            g.strokePath(sourceWaveformPath, juce::PathStrokeType(1.5f));
            
            g.restoreState();
        }
    }

    // Draw handles
    if (handles)
    {
        g.setFont(14.0f);
        for (const auto& handle : *handles)
        {
            auto x = timeToX(handle.sourceTime);
            bool isHovered = (handle.id == hoveredHandleId);
            
            // Draw handle line - brighter and thicker when hovered
            if (isHovered)
            {
                g.setColour(juce::Colours::white);
                g.drawLine(x, 0.0f, x, (float)getHeight(), 2.5f);
            }
            else
            {
                g.setColour(juce::Colours::yellow);
                g.drawLine(x, 0.0f, x, (float)getHeight(), 1.5f);
            }
            
            g.setColour(juce::Colours::black);
            g.drawText(juce::String(handle.id), (int)x + 3, 2, 100, 20, juce::Justification::left);
        }
    }
    
    // Draw drag preview
    if (activeHandleId != -1)
    {
        g.setColour(juce::Colours::yellow.withAlpha(0.5f));
        g.drawVerticalLine((int)timeToX(currentDragPoint.x), 0, getHeight());
    }

    // Draw Cursors
    if (!cursorPositions.empty())
    {
        g.setColour(juce::Colours::white.withAlpha(0.8f));
        for (double t : cursorPositions)
        {
            int x = (int)timeToX(t);
            g.drawVerticalLine(x, 0, getHeight());
        }
    }

    // Draw Playhead
    if (playheadPositionSeconds >= 0.0)
    {
        float x = timeToX(playheadPositionSeconds);
        g.setColour(juce::Colours::red);
        
        // Use thicker line when this playhead is leading
        if (playheadIsLeading)
        {
            // Leading playhead: 2 pixels thick
            g.fillRect(x - 1.0f, 0.0f, 2.0f, (float)getHeight());
        }
        else
        {
            // Following playhead: 1 pixel thick
            g.drawVerticalLine((int)x, 0, getHeight());
        }
    }
}

void SourceOverviewComponent::resized()
{
    if (sourceAudioBuffer.getNumSamples() > 0)
    {
        generateWaveformPath();
        updateBackgroundCache();
    }
}
