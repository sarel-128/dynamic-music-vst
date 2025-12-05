/* TimelineComponent.cpp - Restored */
#include "TimelineComponent.h"
#include "PluginEditor.h"
#include <cmath> // For std::pow

TimelineComponent::TimelineComponent()
{
    setWantsKeyboardFocus(true);
}

TimelineComponent::~TimelineComponent() {}

void TimelineComponent::setSourceAudio(const juce::AudioBuffer<float>& newSource, double newSampleRate)
{
    sourceAudioBuffer = newSource;
    manipulatedAudioBuffer = newSource;
    sampleRate = newSampleRate;
    zoomFactor = 1.0;
    viewStartSeconds = 0.0;
    generateWaveformPath(sourceAudioBuffer, sourceWaveformPath);
    manipulatedWaveformPath = sourceWaveformPath;
    updateBackgroundCache();
    repaint();
}

void TimelineComponent::setHandles(const std::vector<Handle>* handlesToDisplay)
{
    handles = handlesToDisplay;
}

void TimelineComponent::setCuts(const std::vector<CutInfo>& newCuts)
{
    cuts = newCuts;
    repaint();
}

double TimelineComponent::getTotalSourceDuration() const
{
    if (sourceAudioBuffer.getNumSamples() > 0 && sampleRate > 0)
        return (double)sourceAudioBuffer.getNumSamples() / sampleRate;
    return 1.0;
}

double TimelineComponent::getVisibleDuration() const { return getTotalSourceDuration() / zoomFactor; }
float TimelineComponent::timeToX(double time) const { return (float)getWidth() * (float)((time - viewStartSeconds) / getVisibleDuration()); }
double TimelineComponent::xToTime(float x) const { return (x / (float)getWidth()) * getVisibleDuration() + viewStartSeconds; }

void TimelineComponent::mouseEnter(const juce::MouseEvent& event) { grabKeyboardFocus(); }
void TimelineComponent::mouseExit(const juce::MouseEvent& event) 
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

void TimelineComponent::mouseDown(const juce::MouseEvent& event)
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
        currentState = UIState::DraggingHandle;
        // Don't change cursor - let OS handle the drag state
    }
    else
    {
        // Ready to scrub/seek from wherever we clicked
        currentState = UIState::Scrubbing;
        hasStartedScrubbing = false;  // Will be set to true on first drag
        // Don't call onScrubStart() yet - wait until actual drag to stop playback
        // Just update position for visual feedback
        if (onScrubRequest)
            onScrubRequest(xToTime(event.x));
    }
    repaint();
}

void TimelineComponent::mouseDrag(const juce::MouseEvent& event)
{
    if (currentState == UIState::DraggingHandle && activeHandleId != -1)
    {
        double newTime = xToTime(event.x);
        currentDragPoint = { newTime, (double)event.y };
        
        // Trigger real-time retargeting during drag
        if (onHandleMoved)
        {
            onHandleMoved(activeHandleId, newTime);
        }
        
        repaint(); // Repaint to show drag preview
        if (onViewChanged) onViewChanged(); // Notify parent to update connection lines
    }
    else if (currentState == UIState::Scrubbing)
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
}

void TimelineComponent::mouseMove(const juce::MouseEvent& event)
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
    if (currentState == UIState::Scrubbing || currentState == UIState::DraggingHandle)
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

void TimelineComponent::setCursorPosition(const std::vector<double>& times)
{
    cursorPositions = times;
    repaint();
}

void TimelineComponent::setPlayheadPosition(double time)
{
    playheadPositionSeconds = time;
    repaint();
}

void TimelineComponent::setPlayheadIsLeading(bool isLeading)
{
    if (playheadIsLeading != isLeading)
    {
        playheadIsLeading = isLeading;
        repaint();
    }
}

void TimelineComponent::setOutdatedRange(juce::Range<double> range)
{
    if (outdatedRange != range)
    {
        outdatedRange = range;
        repaint();
    }
}

void TimelineComponent::setHoveredHandle(int handleId)
{
    if (hoveredHandleId != handleId)
    {
        hoveredHandleId = handleId;
        repaint();
    }
}

void TimelineComponent::mouseUp(const juce::MouseEvent& event)
{
    if (currentState == UIState::DraggingHandle && activeHandleId != -1)
    {
        if (onHandleMoved)
        {
            onHandleMoved(activeHandleId, xToTime(event.x));
        }
    }
    else if (currentState == UIState::Scrubbing)
    {
        // After scrubbing (or clicking), seek the main transport to the final position.
        if (onSeekRequest)
            onSeekRequest(xToTime(event.x));
        
        // Only call onScrubEnd if we actually started scrubbing (dragged)
        if (hasStartedScrubbing && onScrubEnd)
            onScrubEnd();
    }
    currentState = UIState::Idle;
    hasStartedScrubbing = false;
    activeHandleId = -1;
    
    // Update cursor based on what's under the mouse after release
    int handleId = getHandleIdAt(event.getPosition().toFloat());
    if (handleId != -1)
        setMouseCursor(juce::MouseCursor::DraggingHandCursor); // Open hand when hovering
    else
        setMouseCursor(juce::MouseCursor::NormalCursor);
    
    repaint();
}


void TimelineComponent::mouseDoubleClick(const juce::MouseEvent& event)
{
    if (onHandleAddRequest)
    {
        onHandleAddRequest(xToTime(event.x));
    }
}

void TimelineComponent::mouseWheelMove(const juce::MouseEvent& event, const juce::MouseWheelDetails& wheel)
{
    if (sourceAudioBuffer.getNumSamples() == 0) return;
    double scrollAmount = -wheel.deltaX;
    if (std::abs(wheel.deltaY) > std::abs(wheel.deltaX)) scrollAmount = -wheel.deltaY;
    viewStartSeconds += scrollAmount * getVisibleDuration() * 0.2;
    double maxViewStart = getTotalSourceDuration() - getVisibleDuration();
    if (maxViewStart < 0) maxViewStart = 0;
    viewStartSeconds = juce::jlimit(0.0, maxViewStart, viewStartSeconds);
    generateWaveformPath(sourceAudioBuffer, sourceWaveformPath);
    generateWaveformPath(manipulatedAudioBuffer, manipulatedWaveformPath);
    updateBackgroundCache();
    repaint();
    if (onViewChanged) onViewChanged(); // Notify parent to update connection lines
}

juce::Point<float> TimelineComponent::getScreenPositionForHandle(int handleId) const
{
    if (handles)
    {
        for (const auto& handle : *handles)
            if (handle.id == handleId)
                return { timeToX(handle.destinationTime), (float)getHeight() / 2.0f };
    }
    return {};
}

int TimelineComponent::getHandleIdAt(juce::Point<float> localPoint) const
{
    if (handles)
    {
        for (const auto& handle : *handles)
        {
            float handleX = timeToX(handle.destinationTime);
            if (std::abs(localPoint.x - handleX) < 8.0f)
                return handle.id;
        }
    }
    return -1;
}

void TimelineComponent::mouseMagnify(const juce::MouseEvent& event, float scaleFactor)
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
    generateWaveformPath(sourceAudioBuffer, sourceWaveformPath);
    generateWaveformPath(manipulatedAudioBuffer, manipulatedWaveformPath);
    updateBackgroundCache();
    repaint();
    if (onViewChanged) onViewChanged(); // Notify parent to update connection lines
}

void TimelineComponent::generateWaveformPath(const juce::AudioBuffer<float>& buffer, juce::Path& path)
{
    path.clear();
    if (buffer.getNumSamples() == 0) return;

    auto bounds = getLocalBounds().toFloat();
    auto numSamples = buffer.getNumSamples();
    auto const* data = buffer.getReadPointer(0);
    double bufferDuration = (double)numSamples / sampleRate;

    auto pathBounds = bounds.reduced(5.0f);
    for (int x = 0; x < pathBounds.getWidth(); ++x)
    {
        auto t1 = xToTime(pathBounds.getX() + x);
        auto t2 = xToTime(pathBounds.getX() + x + 1);
        
        // CLIP: Stop drawing if we go past the actual duration of this buffer
        if (t1 > bufferDuration)
        {
            // Draw flat line for silence
            path.lineTo(pathBounds.getX() + x, pathBounds.getCentreY());
            continue;
        }

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
        
        if (x == 0) path.startNewSubPath(pathBounds.getX() + x, y1);
        else path.lineTo(pathBounds.getX() + x, y1);
        path.lineTo(pathBounds.getX() + x, y2);
    }
}

void TimelineComponent::updateBackgroundCache()
{
    backgroundCache = juce::Image(juce::Image::ARGB, getWidth(), getHeight(), true);
    juce::Graphics g(backgroundCache);

    g.fillAll(juce::Colours::darkslategrey);
    g.setColour(juce::Colours::black.withAlpha(0.5f));
    g.setFont(12.0f);

    double visible = getVisibleDuration();
    double interval = 1.0;
    if (visible > 600) interval = 60.0; else if (visible > 120) interval = 10.0;
    else if (visible > 30) interval = 5.0; else if (visible < 5) interval = 0.1;
    else if (visible < 1) interval = 0.01;
    
    for (double t = std::ceil(viewStartSeconds/interval)*interval; t < viewStartSeconds + visible; t += interval)
    {
        auto x = timeToX(t);
        g.drawVerticalLine((int)x, 0.0f, (float)getHeight());
        juce::String label = juce::String(t, 2) + "s";
        g.drawText(label, (int)x + 3, 3, 100, 14, juce::Justification::left, false);
    }
}

void TimelineComponent::paint(juce::Graphics& g)
{
    if (backgroundCache.isValid())
        g.drawImageAt(backgroundCache, 0, 0);
    else
    {
        g.fillAll(juce::Colours::darkgrey);
        g.setColour(juce::Colours::white);
        g.drawFittedText("Load an audio file", getLocalBounds(), juce::Justification::centred, 1);
    }

    // Waveform
    if (outdatedRange.isEmpty())
    {
        g.setColour(juce::Colours::orange);
        g.strokePath(manipulatedWaveformPath, juce::PathStrokeType(1.5f));
    }
    else
    {
        float x1 = timeToX(outdatedRange.getStart());
        float x2 = timeToX(outdatedRange.getEnd());
        float w = x2 - x1;
        
        // Draw normal parts (excluding outdated range)
        {
            juce::Graphics::ScopedSaveState save(g);
            g.excludeClipRegion(juce::Rectangle<int>((int)x1, 0, (int)std::ceil(w), getHeight()));
            g.setColour(juce::Colours::orange);
            g.strokePath(manipulatedWaveformPath, juce::PathStrokeType(1.5f));
        }
        
        // Draw outdated part (dimmed)
        {
            juce::Graphics::ScopedSaveState save(g);
            g.reduceClipRegion(juce::Rectangle<int>((int)x1, 0, (int)std::ceil(w), getHeight()));
            g.setColour(juce::Colours::orange.darker(0.2f).withAlpha(1.0f));
            g.strokePath(manipulatedWaveformPath, juce::PathStrokeType(1.5f));
        }
    }

    // Draw cut indicators (zigzag lines)
    if (!cuts.empty())
    {
        for (const auto& cut : cuts)
        {
            float x = timeToX(cut.targetTime);
            
            // Color based on quality, using the same logic as the research UI's SimilarityMatrixComponent
            // 1. Normalize similarity score
            float normalizedSim = juce::jlimit(0.0f, 1.0f, cut.quality);
            
            // 2. Apply power function to emphasize good vs. bad cuts. Reduced from 4.0 to 2.0 for better green range.
            normalizedSim = std::pow(normalizedSim, 2.0f);
            
            // 3. Map to a hue from Red (0.0) to Green (0.33)
            float hue = normalizedSim * 0.33f; 
            
            juce::Colour cutColor = juce::Colour::fromHSV(hue, 1.0f, 1.0f, 0.8f); // Use 0.8f alpha
            
            // Draw zigzag pattern
            juce::Path zigzag;
            float height = (float)getHeight();
            float zigzagWidth = 2.5f; // Reduced from 4.0f for a smaller zigzag
            float segmentHeight = 8.0f;
            
            zigzag.startNewSubPath(x, 0);
            for (float y = 0; y < height; y += segmentHeight)
            {
                float nextY = juce::jmin(y + segmentHeight, height);
                float nextX = x + ((int)(y / segmentHeight) % 2 == 0 ? zigzagWidth : -zigzagWidth);
                zigzag.lineTo(nextX, nextY);
            }
            
            // Draw black outline by stroking path with a thicker line first
            g.setColour(juce::Colours::black.withAlpha(0.5f));
            g.strokePath(zigzag, juce::PathStrokeType(2.0f)); // Reduced from 3.0f for a thinner outline

            // Draw the colored zigzag on top
            g.setColour(cutColor);
            g.strokePath(zigzag, juce::PathStrokeType(1.0f));
        }
    }

    // Draw handles
    if (handles)
    {
        g.setFont(14.0f);
        for (const auto& handle : *handles)
        {
            auto x = timeToX(handle.destinationTime);
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
    if (currentState == UIState::DraggingHandle && activeHandleId != -1)
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

void TimelineComponent::resized()
{
    if (sourceAudioBuffer.getNumSamples() > 0)
    {
        generateWaveformPath(sourceAudioBuffer, sourceWaveformPath);
        generateWaveformPath(manipulatedAudioBuffer, manipulatedWaveformPath);
        updateBackgroundCache();
    }
}
