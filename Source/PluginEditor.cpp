#include "PluginProcessor.h"
#include "PluginEditor.h"

DynamicMusicVstAudioProcessorEditor::DynamicMusicVstAudioProcessorEditor (DynamicMusicVstAudioProcessor& p, juce::AudioProcessorValueTreeState& vts)
    : AudioProcessorEditor (&p), audioProcessor (p), valueTreeState(vts)
{
    setSize (1000, 800);
    setWantsKeyboardFocus(true); // Allow keyboard input
    audioProcessor.addChangeListener(this);

    addAndMakeVisible(sourceOverviewComponent);
    addAndMakeVisible(timelineComponent);

    // Set up callbacks from children to parent editor
    sourceOverviewComponent.onHandleAddRequest = [this](double sourceTime) 
    { 
        auto targetTimes = audioProcessor.getTargetTimesAt(sourceTime);
        if (!targetTimes.empty())
        {
            addHandle(sourceTime, targetTimes[0]); // Use first matching target time
        }
        else
        {
            // Fallback if no mapping exists (shouldn't happen in normal use)
            addHandle(sourceTime, sourceTime);
        }
    };
    sourceOverviewComponent.onHandleMoved = [this](int id, double newTime)
    {
        for (const auto& handle : handles)
            if (handle.id == id)
                moveHandle(id, newTime, handle.destinationTime);
    };
    sourceOverviewComponent.onHandleDeleteRequest = [this](int id)
    {
        deleteHandle(id);
    };
    sourceOverviewComponent.onViewChanged = [this]() { repaint(); }; // Trigger repaint for wiring lines

    timelineComponent.onHandleAddRequest = [this](double destTime) 
    {
        double sourceTime = audioProcessor.getSourceTimeAt(destTime);
        addHandle(sourceTime, destTime); 
    };
    timelineComponent.onHandleMoved = [this](int id, double newTime)
    {
        for (const auto& handle : handles)
            if (handle.id == id)
                moveHandle(id, handle.sourceTime, newTime);
    };
    timelineComponent.onHandleDeleteRequest = [this](int id)
    {
        deleteHandle(id);
    };
    timelineComponent.onViewChanged = [this]() { repaint(); }; // Trigger repaint for wiring lines

    // --- Cursor Synchronization ---
    timelineComponent.onCursorMoved = [this](double time)
    {
        if (time < 0)
        {
             sourceOverviewComponent.setCursorPosition({});
             return;
        }
        // Map target time -> source time
        double sourceTime = audioProcessor.getSourceTimeAt(time);
        sourceOverviewComponent.setCursorPosition({sourceTime});
    };

    sourceOverviewComponent.onCursorMoved = [this](double time)
    {
        if (time < 0)
        {
             timelineComponent.setCursorPosition({});
             return;
        }
        // Map source time -> target times
        std::vector<double> targetTimes = audioProcessor.getTargetTimesAt(time);
        timelineComponent.setCursorPosition(targetTimes);
    };

    // --- Playhead Seeking & Scrubbing ---
    timelineComponent.onSeekRequest = [this](double time) 
    {
        audioProcessor.setPlaybackPosition(time);
    };
    timelineComponent.onScrubStart = [this]() 
    { 
        isScrubbing = true;
        audioProcessor.startScrubbing(); 
    };
    timelineComponent.onScrubRequest = [this](double time) 
    { 
        audioProcessor.triggerScrubSnippet(time);
        // Update playhead position visually
        timelineComponent.setPlayheadPosition(time);
        double sourceTime = audioProcessor.getSourceTimeAt(time);
        sourceOverviewComponent.setPlayheadPosition(sourceTime);
    };
    timelineComponent.onScrubEnd = [this]() 
    { 
        isScrubbing = false;
        audioProcessor.stopScrubbing(); 
    };

    sourceOverviewComponent.onSeekRequest = [this](double time)
    {
        // When seeking from source view, play from source position directly
        // This allows playing gray areas (unused regions)
        audioProcessor.setPlaybackPositionFromSource(time);
    };
    sourceOverviewComponent.onScrubStart = [this]() 
    { 
        isScrubbing = true;
        audioProcessor.startScrubbing(); 
    };
    sourceOverviewComponent.onScrubRequest = [this](double time)
    {
        // Play directly from source audio at the source time
        audioProcessor.triggerScrubSnippetFromSource(time);
        
        // Update playhead position visually
        sourceOverviewComponent.setPlayheadPosition(time);
        
        // Also update timeline playhead to show where this source time maps to
        auto targetTimes = audioProcessor.getTargetTimesAt(time);
        if (!targetTimes.empty())
        {
            timelineComponent.setPlayheadPosition(targetTimes[0]);
        }
    };
    sourceOverviewComponent.onScrubEnd = [this]() 
    { 
        isScrubbing = false;
        audioProcessor.stopScrubbing(); 
    };

    // --- Handle Hovering ---
    auto hoverCallback = [this](int handleId) {
        if (hoveredHandleId != handleId)
        {
            hoveredHandleId = handleId;
            timelineComponent.setHoveredHandle(handleId);
            sourceOverviewComponent.setHoveredHandle(handleId);
            repaint(); // Repaint editor to update connector line
        }
    };
    timelineComponent.onHandleHoverChanged = hoverCallback;
    sourceOverviewComponent.onHandleHoverChanged = hoverCallback;

    // --- Keep essential controls ---

    // Open File Button
    openFileButton.setButtonText("Open Audio File");
    openFileButton.onClick = [this] { openAudioFile(); };
    addAndMakeVisible(openFileButton);
    
    // Play/Stop Buttons
    playButton.setButtonText("Play");
    playButton.onClick = [this] { audioProcessor.startPlayback(); };
    addAndMakeVisible(playButton);

    stopButton.setButtonText("Stop");
    stopButton.onClick = [this] { audioProcessor.stopPlayback(); };
    addAndMakeVisible(stopButton);

    // Status Label
    statusLabel.setText("Ready", juce::dontSendNotification);
    statusLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(statusLabel);
    
    startTimerHz(30);
}

DynamicMusicVstAudioProcessorEditor::~DynamicMusicVstAudioProcessorEditor()
{
    audioProcessor.removeChangeListener(this);
    stopTimer();
}

void DynamicMusicVstAudioProcessorEditor::paint (juce::Graphics& g)
{
    g.fillAll (getLookAndFeel().findColour (juce::ResizableWindow::backgroundColourId));
}

void DynamicMusicVstAudioProcessorEditor::paintOverChildren(juce::Graphics& g)
{
    // Draw wiring diagram
    for (const auto& handle : handles)
    {
        auto topPos = sourceOverviewComponent.getBounds().getPosition().toFloat() + sourceOverviewComponent.getScreenPositionForHandle(handle.id);
        auto bottomPos = timelineComponent.getBounds().getPosition().toFloat() + timelineComponent.getScreenPositionForHandle(handle.id);
        
        topPos.y = sourceOverviewComponent.getBounds().getBottom();
        bottomPos.y = timelineComponent.getBounds().getY();

        juce::Path path;
        path.startNewSubPath(topPos);
        path.cubicTo(topPos.x, topPos.y + 40, bottomPos.x, bottomPos.y - 40, bottomPos.x, bottomPos.y);

        float thickness = (handle.id == hoveredHandleId) ? 2.5f : 1.5f;
        juce::Colour colour = (handle.id == hoveredHandleId) ? juce::Colours::yellow : juce::Colours::lightgrey.withAlpha(0.7f);

        g.setColour(colour);
        g.strokePath(path, juce::PathStrokeType(thickness));
    }
}

bool DynamicMusicVstAudioProcessorEditor::keyPressed(const juce::KeyPress& key)
{
    // Space bar toggles play/pause
    if (key.isKeyCode(juce::KeyPress::spaceKey))
    {
        if (audioProcessor.isPlaying())
        {
            audioProcessor.stopPlayback();
        }
        else
        {
            audioProcessor.startPlayback();
        }
        return true; // Key was handled
    }
    return false; // Key was not handled
}

void DynamicMusicVstAudioProcessorEditor::openAudioFile()
{
    fileChooser = std::make_unique<juce::FileChooser>("Select an audio file...",
                                                       juce::File(),
                                                       "*.wav;*.mp3;*.aiff;*.flac");
    
    auto flags = juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles;
    
    fileChooser->launchAsync(flags, [this](const juce::FileChooser& fc)
    {
        auto file = fc.getResult();
        if (file.existsAsFile())
        {
            audioProcessor.loadAudioFile(file);
            statusLabel.setText("Loaded: " + file.getFileNameWithoutExtension(), juce::dontSendNotification);
            
            // Clear handles when a new file is loaded
            handles.clear();
            nextHandleId = 0;
            addHandle(0, 0); // Add start handle
            addHandle(audioProcessor.getOriginalTotalLengthSecs(), audioProcessor.getOriginalTotalLengthSecs()); // Add end handle
        }
    });
}

void DynamicMusicVstAudioProcessorEditor::timerCallback()
{
    // Don't update playhead position if we're actively scrubbing
    if (isScrubbing)
        return;
    
    double currentPos = audioProcessor.getCurrentPlaybackPosition();
    
    if (audioProcessor.isInSourcePlaybackMode())
    {
        // In source playback mode, currentPos is the source position
        // Source is leading, timeline is following
        sourceOverviewComponent.setPlayheadIsLeading(true);
        timelineComponent.setPlayheadIsLeading(false);
        
        sourceOverviewComponent.setPlayheadPosition(currentPos);
        
        // Map source position to target for timeline playhead
        auto targetTimes = audioProcessor.getTargetTimesAt(currentPos);
        if (!targetTimes.empty())
        {
            timelineComponent.setPlayheadPosition(targetTimes[0]);
        }
    }
    else
    {
        // Normal mode: currentPos is target position
        // Timeline is leading, source is following
        timelineComponent.setPlayheadIsLeading(true);
        sourceOverviewComponent.setPlayheadIsLeading(false);
        
        timelineComponent.setPlayheadPosition(currentPos);
        
        double sourcePos = audioProcessor.getSourceTimeAt(currentPos);
        sourceOverviewComponent.setPlayheadPosition(sourcePos);
    }
}

void DynamicMusicVstAudioProcessorEditor::addHandle(double sourceTime, double destinationTime)
{
    int newId = audioProcessor.addConstraint(sourceTime, destinationTime);
    handles.push_back({newId, sourceTime, destinationTime});
    
    // Sort handles by destination time for correct processing order
    std::sort(handles.begin(), handles.end(), [](const Handle& a, const Handle& b) {
        return a.destinationTime < b.destinationTime;
    });
    
    sourceOverviewComponent.setHandles(&handles);
    timelineComponent.setHandles(&handles);
    
    repaint(); // Repaint editor for wires
    sourceOverviewComponent.repaint();
    timelineComponent.repaint();
    }

void DynamicMusicVstAudioProcessorEditor::moveHandle(int handleId, double newSourceTime, double newDestinationTime)
{
    for (auto& handle : handles)
    {
        if (handle.id == handleId)
        {
            handle.sourceTime = newSourceTime;
            handle.destinationTime = newDestinationTime;
            break;
    }
}

    // Re-sort after moving
    std::sort(handles.begin(), handles.end(), [](const Handle& a, const Handle& b) {
        return a.destinationTime < b.destinationTime;
    });
    
    // Pass updated handles to children for repaint
    sourceOverviewComponent.setHandles(&handles);
    timelineComponent.setHandles(&handles);
    
    // Calculate outdated range (between neighbors)
    double startOutdated = 0.0;
    double endOutdated = 100000.0; // Large value to cover end
    
    for (size_t i = 0; i < handles.size(); ++i)
    {
        if (handles[i].id == handleId)
        {
            if (i > 0)
                startOutdated = handles[i-1].destinationTime;
            
            if (i < handles.size() - 1)
                endOutdated = handles[i+1].destinationTime;
            break;
        }
    }
    
    // Mark timeline as outdated since background retargeting is starting
    timelineComponent.setOutdatedRange(juce::Range<double>(startOutdated, endOutdated));
    
    repaint();
    sourceOverviewComponent.repaint();
    timelineComponent.repaint();

    // Trigger asynchronous call to PluginProcessor
    audioProcessor.moveConstraint(handleId, newSourceTime, newDestinationTime);
    }

void DynamicMusicVstAudioProcessorEditor::deleteHandle(int handleId)
{
    // Remove handle from vector
    handles.erase(
        std::remove_if(handles.begin(), handles.end(),
                      [handleId](const Handle& h) { return h.id == handleId; }),
        handles.end()
    );
    
    // Update components with new handles list
    sourceOverviewComponent.setHandles(&handles);
    timelineComponent.setHandles(&handles);
    
    // Mark entire timeline as outdated (full retarget)
    timelineComponent.setOutdatedRange(juce::Range<double>(0.0, 100000.0));
    
    repaint();
    sourceOverviewComponent.repaint();
    timelineComponent.repaint();
    
    // Trigger asynchronous call to PluginProcessor
    audioProcessor.removeConstraint(handleId);
    }


void DynamicMusicVstAudioProcessorEditor::changeListenerCallback(juce::ChangeBroadcaster *source)
{
    if (source == &audioProcessor)
    {
        // This callback is now for both initial load and retarget completion
        if (audioProcessor.isRetargeted.load())
        {
            // Update the timeline with the new audio
            timelineComponent.updateManipulatedAudio(audioProcessor.getRetargetedAudioBuffer());
            
            // Update visual feedback for cuts and unused regions
            auto cuts = audioProcessor.getCuts();
            std::vector<TimelineComponent::CutInfo> timelineCuts;
            for (const auto& cut : cuts)
            {
                timelineCuts.push_back({cut.targetTime, cut.sourceTimeFrom, cut.sourceTimeTo, cut.quality});
            }
            timelineComponent.setCuts(timelineCuts);
            
            auto unusedRegions = audioProcessor.getUnusedSourceRegions();
            sourceOverviewComponent.setUnusedRegions(unusedRegions);
        }
        else
        {
            // Initial file load
            const auto& buffer = audioProcessor.getAudioBuffer();
            auto sampleRate = audioProcessor.getFileSampleRate();
            sourceOverviewComponent.setSourceAudio(buffer, sampleRate);
            timelineComponent.setSourceAudio(buffer, sampleRate);
        }
    }
}

void DynamicMusicVstAudioProcessorEditor::resized()
{
    auto bounds = getLocalBounds();
    
    auto controlsHeight = 40;
    auto controlsBounds = bounds.removeFromTop(controlsHeight);
    
    juce::FlexBox flexBox;
    flexBox.flexDirection = juce::FlexBox::Direction::column;
    flexBox.items.add(juce::FlexItem(sourceOverviewComponent).withFlex(1.0f).withMargin(juce::FlexItem::Margin(0, 0, 20, 0))); // 20px bottom margin
    flexBox.items.add(juce::FlexItem(timelineComponent).withFlex(1.0f).withMargin(juce::FlexItem::Margin(20, 0, 0, 0))); // 20px top margin
    flexBox.performLayout(bounds);

    juce::FlexBox controlsBox;
    controlsBox.flexDirection = juce::FlexBox::Direction::row;
    controlsBox.justifyContent = juce::FlexBox::JustifyContent::flexStart;
    controlsBox.alignItems = juce::FlexBox::AlignItems::center;
    
    controlsBox.items.add(juce::FlexItem(openFileButton).withWidth(120).withHeight(30).withMargin(5));
    controlsBox.items.add(juce::FlexItem(playButton).withWidth(60).withHeight(30).withMargin(5));
    controlsBox.items.add(juce::FlexItem(stopButton).withWidth(60).withHeight(30).withMargin(5));
    controlsBox.items.add(juce::FlexItem(statusLabel).withFlex(1.0f).withHeight(30).withMargin(5));

    controlsBox.performLayout(controlsBounds.reduced(5));
}
