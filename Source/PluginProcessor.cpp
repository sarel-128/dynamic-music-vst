#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <set>

juce::AudioProcessorValueTreeState::ParameterLayout DynamicMusicVstAudioProcessor::createParameterLayout()
{
    juce::AudioProcessorValueTreeState::ParameterLayout layout;

    layout.add(std::make_unique<juce::AudioParameterFloat>("targetDuration",
                                                          "Target Duration",
                                                          0.1f,
                                                          200.0f,
                                                          5.0f));
    
    layout.add(std::make_unique<juce::AudioParameterFloat>("beatTightness",
                                                          "Beat Tightness",
                                                          1.0f,
                                                          200.0f,
                                                          100.0f));
                                                          
    layout.add(std::make_unique<juce::AudioParameterBool>("showSimilarityMatrix",
                                                         "Show Similarity Matrix",
                                                         true));

    layout.add(std::make_unique<juce::AudioParameterFloat>("trimStart",
                                                          "Trim Start",
                                                          0.0f,
                                                          1.0f,
                                                          0.0f));

    layout.add(std::make_unique<juce::AudioParameterFloat>("trimEnd",
                                                          "Trim End",
                                                          0.0f,
                                                          1.0f,
                                                          1.0f));
    return layout;
}

DynamicMusicVstAudioProcessor::DynamicMusicVstAudioProcessor()
#ifndef JucePlugin_PreferredChannelConfigurations
     : AudioProcessor (BusesProperties()
                     #if ! JucePlugin_IsMidiEffect
                      #if ! JucePlugin_IsSynth
                       .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
                      #endif
                       .withOutput ("Output", juce::AudioChannelSet::stereo(), true)
                     #endif
                        ),
#endif
    parameters(*this, nullptr, juce::Identifier("DynamicMusicVst"), createParameterLayout()),
    audioThumbnail(512, formatManager, thumbnailCache)
{
    formatManager.registerBasicFormats();
    audioThumbnail.addChangeListener(this);
    
    // Start background retargeting thread
    retargetThread = std::make_unique<RetargetThread>(*this);
    retargetThread->startThread();
}

DynamicMusicVstAudioProcessor::~DynamicMusicVstAudioProcessor()
{
    retargetThread->stopThread(2000);
    audioThumbnail.removeChangeListener(this);
}

const juce::String DynamicMusicVstAudioProcessor::getName() const
{
    return JucePlugin_Name;
}

bool DynamicMusicVstAudioProcessor::acceptsMidi() const
{
   #if JucePlugin_WantsMidiInput
    return true;
   #else
    return false;
   #endif
}

bool DynamicMusicVstAudioProcessor::producesMidi() const
{
   #if JucePlugin_ProducesMidiOutput
    return true;
   #else
    return false;
   #endif
}

bool DynamicMusicVstAudioProcessor::isMidiEffect() const
{
   #if JucePlugin_IsMidiEffect
    return true;
   #else
    return false;
   #endif
}

double DynamicMusicVstAudioProcessor::getTailLengthSeconds() const
{
    return 0.0;
}

int DynamicMusicVstAudioProcessor::getNumPrograms()
{
    return 1;
}

int DynamicMusicVstAudioProcessor::getCurrentProgram()
{
    return 0;
}

void DynamicMusicVstAudioProcessor::setCurrentProgram (int index)
{
}

const juce::String DynamicMusicVstAudioProcessor::getProgramName (int index)
{
    return {};
}

void DynamicMusicVstAudioProcessor::changeProgramName (int index, const juce::String& newName)
{
}

void DynamicMusicVstAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    hostSampleRate = sampleRate;
    hostSamplesPerBlock = samplesPerBlock;
    
    sourceAudioBuffer.setSize(getTotalNumInputChannels(), (int)(sampleRate * 30)); // Pre-allocate for up to 30s of audio
    sourceAudioBuffer.clear();

    const juce::ScopedLock lock (transportSourceLock);
    fileTransportSource.prepareToPlay(samplesPerBlock, sampleRate);
    retargetedTransportSource.prepareToPlay(samplesPerBlock, sampleRate);

    // --- Metronome Click Setup ---
    clickAngle = 0.0;
    clickAngleDelta = 2.0 * juce::MathConstants<double>::pi * clickFrequency / sampleRate;
    clickSamplesRemaining = 0;
}

void DynamicMusicVstAudioProcessor::loadAudioFile(const juce::File& file)
{
    const juce::ScopedLock lock (transportSourceLock);
    auto* reader = formatManager.createReaderFor(file);
    if (reader != nullptr)
    {
        double hostSampleRate = getSampleRate();
        if (hostSampleRate <= 0.0) hostSampleRate = 44100.0;

        // Read original file into a temp buffer first
        juce::AudioBuffer<float> tempBuffer;
        tempBuffer.setSize((int)reader->numChannels, (int)reader->lengthInSamples);
        reader->read(&tempBuffer,
                     0,
                     (int)reader->lengthInSamples,
                     0,
                     true,
                     true);

        // Check if resampling is needed
        if (std::abs(reader->sampleRate - hostSampleRate) > 1.0)
        {
            // Resample to host rate
            double ratio = reader->sampleRate / hostSampleRate;
            int newLength = (int) (reader->lengthInSamples / ratio) + 1024; // buffer safety
            
            sourceAudioBuffer.setSize((int)reader->numChannels, newLength);
            sourceAudioBuffer.clear();

            // Use ResamplingAudioSource to do the job
            auto memorySource = std::make_unique<juce::MemoryAudioSource>(tempBuffer, false, false);
            juce::ResamplingAudioSource resampler(memorySource.get(), false, (int)reader->numChannels);
            
            resampler.setResamplingRatio(ratio);
            resampler.prepareToPlay(1024, hostSampleRate); 

            juce::AudioSourceChannelInfo info(&sourceAudioBuffer, 0, newLength);
            resampler.getNextAudioBlock(info);
            
            // Trim the buffer to the actual size if we wanted to be precise, but for now this is fine.
            // We should ideally track how many samples were actually written.
            
            fileSampleRate = hostSampleRate;
        }
        else
        {
            // No resampling needed
            sourceAudioBuffer.makeCopyOf(tempBuffer);
            fileSampleRate = reader->sampleRate;
        }

        // Set up the transport source for playback
        auto newSource = std::make_unique<juce::AudioFormatReaderSource>(reader, true);
        fileTransportSource.setSource(newSource.get(), 0, nullptr, reader->sampleRate);
        fileReaderSource.reset(newSource.release());
        audioThumbnail.setSource(new juce::FileInputSource(file));
        isPlayingFile = true;
        nextBeatToPlay = 0;
        
        // --- Trigger Analysis ---
        // With the new interactive UI, analysis should happen automatically on load.
        analysisState = AnalysisState::AnalysisNeeded;
        
        sendChangeMessage(); // Notify listeners (like the editor) that the main buffer has changed.
    }
    isRetargeted = false;
    
    // Also initialize the retargeted buffer and transport source with the original audio
    // so it's playable before any retargeting happens.
    retargetedAudioBuffer.makeCopyOf(sourceAudioBuffer);
    retargetedMemorySource = std::make_unique<juce::MemoryAudioSource>(retargetedAudioBuffer, false);
    retargetedTransportSource.prepareToPlay(hostSamplesPerBlock, hostSampleRate);
    retargetedTransportSource.setSource(retargetedMemorySource.get(), 0, nullptr, fileSampleRate);
    
    sendChangeMessage();
}

void DynamicMusicVstAudioProcessor::startPlayback()
    {
        retargetedTransportSource.start();
}

void DynamicMusicVstAudioProcessor::stopPlayback()
{
    retargetedTransportSource.stop();
    isPlayingFromSource.store(false);  // Clear source playback mode when stopping
}

double DynamicMusicVstAudioProcessor::getCurrentPlaybackPosition() const
{
    // Return source position if in source playback mode
    if (isPlayingFromSource.load())
    {
        return sourcePlaybackPosition.load();
    }
    return retargetedTransportSource.getCurrentPosition();
}

void DynamicMusicVstAudioProcessor::setPlaybackPosition(double newPositionSecs)
{
        // Disable source playback mode when seeking from target view
        isPlayingFromSource.store(false);
        retargetedTransportSource.setPosition(newPositionSecs);
}

void DynamicMusicVstAudioProcessor::setPlaybackPositionFromSource(double sourcePositionSecs)
{
    // Enable source playback mode and set position
    const juce::ScopedLock lock(sourcePlaybackLock);
    isPlayingFromSource.store(true);
    sourcePlaybackPosition.store(sourcePositionSecs);
}

bool DynamicMusicVstAudioProcessor::isPlaying() const
{
    const juce::ScopedLock lock (transportSourceLock);
    if (isRetargeted.load())
        return retargetedTransportSource.isPlaying();
    
    return fileTransportSource.isPlaying();
}

double DynamicMusicVstAudioProcessor::getCurrentPositionSecs() const
{
    // Return the tracked position, which works for both normal and retargeted playback
    return currentPlaybackPositionSecs.load();
}

double DynamicMusicVstAudioProcessor::getTotalLengthSecs() const
{
    const juce::ScopedLock lock (transportSourceLock);
    if (isRetargeted.load())
        return retargetedTransportSource.getLengthInSeconds();

    return fileTransportSource.getLengthInSeconds();
}

double DynamicMusicVstAudioProcessor::getOriginalTotalLengthSecs() const
{
    const juce::ScopedLock lock (transportSourceLock);
    return fileTransportSource.getLengthInSeconds();
}

void DynamicMusicVstAudioProcessor::retargetWithHandles(const std::vector<Handle>& handles)
{
    // TODO: This is where the real audio processing will be triggered.
    // For now, we'll just log the handles to confirm the connection is working.
    DBG("Processor received retarget request with " + juce::String(handles.size()) + " handles.");
    for(const auto& handle : handles)
    {
        DBG("  Handle " + juce::String(handle.id) + ": Source " + juce::String(handle.sourceTime, 2)
            + "s -> Dest " + juce::String(handle.destinationTime, 2) + "s");
    }

    // This would eventually be an async task.
    // After processing, it would update the manipulatedAudioBuffer
    // and call sendChangeMessage() to notify the editor.
}


void DynamicMusicVstAudioProcessor::releaseResources()
{
    sourceAudioBuffer.setSize(0, 0);
}

#ifndef JucePlugin_PreferredChannelConfigurations
bool DynamicMusicVstAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
  #if JucePlugin_IsMidiEffect
    juce::ignoreUnused (layouts);
    return true;
  #else
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
     && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

   #if ! JucePlugin_IsSynth
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;
   #endif

    return true;
  #endif
}
#endif

void DynamicMusicVstAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    auto totalNumInputChannels  = getTotalNumInputChannels();
    auto totalNumOutputChannels = getTotalNumOutputChannels();

    for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear (i, 0, buffer.getNumSamples());
        
    // --- Handle one-shot scrub playback ---
    // The snippetLock here ensures that the UI thread isn't resizing the buffer
    // while the audio thread is trying to read from it.
    const juce::ScopedLock sl(snippetLock);
    auto snippetNumSamples = scrubSnippetBuffer.getNumSamples();
    auto currentReadPos = scrubSnippetReadPos.load();

    if (currentReadPos < snippetNumSamples)
        {
        auto numSamplesToRead = juce::jmin(buffer.getNumSamples(), snippetNumSamples - currentReadPos);
        
        for (int channel = 0; channel < totalNumOutputChannels; ++channel)
            {
            if (channel < scrubSnippetBuffer.getNumChannels())
                {
                // Add the snippet to the output buffer instead of copying to allow main playback to continue underneath if desired
                buffer.addFrom(channel, 0, scrubSnippetBuffer, channel, currentReadPos, numSamplesToRead);
                    }
        }
        
        scrubSnippetReadPos.store(currentReadPos + numSamplesToRead);
                }
    
    // --- Regular playback ---
    // Skip regular playback if we're scrubbing (flag provides instant response)
    if (!isScrubbing.load() && retargetedTransportSource.isPlaying())
                {
        // Check if we're in source playback mode (playing gray areas)
        if (isPlayingFromSource.load())
        {
            // Play directly from source buffer
            const juce::ScopedLock lock(sourcePlaybackLock);
            auto currentPos = sourcePlaybackPosition.load();
            int startSample = static_cast<int>(currentPos * fileSampleRate);
            
            if (startSample >= 0 && startSample < sourceAudioBuffer.getNumSamples())
                {
                int samplesToRead = juce::jmin(buffer.getNumSamples(), 
                                                sourceAudioBuffer.getNumSamples() - startSample);
                
                for (int channel = 0; channel < totalNumOutputChannels; ++channel)
                {
                    if (channel < sourceAudioBuffer.getNumChannels())
                    {
                        buffer.copyFrom(channel, 0, sourceAudioBuffer, channel, startSample, samplesToRead);
                    }
                }
                
                // Update position for next block
                sourcePlaybackPosition.store(currentPos + (samplesToRead / fileSampleRate));
                
                // Check if we've reached the end of the buffer or should switch back to retargeted
                if (startSample + samplesToRead >= sourceAudioBuffer.getNumSamples())
                {
                    isPlayingFromSource.store(false);
                    retargetedTransportSource.stop();
                }
            }
            else
            {
                // Invalid position, switch back to retargeted playback
                isPlayingFromSource.store(false);
            }
        }
        else
        {
            // Normal retargeted playback
            juce::AudioSourceChannelInfo bufferToFill(buffer);
            retargetedTransportSource.getNextAudioBlock(bufferToFill);
        }
    }
        
    // --- Analysis ---
    if (analysisState == AnalysisState::AnalysisNeeded) {
        analysisState = AnalysisState::Analyzing;

        DBG(" ");
        DBG("Starting analysis...");
        auto analysisStartTime = juce::Time::getMillisecondCounterHiRes();
        
        // --- Trim Audio Buffer ---
        auto trimStartNormalized = parameters.getRawParameterValue("trimStart")->load();
        auto trimEndNormalized = parameters.getRawParameterValue("trimEnd")->load();
        
        int totalSamples = sourceAudioBuffer.getNumSamples();
        int startSample = static_cast<int>(trimStartNormalized * totalSamples);
        int endSample = static_cast<int>(trimEndNormalized * totalSamples);
        int numSamplesToProcess = endSample - startSample;

        juce::AudioBuffer<float> analysisBuffer;
        if (numSamplesToProcess > 0)
        {
            analysisBuffer.setSize(sourceAudioBuffer.getNumChannels(), numSamplesToProcess);
            for (int ch = 0; ch < sourceAudioBuffer.getNumChannels(); ++ch)
            {
                analysisBuffer.copyFrom(ch, 0, sourceAudioBuffer, ch, startSample, numSamplesToProcess);
            }
        }
        else
        {
            // If trim range is invalid, use the whole buffer
            analysisBuffer.makeCopyOf(sourceAudioBuffer);
        }

        // --- Real Analysis ---
        auto stepStartTime = juce::Time::getMillisecondCounterHiRes();
        const int hopSize = 512; 
        const int fftSize = 2048;
        onsetEnvelope = audioAnalyser.getOnsetStrengthEnvelope(analysisBuffer, fileSampleRate, hopSize, fftSize);
        DBG("1. Onset Strength Envelope: " << juce::String(juce::Time::getMillisecondCounterHiRes() - stepStartTime, 2) << " ms");

        stepStartTime = juce::Time::getMillisecondCounterHiRes();
        auto onsets = audioAnalyser.detectOnsets(onsetEnvelope, hopSize);
        DBG("2. Detect Onsets: " << juce::String(juce::Time::getMillisecondCounterHiRes() - stepStartTime, 2) << " ms");

        stepStartTime = juce::Time::getMillisecondCounterHiRes();
        estimatedBPM = audioAnalyser.estimateTempo(onsetEnvelope, fileSampleRate, hopSize);
        tempogram = audioAnalyser.getLastTempogram();
        globalAcf = audioAnalyser.getLastGlobalAcf(); // Store aggregated profile for the UI
        DBG("3. Estimate Tempo: " << juce::String(juce::Time::getMillisecondCounterHiRes() - stepStartTime, 2) << " ms");
        
        auto tightnessValue = parameters.getRawParameterValue("beatTightness")->load();
        
        // Beat timestamps are relative to the start of the analysisBuffer
        stepStartTime = juce::Time::getMillisecondCounterHiRes();
        auto relativeBeatTimestamps = audioAnalyser.findBeats(onsetEnvelope, estimatedBPM, fileSampleRate, hopSize, tightnessValue);
        DBG("4. Find Beats: " << juce::String(juce::Time::getMillisecondCounterHiRes() - stepStartTime, 2) << " ms");
        
        // --- Calculate MFCCs and Similarity Matrix ---
        if (!relativeBeatTimestamps.empty())
        {
            stepStartTime = juce::Time::getMillisecondCounterHiRes();
            auto allMfccs = audioAnalyser.calculateMFCCs(analysisBuffer, fileSampleRate, 128, fftSize, hopSize);
            DBG("5. Calculate MFCCs: " << juce::String(juce::Time::getMillisecondCounterHiRes() - stepStartTime, 2) << " ms");

            stepStartTime = juce::Time::getMillisecondCounterHiRes();
            auto beatMfccs = audioAnalyser.calculateBeatMFCCs(allMfccs, relativeBeatTimestamps, fileSampleRate, hopSize);
            DBG("6. Calculate Beat MFCCs: " << juce::String(juce::Time::getMillisecondCounterHiRes() - stepStartTime, 2) << " ms");

            stepStartTime = juce::Time::getMillisecondCounterHiRes();
            similarityMatrix = audioAnalyser.createSimilarityMatrix(beatMfccs);
            DBG("7. Create Similarity Matrix: " << juce::String(juce::Time::getMillisecondCounterHiRes() - stepStartTime, 2) << " ms");
        }
        else
        {
            similarityMatrix.clear();
        }

        // Offset the relative beat timestamps to get the absolute timestamps for playback
        double trimStartTimeSecs = trimStartNormalized * getTotalLengthSecs();
        beatTimestamps = relativeBeatTimestamps;
        for (auto& timestamp : beatTimestamps)
        {
            timestamp += trimStartTimeSecs;
        }

        DBG("Total Analysis Time: " << juce::String(juce::Time::getMillisecondCounterHiRes() - analysisStartTime, 2) << " ms");
        DBG("------------------------------------");

        // We will still calculate the remix map for future use, but won't apply it to the audio for now.
        int numSamples = analysisBuffer.getNumSamples();
        remixMap.clear();
        if (onsets.size() > 1) {
            for (size_t i = 0; i < onsets.size() - 1; ++i) {
                remixMap.push_back({onsets[i], onsets[i+1] - onsets[i]});
            }
        }
        if (remixMap.empty() && numSamples > 0)
        {
            int numChunks = 16;
            int chunkSize = numSamples / numChunks;
            for (int i = 0; i < numChunks; ++i)
            {
                remixMap.push_back({i * chunkSize, chunkSize});
            }
        }
        float currentDuration = (float)analysisBuffer.getNumSamples() / fileSampleRate;
        if (currentDuration > 0) {
            float targetDuration = *parameters.getRawParameterValue("targetDuration");
            if (currentDuration > targetDuration && !remixMap.empty()) {
                int segmentsToRemove = static_cast<int>(remixMap.size() * (1.0f - targetDuration / currentDuration));
                for(int i=0; i<segmentsToRemove; ++i) if (!remixMap.empty()) remixMap.erase(remixMap.begin() + (rand() % remixMap.size()));
            } else if (currentDuration < targetDuration && !remixMap.empty()) {
                int segmentsToAdd = static_cast<int>(remixMap.size() * (targetDuration / currentDuration - 1.0f));
                for(int i=0; i<segmentsToAdd; ++i) if (!remixMap.empty()) remixMap.insert(remixMap.begin() + (rand() % remixMap.size()), remixMap[rand() % remixMap.size()]);
            }
        }

        remixPlaybackPosition = 0;
        analysisState = AnalysisState::AnalysisComplete;
    }

    // --- Retargeting ---
    if (retargetState == RetargetState::RetargetingNeeded)
    {
        retargetState = RetargetState::Retargeting;

        DBG(" ");
        DBG("Starting retargeting...");
        auto retargetingStartTime = juce::Time::getMillisecondCounterHiRes();
        
        // Ensure we have analysis data before retargeting
        if (!similarityMatrix.empty() && !beatTimestamps.empty())
        {
            // Update end constraint based on target duration slider
            float targetDuration = *parameters.getRawParameterValue("targetDuration");
            
            // Find and update the end constraint (if it exists)
            bool hasEndConstraint = false;
            for (auto& uc : userConstraints)
            {
                // The end constraint is typically the one with the highest targetTime
                // and sourceTime close to original length
                if (uc.sourceTime >= getOriginalTotalLengthSecs() * 0.9f)
                {
                    uc.targetTime = targetDuration;
                    hasEndConstraint = true;
                }
            }
            
            // If no constraints exist yet, initialize them
            if (userConstraints.empty())
            {
                initializeDefaultConstraints();
                // Update end constraint to target duration
                for (auto& uc : userConstraints)
                {
                    if (uc.sourceTime >= getOriginalTotalLengthSecs() * 0.9f)
                    {
                        uc.targetTime = targetDuration;
                    }
                }
            }
            
            // Use the new full retarget method
            performFullRetarget();

            // Reset playback position for the new path
            currentRetargetBeatIndex = 0;
            samplesIntoCurrentBeat = 0;
        }

        DBG("Total Retargeting Time: " << juce::String(juce::Time::getMillisecondCounterHiRes() - retargetingStartTime, 2) << " ms");
        DBG("------------------------------------");
        
        retargetState = RetargetState::RetargetingComplete;
    }

    // --- Metronome Click Generation and Mixing ---
    const juce::ScopedLock lock (clickLock);
    if (isPlaying() && !beatTimestamps.empty())
    {
        double blockStartTimeSecs = fileTransportSource.getCurrentPosition();
        double sampleRate = getSampleRate();
        double secsPerSample = 1.0 / sampleRate;
        int clickDurationInSamples = (int)(clickDurationSecs * sampleRate);

        for (int sample = 0; sample < buffer.getNumSamples(); ++sample)
        {
            double currentSampleTime = blockStartTimeSecs + sample * secsPerSample;

            if (nextBeatToPlay < (int)beatTimestamps.size() && currentSampleTime >= beatTimestamps[nextBeatToPlay])
            {
                clickSamplesRemaining = clickDurationInSamples;
                clickAngle = 0.0;
                nextBeatToPlay++;
            }

            if (clickSamplesRemaining > 0)
            {
                auto clickSample = (float)(std::sin(clickAngle) * 0.25); // 0.25 is amplitude
                for (int channel = 0; channel < buffer.getNumChannels(); ++channel)
                {
                    buffer.addSample(channel, sample, clickSample);
                }
                clickAngle += clickAngleDelta;
                clickSamplesRemaining--;
            }
        }
    }
}

bool DynamicMusicVstAudioProcessor::hasEditor() const
{
    return true;
}

juce::AudioProcessorEditor* DynamicMusicVstAudioProcessor::createEditor()
{
    return new DynamicMusicVstAudioProcessorEditor (*this, parameters);
}

void DynamicMusicVstAudioProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    auto state = parameters.copyState();
    std::unique_ptr<juce::XmlElement> xml (state.createXml());
    copyXmlToBinary (*xml, destData);
}

void DynamicMusicVstAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xmlState (getXmlFromBinary (data, sizeInBytes));
    if (xmlState.get() != nullptr)
        if (xmlState->hasTagName (parameters.state.getType()))
            parameters.replaceState (juce::ValueTree::fromXml (*xmlState));
}

double DynamicMusicVstAudioProcessor::getSourceTimeAt(double targetTime) const
{
    // If not retargeted, the mapping is 1:1 (identity)
    if (!isRetargeted.load() || retargetedBeatPath.empty())
        return targetTime;

    // 1. Convert targetTime to step index
    // Note: using currentSecondsPerBeat which was set during retargeting
    int step = static_cast<int>(targetTime / currentSecondsPerBeat);

    // 2. Clamp step to valid range
    if (step < 0) step = 0;
    if (step >= static_cast<int>(retargetedBeatPath.size()))
        step = static_cast<int>(retargetedBeatPath.size()) - 1;

    // 3. Look up source beat index
    int sourceBeatIndex = retargetedBeatPath[step];

    // 4. Convert source beat index to source time
    if (sourceBeatIndex >= 0 && sourceBeatIndex < static_cast<int>(beatTimestamps.size()))
    {
        return beatTimestamps[sourceBeatIndex];
    }

    return 0.0;
}

std::vector<double> DynamicMusicVstAudioProcessor::getTargetTimesAt(double sourceTime) const
{
    std::vector<double> targetTimes;
    
    if (!isRetargeted.load() || retargetedBeatPath.empty() || beatTimestamps.empty())
    {
        targetTimes.push_back(sourceTime);
        return targetTimes;
    }

    // 1. Find closest source beat index
    int bestBeatIdx = 0;
    float minDiff = std::numeric_limits<float>::max();
    for (size_t b = 0; b < beatTimestamps.size(); ++b)
    {
        float diff = std::abs((float)beatTimestamps[b] - (float)sourceTime);
        if (diff < minDiff)
        {
            minDiff = diff;
            bestBeatIdx = static_cast<int>(b);
        }
    }

    // 2. Iterate through path to find all occurrences
    for (size_t step = 0; step < retargetedBeatPath.size(); ++step)
    {
        if (retargetedBeatPath[step] == bestBeatIdx)
        {
            // Convert step to target time
            targetTimes.push_back(step * currentSecondsPerBeat);
        }
    }

    // 3. If the beat wasn't found in the path (it's in a skipped/gray region),
    // find the next beat that IS in the path and return its target time(s).
    // This is the "landing point" of the jump that skipped this region.
    if (targetTimes.empty())
    {
        // Search for the smallest beat index in the path that's greater than bestBeatIdx
        int nextUsedBeat = -1;
        for (int pathBeat : retargetedBeatPath)
        {
            if (pathBeat > bestBeatIdx)
            {
                if (nextUsedBeat == -1 || pathBeat < nextUsedBeat)
                {
                    nextUsedBeat = pathBeat;
                }
            }
        }
        
        // If we found a next beat, return all its occurrences
        if (nextUsedBeat != -1)
        {
            for (size_t step = 0; step < retargetedBeatPath.size(); ++step)
            {
                if (retargetedBeatPath[step] == nextUsedBeat)
                {
                    targetTimes.push_back(step * currentSecondsPerBeat);
                }
            }
        }
        else
        {
            // Fallback: if no next beat exists (shouldn't happen), return the last step
            if (!retargetedBeatPath.empty())
            {
                targetTimes.push_back((retargetedBeatPath.size() - 1) * currentSecondsPerBeat);
            }
        }
    }

    return targetTimes;
}

void DynamicMusicVstAudioProcessor::createRetargetedAudio(const std::vector<int>& path)
{
    if (path.empty() || beatTimestamps.empty())
    {
        isRetargeted = false;
        return;
    }
    
    // 1. Calculate a safe upper bound for total samples for the new buffer
    int totalSamples = 0;
    for (int originalBeatIndex : path)
    {
        double beatStartSecs = beatTimestamps[originalBeatIndex];
        double nextBeatStartSecs;
        if (originalBeatIndex + 1 < beatTimestamps.size())
            nextBeatStartSecs = beatTimestamps[originalBeatIndex + 1];
        else
            nextBeatStartSecs = getOriginalTotalLengthSecs();

        int beatNumSamples = static_cast<int>((nextBeatStartSecs - beatStartSecs) * fileSampleRate);
        if (beatNumSamples > 0)
            totalSamples += beatNumSamples;
    }

    if (totalSamples <= 0)
    {
        isRetargeted = false;
        return;
    }
    
    // 2. Create and fill the new buffer
    retargetedAudioBuffer.setSize(sourceAudioBuffer.getNumChannels(), totalSamples);
    retargetedAudioBuffer.clear();
    
    const int crossfadeSamples = static_cast<int>((crossfadeMs / 1000.0f) * fileSampleRate);

    int currentWritePos = 0;
    for (size_t i = 0; i < path.size(); ++i)
    {
        int originalBeatIndex = path[i];

        double beatStartSecs = beatTimestamps[originalBeatIndex];
        double nextBeatStartSecs;
        if (originalBeatIndex + 1 < beatTimestamps.size())
            nextBeatStartSecs = beatTimestamps[originalBeatIndex + 1];
        else
            nextBeatStartSecs = getOriginalTotalLengthSecs();

        int beatStartSample = static_cast<int>(beatStartSecs * fileSampleRate);
        int beatNumSamples = static_cast<int>((nextBeatStartSecs - beatStartSecs) * fileSampleRate);
        
        if (beatNumSamples <= 0)
            continue;
        
        bool isJump = (i > 0 && path[i] != path[i-1] + 1);

        if (isJump && currentWritePos > 0)
        {
            int overlapSamples = juce::jmin(crossfadeSamples, beatNumSamples, currentWritePos);

            // Rewind write position for the overlap
            currentWritePos -= overlapSamples;
            
            for (int ch = 0; ch < sourceAudioBuffer.getNumChannels(); ++ch)
            {
                float* target = retargetedAudioBuffer.getWritePointer(ch, currentWritePos);
                const float* source = sourceAudioBuffer.getReadPointer(ch, beatStartSample);

                for (int s = 0; s < overlapSamples; ++s)
                {
                    float fadeOut = 1.0f - (float)s / (float)(overlapSamples -1);
                    float fadeIn  = (float)s / (float)(overlapSamples - 1);
                    target[s] = target[s] * fadeOut + source[s] * fadeIn;
                }
            }
            
            int remainingSamples = beatNumSamples - overlapSamples;
            if (remainingSamples > 0)
            {
                for (int ch = 0; ch < sourceAudioBuffer.getNumChannels(); ++ch)
                {
                    retargetedAudioBuffer.copyFrom(ch,
                                                   currentWritePos + overlapSamples,
                                                   sourceAudioBuffer,
                                                   ch,
                                                   beatStartSample + overlapSamples,
                                                   remainingSamples);
                }
            }
            currentWritePos += beatNumSamples;
        }
        else
        {
            for (int ch = 0; ch < sourceAudioBuffer.getNumChannels(); ++ch)
            {
                retargetedAudioBuffer.copyFrom(ch, currentWritePos, sourceAudioBuffer, ch, beatStartSample, beatNumSamples);
            }
            currentWritePos += beatNumSamples;
        }
    }
    
    // 3. Trim buffer to actual size, as crossfading shortens it
    retargetedAudioBuffer.setSize(retargetedAudioBuffer.getNumChannels(), currentWritePos, true);

    // 4. Set up the new transport source
    const juce::ScopedLock lock (transportSourceLock);

    retargetedTransportSource.stop();
    retargetedTransportSource.setSource(nullptr);
    retargetedMemorySource.reset();

    retargetedMemorySource = std::make_unique<juce::MemoryAudioSource>(retargetedAudioBuffer, false);
    retargetedTransportSource.setSource(retargetedMemorySource.get(), 0, nullptr, fileSampleRate);
    
    isRetargeted = true;
    sendChangeMessage(); // Inform editor that new data is available
}

void DynamicMusicVstAudioProcessor::changeListenerCallback (juce::ChangeBroadcaster* source)
{
    // The thumbnail component in the editor is also listening for this,
    // so we don't need to do anything here. This just fulfills the inheritance requirement.
}

// ========== Constraint Management Methods ==========

void DynamicMusicVstAudioProcessor::initializeDefaultConstraints()
{
    userConstraints.clear();
    nextConstraintId = 0;
    
    // Add start constraint (0,0)
    userConstraints.push_back({nextConstraintId++, 0.0f, 0.0f});
    
    // Add end constraint (end of original -> end of original by default)
    float originalLength = static_cast<float>(getOriginalTotalLengthSecs());
    userConstraints.push_back({nextConstraintId++, originalLength, originalLength});
}

int DynamicMusicVstAudioProcessor::addConstraint(float sourceTime, float targetTime)
{
    int newId = nextConstraintId++;
    userConstraints.push_back({newId, sourceTime, targetTime});
    
    // Sort constraints by targetTime for easier neighbor finding
    std::sort(userConstraints.begin(), userConstraints.end(),
              [](const ConstraintPoint& a, const ConstraintPoint& b) {
                  return a.targetTime < b.targetTime;
              });
    
    // Trigger segmented retarget
    performSegmentedRetarget(newId);
    
    return newId;
}

void DynamicMusicVstAudioProcessor::moveConstraint(int id, float newSourceTime, float newTargetTime)
{
    // Special handling for start anchor (id=0): cannot be moved
    if (id == 0) return;
    
    {
        const juce::ScopedLock lock(constraintLock);
        
        for (auto& c : userConstraints)
        {
            if (c.id == id)
            {
                c.sourceTime = newSourceTime;
                c.targetTime = newTargetTime;
                break;
            }
        }
        
        // Re-sort by targetTime
        std::sort(userConstraints.begin(), userConstraints.end(),
                  [](const ConstraintPoint& a, const ConstraintPoint& b) {
                      return a.targetTime < b.targetTime;
                  });
    } // Unlock
    
    // Request background update
    lastChangedConstraintId.store(id);
    isFullRetargetNeeded.store(false);
    retargetNeeded.store(true);
    // retargetThread will pick this up
}

void DynamicMusicVstAudioProcessor::removeConstraint(int id)
{
    // Start and end anchors (id 0 and 1) cannot be removed
    if (id == 0 || id == 1) return;
    
    {
        const juce::ScopedLock lock(constraintLock);
        auto it = std::find_if(userConstraints.begin(), userConstraints.end(),
                               [id](const ConstraintPoint& c) { return c.id == id; });
        
        if (it != userConstraints.end())
        {
            userConstraints.erase(it);
        }
    } // Unlock
    
    // Request full background update
    isFullRetargetNeeded.store(true);
    retargetNeeded.store(true);
}

void DynamicMusicVstAudioProcessor::performSegmentedRetarget(int changedConstraintId)
{
    const juce::ScopedLock lock(constraintLock);
    performRetarget(false, changedConstraintId, userConstraints);
}

void DynamicMusicVstAudioProcessor::performFullRetarget()
{
    // If called from UI, use lock and call performRetarget
    const juce::ScopedLock lock(constraintLock);
    performRetarget(true, -1, userConstraints);
}

void DynamicMusicVstAudioProcessor::performRetarget(bool isFullRetarget, int changedConstraintId, const std::vector<ConstraintPoint>& activeConstraints)
{
    if (similarityMatrix.empty() || beatTimestamps.empty())
        return;
    
    // For segmented retarget, if the path doesn't exist yet, do a full one instead
    if (!isFullRetarget && retargetedBeatPath.empty())
        isFullRetarget = true;

    auto startTime = juce::Time::getMillisecondCounterHiRes();

    // activeConstraints are already sorted by the caller (thread or main)

    int changedIdx = -1;
    if (!isFullRetarget)
    {
        for (size_t i = 0; i < activeConstraints.size(); ++i)
        {
            if (activeConstraints[i].id == changedConstraintId)
            {
                changedIdx = static_cast<int>(i);
                break;
            }
        }
        // If constraint not found, fall back to a full retarget
        if (changedIdx < 0)
            isFullRetarget = true;
    }

    if (isFullRetarget)
    {
        DBG(" ");
        DBG("=== Performing FULL Retarget ===");
    }
    else
    {
        DBG(" ");
        DBG("=== Performing SEGMENTED Retarget for constraint ID " << changedConstraintId << " ===");
    }

    // 1. ==================================================================
    // Setup common variables and lambdas
    
    float secondsPerBeat = 0.5f;
    if (beatTimestamps.size() > 1)
    {
        secondsPerBeat = static_cast<float>((beatTimestamps.back() - beatTimestamps.front()) / (beatTimestamps.size() - 1));
    }
    
    // Update member variable for cursor mapping
    currentSecondsPerBeat = secondsPerBeat;
    
    int numBeats = static_cast<int>(similarityMatrix.size());
    std::vector<float> beatsFloat(beatTimestamps.begin(), beatTimestamps.end());
    
    float similarityPenalty = 0.95f;
    float backwardJumpPenalty = 0.1f;
    float timeContinuityPenalty = 0.1f;
    int beatTolerance = 20;
    
    auto getStepForTime = [secondsPerBeat](float time) -> int {
        return static_cast<int>(time / secondsPerBeat);
    };
    
    auto findClosestBeat = [&beatsFloat, numBeats](float sourceTime) -> int {
        int bestBeatIdx = 0;
        float minDiff = std::numeric_limits<float>::max();
        for (int b = 0; b < numBeats; ++b)
        {
            float diff = std::abs(beatsFloat[b] - sourceTime);
            if (diff < minDiff)
            {
                minDiff = diff;
                bestBeatIdx = b;
            }
        }
        return bestBeatIdx;
    };

    auto getBeatForStep = [&](int step) -> int {
        if (step < 0) return 0;
        if (step >= static_cast<int>(retargetedBeatPath.size()))
            return retargetedBeatPath.empty() ? 0 : retargetedBeatPath.back();
        return retargetedBeatPath[step];
    };

    // 2. ==================================================================
    // Determine processing range and initialize path
    
    int startRangeIdx = 0;
    int endRangeIdx = static_cast<int>(activeConstraints.size()) - 1;
    
    std::vector<int> newPath;
    if (!isFullRetarget)
    {
        // IMPORTANT: When performing a segmented retarget, especially for the end handle,
        // we must ensure the new path vector has the correct size for the NEW target duration.
        // The previous logic was clamping the endStep to the OLD path size, preventing extension,
        // and not shrinking the vector when shortening.
        
        int expectedTotalSteps = getStepForTime(activeConstraints.back().targetTime) + 1;
        newPath = retargetedBeatPath;
        
        if (newPath.size() != expectedTotalSteps)
            newPath.resize(expectedTotalSteps, newPath.empty() ? 0 : newPath.back());
            
        startRangeIdx = (changedIdx > 0) ? changedIdx - 1 : changedIdx;
        endRangeIdx = (changedIdx < static_cast<int>(activeConstraints.size()) - 1) ? changedIdx + 1 : changedIdx;
    }

    // 3. ==================================================================
    // Process segments in the determined range

    int lockedPreviousEndBeat = -1;
    if (!isFullRetarget && startRangeIdx > 0)
    {
        int step = getStepForTime(activeConstraints[startRangeIdx].targetTime);
        lockedPreviousEndBeat = getBeatForStep(step);
    }
    
    std::vector<int> pathSegments; // Used for building path from scratch in full retarget

    for (int i = startRangeIdx; i < endRangeIdx; ++i)
    {
        const auto& startConst = activeConstraints[i];
        const auto& endConst = activeConstraints[i+1];
        
        int startStep = getStepForTime(startConst.targetTime);
        int endStep = getStepForTime(endConst.targetTime);
        
        if (!isFullRetarget)
        {
            // Clamp startStep, but allow endStep to go up to the newly resized boundary
            startStep = juce::jmax(0, startStep);
            endStep = juce::jmin(static_cast<int>(newPath.size()) - 1, endStep);
        }
        else
        {
            startStep = juce::jmax(0, startStep);
        }
        
        if (endStep <= startStep) continue;

        int startBeat, endBeat, startTolerance, endTolerance;

        // --- Determine Start Beat & Tolerance ---
        if (lockedPreviousEndBeat != -1)
        {
            startBeat = lockedPreviousEndBeat;
            startTolerance = 0;
        }
        else
        {
            bool isGlobalStart = (i == 0 && startConst.id == 0);
            startBeat = isFullRetarget ? findClosestBeat(startConst.sourceTime) : getBeatForStep(startStep);
            if (isFullRetarget && isGlobalStart) startBeat = 0;
            startTolerance = (isFullRetarget && !isGlobalStart) ? beatTolerance : 0;
        }

        // --- Determine End Beat & Tolerance ---
        bool isChangedConstraint = !isFullRetarget && (i + 1 == changedIdx);
        bool isGlobalEnd = (i + 1 == activeConstraints.size() - 1 && endConst.id == 1);

        if (isChangedConstraint)
        {
            endBeat = findClosestBeat(endConst.sourceTime);
            endTolerance = beatTolerance;
        }
        else
        {
            endBeat = isFullRetarget ? findClosestBeat(endConst.sourceTime) : getBeatForStep(endStep);
            if (isFullRetarget && isGlobalEnd) endBeat = numBeats - 1;
            endTolerance = (isFullRetarget && !isGlobalEnd) ? beatTolerance : 0;
        }
        
        // --- Execute Retargeting ---
        auto result = audioRetargeter.retargetSegment(
            similarityMatrix, beatsFloat,
            startStep, endStep + 1,
            startBeat, endBeat,
            secondsPerBeat,
            similarityPenalty, backwardJumpPenalty, timeContinuityPenalty,
            startTolerance, endTolerance
        );
        
        lockedPreviousEndBeat = result.actualEndBeat;

        // --- Update Path ---
        if (isFullRetarget)
        {
            if (!pathSegments.empty() && !result.path.empty())
                pathSegments.insert(pathSegments.end(), result.path.begin() + 1, result.path.end());
            else
                pathSegments.insert(pathSegments.end(), result.path.begin(), result.path.end());
        }
        else
        {
            for (size_t j = 0; j < result.path.size(); ++j)
            {
                int targetIdx = startStep + static_cast<int>(j);
                if (targetIdx < newPath.size())
                    newPath[targetIdx] = result.path[j];
            }
        }
    }

    // 4. ==================================================================
    // Finalize and apply the new path
    
    retargetedBeatPath = isFullRetarget ? pathSegments : newPath;
    createRetargetedAudio(retargetedBeatPath);
    
    const auto duration = juce::String(juce::Time::getMillisecondCounterHiRes() - startTime, 2);
    if (isFullRetarget)
        DBG("Full retarget completed in " << duration << " ms");
    else
        DBG("Segmented retarget completed in " << duration << " ms");
}

std::vector<std::pair<double, double>> DynamicMusicVstAudioProcessor::getUnusedSourceRegions() const
{
    std::vector<std::pair<double, double>> unusedRegions;
    
    if (beatTimestamps.empty() || retargetedBeatPath.empty())
        return unusedRegions;
    
    // Create a set of all source beat indices that ARE used
    std::set<int> usedBeats;
    for (int beatIdx : retargetedBeatPath)
    {
        if (beatIdx >= 0 && beatIdx < (int)beatTimestamps.size())
            usedBeats.insert(beatIdx);
    }
    
    // Find contiguous ranges of unused beats
    int numBeats = (int)beatTimestamps.size();
    bool inUnusedRegion = false;
    double regionStart = 0.0;
    
    for (int i = 0; i < numBeats; ++i)
    {
        bool isUsed = usedBeats.count(i) > 0;
        
        if (!isUsed && !inUnusedRegion)
        {
            // Start of unused region
            regionStart = beatTimestamps[i];
            inUnusedRegion = true;
        }
        else if (isUsed && inUnusedRegion)
        {
            // End of unused region
            double regionEnd = beatTimestamps[i];
            unusedRegions.push_back({regionStart, regionEnd});
            inUnusedRegion = false;
        }
    }
    
    // Handle case where unused region extends to the end
    if (inUnusedRegion)
    {
        double endTime = (double)sourceAudioBuffer.getNumSamples() / fileSampleRate;
        unusedRegions.push_back({regionStart, endTime});
    }
    
    return unusedRegions;
}

std::vector<DynamicMusicVstAudioProcessor::CutInfo> DynamicMusicVstAudioProcessor::getCuts() const
{
    std::vector<CutInfo> cuts;
    
    if (retargetedBeatPath.empty() || beatTimestamps.empty() || currentSecondsPerBeat <= 0)
        return cuts;
    
    // Iterate through the retargeted path and find discontinuities
    for (size_t i = 0; i < retargetedBeatPath.size() - 1; ++i)
    {
        int currentBeat = retargetedBeatPath[i];
        int nextBeat = retargetedBeatPath[i + 1];
        
        // Check if there's a discontinuity (jump in source)
        if (std::abs(nextBeat - currentBeat) > 1)
        {
            CutInfo cut;
            
            // Target time is where the cut appears in the timeline
            cut.targetTime = (i + 1) * currentSecondsPerBeat;
            
            // Source times before and after the cut
            if (currentBeat >= 0 && currentBeat < (int)beatTimestamps.size())
                cut.sourceTimeFrom = beatTimestamps[currentBeat];
            else
                cut.sourceTimeFrom = 0.0;
                
            if (nextBeat >= 0 && nextBeat < (int)beatTimestamps.size())
                cut.sourceTimeTo = beatTimestamps[nextBeat];
            else
                cut.sourceTimeTo = 0.0;
            
            // Calculate quality based on similarity matrix
            cut.quality = 0.5f; // Default
            
            if (!similarityMatrix.empty() && 
                currentBeat >= 0 && currentBeat < (int)similarityMatrix.size() &&
                nextBeat >= 0 && nextBeat < (int)similarityMatrix[currentBeat].size())
            {
                cut.quality = similarityMatrix[currentBeat][nextBeat];
            }
            
            cuts.push_back(cut);
        }
    }
    
    return cuts;
}

void DynamicMusicVstAudioProcessor::startScrubbing()
{
    // Set the flag immediately for instant response (non-blocking)
    isScrubbing.store(true);
    
    // Don't stop playback - let scrubbing play on top of ongoing playback
}

void DynamicMusicVstAudioProcessor::stopScrubbing()
{
    // Clear the scrubbing flag
    isScrubbing.store(false);
    
    // By setting the read position to the end, we ensure any playing snippet stops.
    const juce::ScopedLock sl(snippetLock);
    scrubSnippetReadPos.store(scrubSnippetBuffer.getNumSamples());
}

void DynamicMusicVstAudioProcessor::triggerScrubSnippet(double scrubPositionSecs)
{
    const juce::ScopedLock sl(snippetLock);
    auto& source = retargetedAudioBuffer;
    if (source.getNumSamples() == 0) return;

    const int snippetLengthSamples = static_cast<int>(0.125 * fileSampleRate); // 125ms at any sample rate
    int startSample = static_cast<int>(juce::jmax(0.0, scrubPositionSecs) * fileSampleRate);
    startSample = juce::jmin(startSample, source.getNumSamples() - 1);
    
    int numSamplesToCopy = juce::jmin(snippetLengthSamples, source.getNumSamples() - startSample);

    scrubSnippetBuffer.setSize(source.getNumChannels(), numSamplesToCopy, false, true, true);
    
    for (int channel = 0; channel < source.getNumChannels(); ++channel)
    {
        scrubSnippetBuffer.copyFrom(channel, 0, source, channel, startSample, numSamplesToCopy);
    }

    scrubSnippetReadPos.store(0);
}

void DynamicMusicVstAudioProcessor::triggerScrubSnippetFromSource(double scrubPositionSecs)
{
    const juce::ScopedLock sl(snippetLock);
    auto& source = sourceAudioBuffer;
    if (source.getNumSamples() == 0) return;

    const int snippetLengthSamples = static_cast<int>(0.125 * fileSampleRate); // 125ms at any sample rate
    int startSample = static_cast<int>(juce::jmax(0.0, scrubPositionSecs) * fileSampleRate);
    startSample = juce::jmin(startSample, source.getNumSamples() - 1);
    
    int numSamplesToCopy = juce::jmin(snippetLengthSamples, source.getNumSamples() - startSample);

    scrubSnippetBuffer.setSize(source.getNumChannels(), numSamplesToCopy, false, true, true);
    
    for (int channel = 0; channel < source.getNumChannels(); ++channel)
    {
        scrubSnippetBuffer.copyFrom(channel, 0, source, channel, startSample, numSamplesToCopy);
    }

    scrubSnippetReadPos.store(0);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new DynamicMusicVstAudioProcessor();
}

void DynamicMusicVstAudioProcessor::performBackgroundRetarget()
{
    std::vector<ConstraintPoint> constraintsCopy;
    int changedId = -1;
    bool isFull = false;
    
    {
        const juce::ScopedLock lock(constraintLock);
        constraintsCopy = userConstraints;
        changedId = lastChangedConstraintId.load();
        isFull = isFullRetargetNeeded.load();
    }
    
    // Now perform the heavy lifting without holding the lock
    performRetarget(isFull, changedId, constraintsCopy);
}
