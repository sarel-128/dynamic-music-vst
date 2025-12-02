#include "PluginProcessor.h"
#include "PluginEditor.h"

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
}

DynamicMusicVstAudioProcessor::~DynamicMusicVstAudioProcessor()
{
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
    }
}

void DynamicMusicVstAudioProcessor::startPlayback()
{
    const juce::ScopedLock lock (transportSourceLock);

    if (isRetargeted.load())
    {
        retargetedTransportSource.start();
        currentPlaybackPositionSecs = retargetedTransportSource.getCurrentPosition();
    }
    else
    {
        fileTransportSource.start();
        currentPlaybackPositionSecs = fileTransportSource.getCurrentPosition();
    }

    // Reset retargeting playback position
    currentRetargetBeatIndex = 0;
    samplesIntoCurrentBeat = 0;


    // Find the next beat to play from the current position
    double currentTime = getCurrentPositionSecs();
    nextBeatToPlay = 0;
    for (size_t i = 0; i < beatTimestamps.size(); ++i)
    {
        if (beatTimestamps[i] >= currentTime)
        {
            nextBeatToPlay = (int)i;
            break;
        }
    }
}

void DynamicMusicVstAudioProcessor::stopPlayback()
{
    const juce::ScopedLock lock (transportSourceLock);
    
    fileTransportSource.stop();
    retargetedTransportSource.stop();

    nextBeatToPlay = 0;

    // Reset retargeting playback position
    currentRetargetBeatIndex = 0;
    samplesIntoCurrentBeat = 0;
}

void DynamicMusicVstAudioProcessor::setPlaybackPosition(double newPositionSecs)
{
    const juce::ScopedLock lock (transportSourceLock);
    
    if (isRetargeted.load())
    {
        retargetedTransportSource.setPosition(newPositionSecs);
        currentPlaybackPositionSecs = newPositionSecs;
    }
    else
    {
        fileTransportSource.setPosition(newPositionSecs);
        currentPlaybackPositionSecs = newPositionSecs;
    }

    // This is tricky for retargeted audio. For now, we'll just reset.
    // A proper implementation would need to map the timeline position to the retargeted path.
    currentRetargetBeatIndex = 0;
    samplesIntoCurrentBeat = 0;
    
    // Find the next beat to play from the new position
    nextBeatToPlay = 0;
    for (size_t i = 0; i < beatTimestamps.size(); ++i)
    {
        if (beatTimestamps[i] >= newPositionSecs)
        {
            nextBeatToPlay = (int)i;
            break;
        }
    }
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
        
    // If we're playing a file, get its audio and replace the input
    if (isPlayingFile && fileReaderSource != nullptr)
    {
        if (isRetargeted.load())
        {
            // --- Retargeted Playback from generated buffer ---
            const juce::ScopedLock lock (transportSourceLock);
            juce::AudioSourceChannelInfo channelInfo(&buffer, 0, buffer.getNumSamples());
            retargetedTransportSource.getNextAudioBlock(channelInfo);
            // The visual playhead is now driven by the editor timer, which calls getCurrentPositionSecs().
            // We just need to update the internal position for other logic if needed.
            currentPlaybackPositionSecs = retargetedTransportSource.getCurrentPosition();

        }
        else if (!retargetedBeatPath.empty() && !beatTimestamps.empty())
        {
            // --- Retargeted Playback Logic ---
            int samplesToProcess = buffer.getNumSamples();
            int bufferWritePos = 0;

            while (samplesToProcess > 0)
            {
                if (currentRetargetBeatIndex >= retargetedBeatPath.size())
                {
                    // Reached end of retargeted path, fill remainder with silence
                    buffer.clear(bufferWritePos, samplesToProcess);
                    break;
                }

                int originalBeatIndex = retargetedBeatPath[currentRetargetBeatIndex];
                
                double beatStartSecs = beatTimestamps[originalBeatIndex];
                double nextBeatStartSecs;
                if (originalBeatIndex + 1 < beatTimestamps.size())
                {
                    nextBeatStartSecs = beatTimestamps[originalBeatIndex + 1];
                }
                else
                {
                    nextBeatStartSecs = getTotalLengthSecs(); // Last beat goes to end of file
                }

                int beatStartSample = static_cast<int>(beatStartSecs * fileSampleRate);
                int nextBeatStartSample = static_cast<int>(nextBeatStartSecs * fileSampleRate);
                int beatNumSamples = nextBeatStartSample - beatStartSample;

                if (beatNumSamples <= 0)
                {
                    currentRetargetBeatIndex++;
                    samplesIntoCurrentBeat = 0;
                    continue;
                }
                
                // Update visual playback position
                currentPlaybackPositionSecs = beatStartSecs + (double)samplesIntoCurrentBeat / fileSampleRate;

                int samplesAvailableInBeat = beatNumSamples - samplesIntoCurrentBeat;
                int samplesToCopy = juce::jmin(samplesToProcess, samplesAvailableInBeat);

                if (samplesToCopy > 0)
                {
                    for (int ch = 0; ch < sourceAudioBuffer.getNumChannels(); ++ch)
                    {
                        buffer.copyFrom(ch, bufferWritePos, sourceAudioBuffer, ch, beatStartSample + samplesIntoCurrentBeat, samplesToCopy);
                    }
                    
                    bufferWritePos += samplesToCopy;
                    samplesToProcess -= samplesToCopy;
                    samplesIntoCurrentBeat += samplesToCopy;
                }

                if (samplesIntoCurrentBeat >= beatNumSamples)
                {
                    currentRetargetBeatIndex++;
                    samplesIntoCurrentBeat = 0;
                }
            }
        }
        else
        {
            // --- Standard Linear Playback ---
            const juce::ScopedLock lock (transportSourceLock);
            juce::AudioSourceChannelInfo channelInfo(&buffer, 0, buffer.getNumSamples());
            fileTransportSource.getNextAudioBlock(channelInfo);
            currentPlaybackPositionSecs = fileTransportSource.getCurrentPosition();
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
    if (id == 0)
    {
        DBG("Start anchor cannot be moved");
        return;
    }
    
    for (auto& c : userConstraints)
    {
        if (c.id == id)
        {
            // Special handling for end anchor (id=1): only X-axis (targetTime) can change
            if (id == 1)
            {
                // Keep sourceTime fixed at original end, only update targetTime
                c.targetTime = newTargetTime;
                DBG("End anchor moved to targetTime: " << newTargetTime);
            }
            else
            {
                // Regular constraint: both axes can change
                c.sourceTime = newSourceTime;
                c.targetTime = newTargetTime;
            }
            break;
        }
    }
    
    // Re-sort by targetTime
    std::sort(userConstraints.begin(), userConstraints.end(),
              [](const ConstraintPoint& a, const ConstraintPoint& b) {
                  return a.targetTime < b.targetTime;
              });
    
    // End anchor movement changes total duration, requiring full retarget
    // (segmented retarget can't change path length)
    if (id == 1)
    {
        performFullRetarget();
    }
    else
    {
        // Regular constraints can use segmented retarget
        performSegmentedRetarget(id);
    }
}

void DynamicMusicVstAudioProcessor::removeConstraint(int id)
{
    // Start and end anchors (id 0 and 1) cannot be removed
    if (id == 0 || id == 1)
    {
        DBG("Start and end anchors cannot be removed");
        return;
    }
    
    auto it = std::find_if(userConstraints.begin(), userConstraints.end(),
                           [id](const ConstraintPoint& c) { return c.id == id; });
    
    if (it != userConstraints.end())
    {
        userConstraints.erase(it);
        
        // After removing, perform full retarget (simpler than segmented for removal)
        performFullRetarget();
    }
}

void DynamicMusicVstAudioProcessor::performFullRetarget()
{
    if (similarityMatrix.empty() || beatTimestamps.empty())
    {
        return;
    }
    
    DBG(" ");
    DBG("=== Performing FULL Retarget ===");
    auto startTime = juce::Time::getMillisecondCounterHiRes();
    
    // Ensure constraints are sorted
    std::sort(userConstraints.begin(), userConstraints.end(),
              [](const ConstraintPoint& a, const ConstraintPoint& b) {
                  return a.targetTime < b.targetTime;
              });
    
    // Calculate duration from the last constraint (End Anchor)
    float targetDuration = userConstraints.back().targetTime;
    if (targetDuration <= 0.0f)
    {
        targetDuration = static_cast<float>(getOriginalTotalLengthSecs());
    }
    
    // Calculate seconds per beat
    float secondsPerBeat = 0.5f;
    if (beatTimestamps.size() > 1)
    {
        secondsPerBeat = static_cast<float>((beatTimestamps.back() - beatTimestamps.front()) / 
                                            (beatTimestamps.size() - 1));
    }
    
    int numBeats = static_cast<int>(similarityMatrix.size());
    std::vector<float> beatsFloat(beatTimestamps.begin(), beatTimestamps.end());
    
    // Penalties
    float similarityPenalty = 0.95f;
    float backwardJumpPenalty = 0.8f;
    float timeContinuityPenalty = 0.7f;
    // Interior soft constraint penalties (if we had any non-boundary constraints)
    // But here we treat ALL user constraints as boundaries (Hard Constraints)
    float constraintTimePenalty = 0.0f;
    float constraintBeatPenalty = 0.0f;
    
    std::vector<int> fullPath;
    
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

    // Chain retargeting segments between every pair of constraints
    // This treats EVERY user constraint as a HARD ANCHOR
    for (size_t i = 0; i < userConstraints.size() - 1; ++i)
    {
        const auto& startConst = userConstraints[i];
        const auto& endConst = userConstraints[i+1];
        
        int startStep = getStepForTime(startConst.targetTime);
        int endStep = getStepForTime(endConst.targetTime);
        
        // For start/end anchors, force exact beat indices if possible
        int startBeat = findClosestBeat(startConst.sourceTime);
        int endBeat = findClosestBeat(endConst.sourceTime);
        
        // Special case for global start/end to be exact
        if (i == 0 && startConst.id == 0) startBeat = 0;
        if (i == userConstraints.size() - 2 && endConst.id == 1) endBeat = numBeats - 1;

        startStep = juce::jmax(0, startStep);
        // Allow path to grow if needed, but usually limited by targetDuration
        
        if (endStep > startStep)
        {
            std::vector<ConstraintPoint> emptySoftConstraints; // No soft constraints inside segments
            
            // Extend endStep by 1 so the segment includes the end anchor step
            // This allows the DP to find a smooth transition TO the anchor beat at the exact time
            auto result = audioRetargeter.retargetSegment(
                similarityMatrix, beatsFloat,
                startStep, endStep + 1,
                startBeat, endBeat,
                secondsPerBeat,
                emptySoftConstraints,
                similarityPenalty, backwardJumpPenalty, timeContinuityPenalty,
                constraintTimePenalty, constraintBeatPenalty
            );
            
            // Append to full path
            // If fullPath is not empty, the new segment's first sample overlaps with the last sample
            // of the previous segment (both are the anchor beat). We skip the first sample of the new segment.
            if (!fullPath.empty() && !result.path.empty())
            {
                fullPath.insert(fullPath.end(), result.path.begin() + 1, result.path.end());
            }
            else
            {
                fullPath.insert(fullPath.end(), result.path.begin(), result.path.end());
            }
        }
    }
    
    retargetedBeatPath = fullPath;
    
    createRetargetedAudio(retargetedBeatPath);
    
    DBG("Full retarget completed in " << juce::String(juce::Time::getMillisecondCounterHiRes() - startTime, 2) << " ms");
}

void DynamicMusicVstAudioProcessor::performSegmentedRetarget(int changedConstraintId)
{
    if (similarityMatrix.empty() || beatTimestamps.empty() || retargetedBeatPath.empty())
    {
        // No existing path - need full retarget
        performFullRetarget();
        return;
    }
    
    DBG(" ");
    DBG("=== Performing SEGMENTED Retarget for constraint ID " << changedConstraintId << " ===");
    auto startTime = juce::Time::getMillisecondCounterHiRes();
    
    // Find the changed constraint and its neighbors
    // Constraints should already be sorted by targetTime
    int changedIdx = -1;
    for (size_t i = 0; i < userConstraints.size(); ++i)
    {
        if (userConstraints[i].id == changedConstraintId)
        {
            changedIdx = static_cast<int>(i);
            break;
        }
    }
    
    if (changedIdx < 0)
    {
        DBG("Constraint not found, performing full retarget");
        performFullRetarget();
        return;
    }
    
    // Calculate seconds per beat
    float secondsPerBeat = 0.5f;  // Default
    if (beatTimestamps.size() > 1)
    {
        secondsPerBeat = static_cast<float>((beatTimestamps.back() - beatTimestamps.front()) / 
                                            (beatTimestamps.size() - 1));
    }
    
    int numBeats = static_cast<int>(similarityMatrix.size());
    std::vector<float> beatsFloat(beatTimestamps.begin(), beatTimestamps.end());
    
    // Penalties
    float similarityPenalty = 0.95f;
    float backwardJumpPenalty = 0.7f;
    float timeContinuityPenalty = 0.4f;
    float constraintTimePenalty = 0.2f;
    float constraintBeatPenalty = 0.2f;
    
    // We need to update segments on both sides of the changed constraint
    // Left segment: from previous neighbor to changed constraint
    // Right segment: from changed constraint to next neighbor
    
    auto getStepForTime = [secondsPerBeat](float time) -> int {
        return static_cast<int>(time / secondsPerBeat);
    };
    
    auto getBeatForStep = [&](int step) -> int {
        if (step < 0) return 0;
        if (step >= static_cast<int>(retargetedBeatPath.size())) 
            return retargetedBeatPath.empty() ? 0 : retargetedBeatPath.back();
        return retargetedBeatPath[step];
    };
    
    // Find closest beat index for a source time
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
    
    std::vector<int> newPath = retargetedBeatPath;
    
    // Process left segment if there's a previous neighbor
    if (changedIdx > 0)
    {
        const auto& leftNeighbor = userConstraints[changedIdx - 1];
        const auto& changedConstraint = userConstraints[changedIdx];
        
        int startStep = getStepForTime(leftNeighbor.targetTime);
        int endStep = getStepForTime(changedConstraint.targetTime);
        int startBeat = findClosestBeat(leftNeighbor.sourceTime);
        int endBeat = findClosestBeat(changedConstraint.sourceTime);
        
        // Clamp steps to valid range
        startStep = juce::jmax(0, startStep);
        endStep = juce::jmin(static_cast<int>(retargetedBeatPath.size()) - 1, endStep);
        
        if (endStep > startStep)
        {
            // Collect interior constraints for this segment
            std::vector<ConstraintPoint> segmentConstraints;
            for (const auto& c : userConstraints)
            {
                int cStep = getStepForTime(c.targetTime);
                if (cStep > startStep && cStep < endStep)
                {
                    segmentConstraints.push_back(c);
                }
            }
            
            // Extend endStep by 1 to include the anchor step
            auto leftResult = audioRetargeter.retargetSegment(
                similarityMatrix, beatsFloat,
                startStep, endStep + 1,
                startBeat, endBeat,
                secondsPerBeat,
                segmentConstraints,
                similarityPenalty, backwardJumpPenalty, timeContinuityPenalty,
                constraintTimePenalty, constraintBeatPenalty
            );
            
            // Splice left segment result into path
            // We update the range [startStep, endStep] (inclusive)
            // leftResult covers indices [0, endStep-startStep] which map to steps [startStep, endStep]
            for (size_t i = 0; i < leftResult.path.size(); ++i)
            {
                int targetIdx = startStep + static_cast<int>(i);
                if (targetIdx < newPath.size())
                {
                    newPath[targetIdx] = leftResult.path[i];
                }
            }
        }
    }
    
    // Process right segment if there's a next neighbor
    if (changedIdx < static_cast<int>(userConstraints.size()) - 1)
    {
        const auto& changedConstraint = userConstraints[changedIdx];
        const auto& rightNeighbor = userConstraints[changedIdx + 1];
        
        int startStep = getStepForTime(changedConstraint.targetTime);
        int endStep = getStepForTime(rightNeighbor.targetTime);
        int startBeat = findClosestBeat(changedConstraint.sourceTime);
        int endBeat = findClosestBeat(rightNeighbor.sourceTime);
        
        // Clamp steps
        startStep = juce::jmax(0, startStep);
        endStep = juce::jmin(static_cast<int>(retargetedBeatPath.size()) - 1, endStep);
        
        if (endStep > startStep)
        {
            // Collect interior constraints for this segment
            std::vector<ConstraintPoint> segmentConstraints;
            for (const auto& c : userConstraints)
            {
                int cStep = getStepForTime(c.targetTime);
                if (cStep > startStep && cStep < endStep)
                {
                    segmentConstraints.push_back(c);
                }
            }
            
            // Extend endStep by 1 to include the anchor step
            auto rightResult = audioRetargeter.retargetSegment(
                similarityMatrix, beatsFloat,
                startStep, endStep + 1,
                startBeat, endBeat,
                secondsPerBeat,
                segmentConstraints,
                similarityPenalty, backwardJumpPenalty, timeContinuityPenalty,
                constraintTimePenalty, constraintBeatPenalty
            );
            
            // Splice right segment result into path
            // We update the range [startStep, endStep]
            // rightResult covers indices [0, endStep-startStep] which map to steps [startStep, endStep]
            for (size_t i = 0; i < rightResult.path.size(); ++i)
            {
                int targetIdx = startStep + static_cast<int>(i);
                if (targetIdx < newPath.size())
                {
                    newPath[targetIdx] = rightResult.path[i];
                }
            }
        }
    }
    
    retargetedBeatPath = newPath;
    
    createRetargetedAudio(retargetedBeatPath);
    
    DBG("Segmented retarget completed in " << juce::String(juce::Time::getMillisecondCounterHiRes() - startTime, 2) << " ms");
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new DynamicMusicVstAudioProcessor();
}
