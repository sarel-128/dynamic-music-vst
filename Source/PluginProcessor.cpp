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
            float targetDuration = *parameters.getRawParameterValue("targetDuration");
            
            // Convert std::vector<double> to std::vector<float> for beats
            std::vector<float> beatsFloat(beatTimestamps.begin(), beatTimestamps.end());

            // These penalties can be exposed as parameters later
            float similarityPenalty = 0.95f;
            float backwardJumpPenalty = 0.1f;
            float timeContinuityPenalty = 0.01f;
            float constraintTimePenalty = 0.5f;
            float constraintBeatPenalty = 0.2f;

            // Determine if duration is dictated by constraints
            float effectiveDuration = targetDuration;
            bool durationDictated = false;
            
            if (!userConstraints.empty())
            {
                float maxConstraintTime = 0.0f;
                for (const auto& uc : userConstraints)
                {
                    if (uc.second > maxConstraintTime) maxConstraintTime = uc.second;
                }
                
                if (maxConstraintTime > 0.1f)
                {
                    effectiveDuration = maxConstraintTime;
                    durationDictated = true;
                }
            }

            // Perform a single retargeting run with the effective duration
            auto stepStartTime = juce::Time::getMillisecondCounterHiRes();

            // Construct constraints
            std::vector<std::pair<float, float>> constraints;
            constraints.push_back({0.0f, 0.0f});

            // Only add automatic end constraint if NOT dictated by user
            if (!durationDictated)
            {
                constraints.push_back({(float)getOriginalTotalLengthSecs(), effectiveDuration});
            }
            
            // Add user constraints
            for (const auto& uc : userConstraints)
            {
                // Only add constraints that fit within the effective duration
                if (uc.second <= effectiveDuration + 0.01f)
                {
                    constraints.push_back(uc);
                }
            }

            auto result = audioRetargeter.retargetDuration(similarityMatrix,
                                                              beatsFloat,
                                                              effectiveDuration,
                                                              similarityPenalty,
                                                              backwardJumpPenalty,
                                                              timeContinuityPenalty,
                                                              constraintTimePenalty,
                                                              constraintBeatPenalty,
                                                              constraints);

            DBG("Retarget run for duration " << effectiveDuration << "s took: " << juce::String(juce::Time::getMillisecondCounterHiRes() - stepStartTime, 2) << " ms. Cost: " << result.cost);

            retargetedBeatPath = result.path;
            
            auto creationStartTime = juce::Time::getMillisecondCounterHiRes();
            createRetargetedAudio(retargetedBeatPath);
            DBG("Audio creation took: " << juce::String(juce::Time::getMillisecondCounterHiRes() - creationStartTime, 2) << " ms.");

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

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new DynamicMusicVstAudioProcessor();
}
