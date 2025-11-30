#include "PluginProcessor.h"
#include "PluginEditor.h"

juce::AudioProcessorValueTreeState::ParameterLayout DynamicMusicVstAudioProcessor::createParameterLayout()
{
    juce::AudioProcessorValueTreeState::ParameterLayout layout;

    layout.add(std::make_unique<juce::AudioParameterFloat>("targetDuration",
                                                          "Target Duration",
                                                          0.1f,
                                                          30.0f,
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
        fileSampleRate = reader->sampleRate; // Store the file's sample rate

        // Store the full audio file in our source buffer for analysis
        sourceAudioBuffer.setSize((int)reader->numChannels, (int)reader->lengthInSamples);
        reader->read(&sourceAudioBuffer,
                     0,
                     (int)reader->lengthInSamples,
                     0,
                     true,
                     true);

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
    fileTransportSource.start();

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
    nextBeatToPlay = 0;
}

void DynamicMusicVstAudioProcessor::setPlaybackPosition(double newPositionSecs)
{
    const juce::ScopedLock lock (transportSourceLock);
    fileTransportSource.setPosition(newPositionSecs);

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
    return fileTransportSource.isPlaying();
}

double DynamicMusicVstAudioProcessor::getCurrentPositionSecs() const
{
    const juce::ScopedLock lock (transportSourceLock);
    return fileTransportSource.getCurrentPosition();
}

double DynamicMusicVstAudioProcessor::getTotalLengthSecs() const
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
        const juce::ScopedLock lock (transportSourceLock);
        juce::AudioSourceChannelInfo channelInfo(&buffer, 0, buffer.getNumSamples());
        fileTransportSource.getNextAudioBlock(channelInfo);
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
            auto allMfccs = audioAnalyser.calculateMFCCs(analysisBuffer, fileSampleRate, 40, fftSize, hopSize);
            DBG("5. Calculate MFCCs: " << juce::String(juce::Time::getMillisecondCounterHiRes() - stepStartTime, 2) << " ms");

            stepStartTime = juce::Time::getMillisecondCounterHiRes();
            auto concatenatedMfccs = audioAnalyser.concatenateMFCCsByBeats(allMfccs, relativeBeatTimestamps, fileSampleRate, hopSize);
            DBG("6. Concatenate MFCCs by Beats: " << juce::String(juce::Time::getMillisecondCounterHiRes() - stepStartTime, 2) << " ms");

            stepStartTime = juce::Time::getMillisecondCounterHiRes();
            similarityMatrix = audioAnalyser.createSimilarityMatrix(concatenatedMfccs);
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

void DynamicMusicVstAudioProcessor::changeListenerCallback (juce::ChangeBroadcaster* source)
{
    // The thumbnail component in the editor is also listening for this,
    // so we don't need to do anything here. This just fulfills the inheritance requirement.
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new DynamicMusicVstAudioProcessor();
}
