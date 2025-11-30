#include "AudioAnalysis.h"
#include <juce_dsp/juce_dsp.h>
#include <cmath>
#include <numeric>
#include <limits>

AudioAnalysis::AudioAnalysis()
{
}

double AudioAnalysis::hzToMel(double hz)
{
    return 2595.0 * std::log10(1.0 + hz / 700.0);
}

double AudioAnalysis::melToHz(double mel)
{
    return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
}

void AudioAnalysis::createMelFilterbank(int numBands, int fftSize, double sampleRate)
{
    double minMel = hzToMel(0);
    double maxMel = hzToMel(sampleRate / 2.0);
    
    std::vector<double> melPoints(numBands + 2);
    for (int i = 0; i < numBands + 2; ++i)
    {
        melPoints[i] = minMel + i * (maxMel - minMel) / (numBands + 1);
    }
    
    std::vector<double> hzPoints(numBands + 2);
    for (int i = 0; i < numBands + 2; ++i)
    {
        hzPoints[i] = melToHz(melPoints[i]);
    }
    
    std::vector<int> binPoints(numBands + 2);
    for (int i = 0; i < numBands + 2; ++i)
    {
        binPoints[i] = static_cast<int>(std::floor((fftSize + 1) * hzPoints[i] / sampleRate));
    }
    
    melFilterbank.resize(numBands);
    for (int i = 0; i < numBands; ++i)
    {
        melFilterbank[i].clear();
        int startBin = binPoints[i];
        int centerBin = binPoints[i+1];
        int endBin = binPoints[i+2];
        
        // --- Librosa Slaney-style normalization ---
        // Scale filters to be approx constant energy per channel
        float norm = 2.0f / (hzPoints[i+2] - hzPoints[i]);

        for (int j = startBin; j < centerBin; ++j)
        {
            if (j < fftSize / 2 + 1)
            {
                float weight = (float)(j - startBin) / (centerBin - startBin);
                melFilterbank[i].push_back({j, weight * norm});
            }
        }
        for (int j = centerBin; j < endBin; ++j)
        {
            if (j < fftSize / 2 + 1)
            {
                float weight = (float)(endBin - j) / (endBin - centerBin);
                melFilterbank[i].push_back({j, weight * norm});
            }
        }
    }
}

const std::vector<std::vector<float>>& AudioAnalysis::getPowerMelSpectrogram(const juce::AudioBuffer<float>& buffer, double sampleRate, int fftSize, int hopSize)
{
    const auto* audioData = buffer.getReadPointer(0);
    int numSamples = buffer.getNumSamples();

    // --- 1. Check if the cache is valid ---
    bool isCacheValid = !cachedPowerMelSpectrogram.empty() &&
                        cachedFftSize == fftSize &&
                        cachedHopSize == hopSize &&
                        cachedSampleRate == sampleRate &&
                        cachedAudioDataPtr == audioData &&
                        cachedNumSamples == numSamples;
    
    if (isCacheValid)
    {
        return cachedPowerMelSpectrogram;
    }

    // --- 2. If cache is invalid, calculate the Mel spectrogram from scratch ---
    // Recalculate Mel filterbank if needed
    // Note: nBands is coming from a member variable `lastNbands` which is set in getOnsetStrengthEnvelope.
    // This assumes getOnsetStrengthEnvelope is called before any other spectrogram-related function that might need a different nBands.
    if (melFilterbank.empty() || lastFftSize != fftSize || lastSampleRate != sampleRate)
    {
        createMelFilterbank(lastNbands, fftSize, sampleRate);
        lastFftSize = fftSize;
        lastSampleRate = sampleRate;
    }

    juce::dsp::FFT fft(log2(fftSize));

    const int padding = fftSize / 2;
    std::vector<float> paddedAudio(numSamples + 2 * padding);
    // Left pad
    for (int i = 0; i < padding; ++i) { paddedAudio[i] = audioData[padding - 1 - i]; }
    // Copy original
    std::copy(audioData, audioData + numSamples, paddedAudio.begin() + padding);
    // Right pad
    for (int i = 0; i < padding; ++i) { paddedAudio[padding + numSamples + i] = audioData[numSamples - 1 - i]; }

    const float* paddedAudioData = paddedAudio.data();
    const int numPaddedSamples = paddedAudio.size();
    
    std::vector<float> window(fftSize);
    for(int j = 0; j < fftSize; ++j) {
        window[j] = 0.5f - 0.5f * std::cos(2.0f * M_PI * j / (fftSize - 1));
    }
    
    // Clear and reserve cache space
    cachedPowerMelSpectrogram.clear();
    cachedPowerMelSpectrogram.reserve((numPaddedSamples - fftSize) / hopSize + 1);
    
    // Calculate Mel power spectrogram
    for (int i = 0; i + fftSize <= numPaddedSamples; i += hopSize)
    {
        std::vector<float> fftBuffer(fftSize * 2, 0.0f);
        for(int j=0; j<fftSize; ++j) {
            fftBuffer[j] = paddedAudioData[i+j] * window[j];
        }

        fft.performFrequencyOnlyForwardTransform(fftBuffer.data());
        
        for (size_t k = 0; k < fftSize / 2 + 1; ++k) {
            fftBuffer[k] = fftBuffer[k] * fftBuffer[k]; // Power = magnitude^2
        }

        std::vector<float> currentMelFrame(melFilterbank.size(), 0.0f);
        for (size_t band = 0; band < melFilterbank.size(); ++band)
        {
            for (const auto& binWeightPair : melFilterbank[band])
            {
                currentMelFrame[band] += fftBuffer[binWeightPair.first] * binWeightPair.second;
            }
        }
        cachedPowerMelSpectrogram.push_back(currentMelFrame);
    }

    // --- 3. Update cache validity info ---
    cachedFftSize = fftSize;
    cachedHopSize = hopSize;
    cachedSampleRate = sampleRate;
    cachedAudioDataPtr = audioData;
    cachedNumSamples = numSamples;
    
    return cachedPowerMelSpectrogram;
}

std::vector<float> AudioAnalysis::getOnsetStrengthEnvelope(const juce::AudioBuffer<float>& buffer,
                                                           double sampleRate,
                                                           int hopSize,
                                                           int fftSize,
                                                           int nBands,
                                                           int lag,
                                                           int max_size,
                                                           AggregationMethod aggregate)
{
    // --- Update last known nBands, as it's used by the shared getPowerMelSpectrogram ---
    lastNbands = nBands;

    // --- 1. Get the power Mel spectrogram (from cache or new calculation) ---
    const auto& powerMelSpectrogram = getPowerMelSpectrogram(buffer, sampleRate, fftSize, hopSize);
    
    // Make a mutable copy for further processing
    auto melSpectrogram = powerMelSpectrogram;

    // --- Corrected dB Conversion to match Librosa ---
    float amin = 1e-10f;
    float top_db = 80.0f;

    // Pass 1: Convert to log scale (ref=1.0) and find the max dB value
    float max_db = -std::numeric_limits<float>::infinity();
    for (auto& frame : melSpectrogram)
    {
        for (auto& val : frame)
        {
            val = 10.0f * std::log10(std::max(amin, val));
            if (val > max_db)
            {
                max_db = val;
            }
        }
    }

    // Pass 2: Apply top_db clipping
    float threshold = max_db - top_db;
    for (auto& frame : melSpectrogram)
    {
        for (auto& val : frame)
        {
            val = std::max(val, threshold);
        }
    }
    
    // --- 2. Compute Reference Spectrogram (Maximum Filtering) ---
    std::vector<std::vector<float>> refSpectrogram = melSpectrogram; // Copy
    if (max_size > 1)
    {
        for (auto& frame : refSpectrogram)
        {
            std::vector<float> originalFrame = frame; // Keep a copy for the filtering op
            for (size_t band = 0; band < frame.size(); ++band)
            {
                float max_val = -std::numeric_limits<float>::infinity();
                for (int i = 0; i < max_size; ++i)
                {
                    int idx = static_cast<int>(band) + i - (max_size / 2);
                    if (idx >= 0 && idx < static_cast<int>(originalFrame.size()))
                    {
                        if (originalFrame[idx] > max_val)
                        {
                            max_val = originalFrame[idx];
                        }
                    }
                }
                frame[band] = max_val;
            }
        }
    }

    // --- 3. Compute Spectral Flux ---
    int numFrames = melSpectrogram.size();
    std::vector<float> onsetEnvelope;
    onsetEnvelope.reserve(numFrames > lag ? numFrames - lag : 0);

    for (int t = lag; t < numFrames; ++t)
    {
        std::vector<float> frame_fluxes;
        frame_fluxes.reserve(melFilterbank.size());
        for (size_t band = 0; band < melFilterbank.size(); ++band)
        {
            float diff = melSpectrogram[t][band] - refSpectrogram[t-lag][band];
            frame_fluxes.push_back(std::max(0.0f, diff)); // Half-wave rectification
        }

        if (aggregate == AggregationMethod::Mean)
        {
            float sum = 0.0f;
            for(float f : frame_fluxes) sum += f;
            if (!frame_fluxes.empty())
                onsetEnvelope.push_back(sum / frame_fluxes.size());
            else
                onsetEnvelope.push_back(0.0f);
        }
        else // Median
        {
            if (frame_fluxes.empty())
            {
                onsetEnvelope.push_back(0.0f);
            }
            else
            {
                std::sort(frame_fluxes.begin(), frame_fluxes.end());
                onsetEnvelope.push_back(frame_fluxes[frame_fluxes.size() / 2]);
            }
        }
    }
    std::vector<float> paddedEnvelope = onsetEnvelope;
    
    
    return paddedEnvelope;
}

std::vector<int> AudioAnalysis::detectOnsets(const std::vector<float>& onsetEnvelope, int hopSize)
{
    std::vector<int> onsets;
    // const int hopSize = 256; // Must match the hop size from getOnsetStrengthEnvelope
    const int windowSize = 10; // Frames for moving average
    const float constant = 0.03; // Constant to add to threshold

    if (onsetEnvelope.size() < windowSize)
    {
        return onsets;
    }

    // A more robust peak picking using a moving average threshold
    for (size_t i = windowSize; i < onsetEnvelope.size() - windowSize; ++i)
    {
        // Calculate local average (threshold)
        float sum = 0.0f;
        for(int j = -windowSize; j <= windowSize; ++j)
        {
            sum += onsetEnvelope[i + j];
        }
        float threshold = sum / (2 * windowSize + 1) + constant;

        // Check for peak
        if (onsetEnvelope[i] > onsetEnvelope[i - 1] && onsetEnvelope[i] > onsetEnvelope[i + 1] && onsetEnvelope[i] > threshold)
        {
            onsets.push_back(i * hopSize);
        }
    }
    return onsets;
}


float AudioAnalysis::estimateTempo(const std::vector<float>& onsetEnvelope,
                                   float sampleRate,
                                   int hopSize,
                                   float start_bpm,
                                   float std_bpm,
                                   float ac_size,
                                   float max_tempo)
{
    if (onsetEnvelope.empty() || start_bpm <= 0) return 120.0f;

    DBG("--- estimateTempo breakdown ---");
    auto timer = juce::Time::getMillisecondCounterHiRes();

    // --- 1. Calculate Tempogram ---
    // Calculate window size in frames from ac_size in seconds, matching librosa
    const int win_length = static_cast<int>(std::round(ac_size * sampleRate / hopSize));
    lastTempogram = calculateTempogram(onsetEnvelope, win_length);

    DBG("  calculateTempogram call: " << juce::String(juce::Time::getMillisecondCounterHiRes() - timer, 2) << " ms");
    timer = juce::Time::getMillisecondCounterHiRes();

    if (lastTempogram.empty()) return 120.0f;

    // --- 2. Aggregate Tempogram (like librosa's aggregate=np.mean) ---
    std::vector<float> globalAcf(win_length, 0.0f);
    if (!lastTempogram.empty())
    {
        for (const auto& frameAcf : lastTempogram)
        {
            if (frameAcf.size() == win_length)
            {
                for (int i = 0; i < win_length; ++i)
                {
                    globalAcf[i] += frameAcf[i];
                }
            }
        }
        float numFrames = static_cast<float>(lastTempogram.size());
        if (numFrames > 0)
        {
            for (float& val : globalAcf)
            {
                val /= numFrames;
            }
        }
    }
    lastGlobalAcf = globalAcf; // Store for UI

    DBG("  Aggregate Tempogram: " << juce::String(juce::Time::getMillisecondCounterHiRes() - timer, 2) << " ms");
    timer = juce::Time::getMillisecondCounterHiRes();

    // --- 3. Calculate BPM values for each bin ---
    std::vector<float> bpms = getTempoFrequencies(win_length, sampleRate, hopSize);

    // --- 4. Create the log-prior distribution ---
    std::vector<float> logprior(win_length);
    float log2_start_bpm = std::log2(start_bpm);

    for (int i = 0; i < win_length; ++i)
    {
        if (bpms[i] > 1e-6) // Check for bpms > 0
        {
            float log2_bpm = std::log2(bpms[i]);
            float term = (log2_bpm - log2_start_bpm) / std_bpm;
            logprior[i] = -0.5f * term * term;
        }
        else
        {
            logprior[i] = -std::numeric_limits<float>::infinity();
        }
    }

    // --- 5. Apply max_tempo constraint ---
    if (max_tempo > 0)
    {
        for (int i = 0; i < win_length; ++i)
        {
            if (bpms[i] >= max_tempo)
            {
                logprior[i] = -std::numeric_limits<float>::infinity();
            }
        }
    }
    
    // --- 6. Combine tempogram and prior, and find the peak ---
    int best_period_idx = -1;
    float max_weighted_val = -std::numeric_limits<float>::infinity();
    
    // Start from index 1 because bpms[0] corresponds to 0 BPM (infinite lag)
    for (int i = 1; i < win_length; ++i)
    {
        // Equivalent to: np.log1p(1e6 * globalAcf) + logprior
        float weighted_val = std::log1p(1e6f * globalAcf[i]) + logprior[i];
        if (weighted_val > max_weighted_val)
        {
            max_weighted_val = weighted_val;
            best_period_idx = i;
        }
    } 

    DBG("  Prior & Peak Finding: " << juce::String(juce::Time::getMillisecondCounterHiRes() - timer, 2) << " ms");

    // --- 7. Return the estimated tempo ---
    if (best_period_idx > 0)
    {
        return bpms[best_period_idx];
    }
    
    return 120.0f; // Fallback tempo
}

std::vector<std::vector<float>> AudioAnalysis::calculateTempogram(const std::vector<float>& onsetEnvelope, int winLength)
{
    if (onsetEnvelope.empty() || winLength < 1) return {};

    DBG("--- calculateTempogram breakdown ---");
    auto timer = juce::Time::getMillisecondCounterHiRes();

    // 1. Pad the onset envelope with a linear ramp to match librosa
    int padding = winLength / 2;
    std::vector<float> paddedEnvelope;
    paddedEnvelope.reserve(onsetEnvelope.size() + 2 * padding);

    // Left padding: linear ramp from 0 to the first element
    if (!onsetEnvelope.empty())
    {
        float firstVal = onsetEnvelope.front();
        float step = firstVal / (padding + 1.0f); // Correct step to match librosa
        for (int i = 0; i < padding; ++i)
        {
            paddedEnvelope.push_back((i + 1) * step); // Ramp from 0 up towards firstVal
        }
    } else {
        paddedEnvelope.insert(paddedEnvelope.end(), padding, 0.0f);
    }

    // Copy original data
    paddedEnvelope.insert(paddedEnvelope.end(), onsetEnvelope.begin(), onsetEnvelope.end());

    // Right padding: linear ramp from the last element down to (but not including) 0
    if (!onsetEnvelope.empty())
    {
        float lastVal = onsetEnvelope.back();
        float step = -lastVal / (padding + 1.0f); // Correct step to match librosa
        for (int i = 0; i < padding; ++i)
        {
            paddedEnvelope.push_back(lastVal + (i + 1) * step); // Ramp from lastVal down towards 0
        }
    } else {
         paddedEnvelope.insert(paddedEnvelope.end(), padding, 0.0f);
    }

    DBG("  1. Padding: " << juce::String(juce::Time::getMillisecondCounterHiRes() - timer, 2) << " ms");
    timer = juce::Time::getMillisecondCounterHiRes();

    // 2. Create the windowing function
    std::vector<float> hanningWindow(winLength);
    for (int i = 0; i < winLength; ++i)
    {
        hanningWindow[i] = 0.5f - 0.5f * std::cos(2.0f * M_PI * i / (winLength - 1));
    }

    DBG("  2. Hanning Window: " << juce::String(juce::Time::getMillisecondCounterHiRes() - timer, 2) << " ms");
    timer = juce::Time::getMillisecondCounterHiRes();

    // 3. Slide over the padded envelope and compute local autocorrelation for each frame
    std::vector<std::vector<float>> tempogram;
    int numFrames = onsetEnvelope.size();
    tempogram.reserve(numFrames);

    const int fftSize = juce::nextPowerOfTwo(winLength * 2);
    int fftOrder = static_cast<int>(std::log2(fftSize));

    // Create or resize FFT object and workspaces if necessary
    if (fftOrder != tempogramfftOrder)
    {
        tempogramfft = std::make_unique<juce::dsp::FFT>(fftOrder);
        tempogramfftOrder = fftOrder;
        tempogramfftWorkspace1.resize(fftSize);
        tempogramfftWorkspace2.resize(fftSize);
    }
    
    // Pre-allocate frame and acf buffers outside the loop
    std::vector<float> frame(winLength);
    std::vector<float> acf(winLength);


    for (int i = 0; i < numFrames; ++i)
    {
        // The frame starts at i in the padded envelope, which corresponds to the
        // original onset frame being at the center of the window.
        
        using SIMD = juce::dsp::SIMDRegister<float>;
        const int numElements = SIMD::SIMDNumElements;
        const size_t alignment = SIMD::SIMDRegisterSize;
        const int numSIMDTrips = winLength / numElements;

        for (int j = 0; j < numSIMDTrips * numElements; j += numElements)
        {
            // Use temporary aligned buffers to avoid crashes with unaligned std::vector data
            alignas(alignment) float envArr[numElements];
            alignas(alignment) float winArr[numElements];
            
            memcpy(envArr, paddedEnvelope.data() + i + j, numElements * sizeof(float));
            memcpy(winArr, hanningWindow.data() + j, numElements * sizeof(float));

            SIMD envelopeChunk = SIMD::fromRawArray(envArr);
            SIMD windowChunk   = SIMD::fromRawArray(winArr);
            
            (envelopeChunk * windowChunk).copyToRawArray(frame.data() + j);
        }
        // Handle remainder
        for (int j = numSIMDTrips * numElements; j < winLength; ++j)
        {
            frame[j] = paddedEnvelope[i + j] * hanningWindow[j];
        }

        // --- Autocorrelation via FFT ---
        using Complex = juce::dsp::Complex<float>;
        std::fill(tempogramfftWorkspace1.begin(), tempogramfftWorkspace1.end(), Complex{0.0f, 0.0f});
        std::fill(tempogramfftWorkspace2.begin(), tempogramfftWorkspace2.end(), Complex{0.0f, 0.0f});

        for(size_t j = 0; j < frame.size(); ++j)
            tempogramfftWorkspace1[j].real(frame[j]);

        tempogramfft->perform(tempogramfftWorkspace1.data(), tempogramfftWorkspace2.data(), false);

        for(int j = 0; j < fftSize; ++j)
            tempogramfftWorkspace2[j] = tempogramfftWorkspace2[j] * std::conj(tempogramfftWorkspace2[j]);
        
        tempogramfft->perform(tempogramfftWorkspace2.data(), tempogramfftWorkspace1.data(), true);

        for(int j=0; j<winLength; ++j)
        {
            acf[j] = tempogramfftWorkspace1[j].real();
        }

        // --- Normalization ---
        // Normalize by the max value (like librosa's norm=np.inf)
        float maxVal = 0.0f;
        for(float val : acf) {
            if (val > maxVal) {
                maxVal = val;
            }
        }

        if (maxVal > 1e-9) { // Avoid division by zero
            for(float& val : acf) {
                val /= maxVal;
            }
        }
        
        tempogram.push_back(acf);
    }

    DBG("  3. Autocorrelation Loop: " << juce::String(juce::Time::getMillisecondCounterHiRes() - timer, 2) << " ms");

    return tempogram;
}

std::vector<float> AudioAnalysis::createLocalScore(const std::vector<float>& onsetEnvelope, double period)
{
    if (onsetEnvelope.empty()) return {};

    // 1. Normalize onsets by standard deviation (more robust than max-scaling), matching librosa
    std::vector<float> normalizedEnvelope = onsetEnvelope;
    
    // Calculate sample standard deviation (ddof=1)
    double sum = std::accumulate(normalizedEnvelope.begin(), normalizedEnvelope.end(), 0.0);
    double mean = normalizedEnvelope.empty() ? 0.0 : sum / normalizedEnvelope.size();
    
    double sq_sum = 0.0;
    for(const auto& val : normalizedEnvelope) {
        sq_sum += (val - mean) * (val - mean);
    }
    double stddev = normalizedEnvelope.size() < 2 ? 0.0 : std::sqrt(sq_sum / (normalizedEnvelope.size() - 1));

    if (stddev > 1e-6) // Use a small epsilon like librosa's 'tiny'
    {
        for (auto& val : normalizedEnvelope)
        {
            val /= stddev; // Just scale, don't center (subtract mean)
        }
    }

    // 2. Create a Gaussian window whose width is proportional to the beat period
    int windowSize = static_cast<int>(std::round(period * 2.0));
    if (windowSize % 2 == 0) windowSize++; // Ensure odd size for a center peak
    std::vector<float> window(windowSize);
    double sigma = period / 32.0; 
    double center = (windowSize - 1) / 2.0;
    for (int i = 0; i < windowSize; ++i)
    {
        // Corrected Gaussian calculation (division, not multiplication)
        window[i] = std::exp(-0.5 * std::pow((i - center) / sigma, 2.0));
    }
    
    // 3. Convolve the normalized onsets with the Gaussian window
    std::vector<float> localScore(normalizedEnvelope.size(), 0.0f);
    int halfWindow = windowSize / 2;
    for (int i = 0; i < (int)normalizedEnvelope.size(); ++i)
    {
        for (int j = 0; j < windowSize; ++j)
        {
            int onsetIndex = i + j - halfWindow;
            if (onsetIndex >= 0 && onsetIndex < (int)normalizedEnvelope.size())
            {
                localScore[i] += normalizedEnvelope[onsetIndex] * window[j];
            }
        }
    }

    return localScore;
}


// Helper function to trim weak leading/trailing beats, inspired by librosa
void AudioAnalysis::trimBeats(const std::vector<float>& localScore, std::vector<int>& beatFrames, int hopSize)
{
    if (beatFrames.size() < 2) return;

    // Get localscore values at beat locations
    std::vector<float> beatScores;
    beatScores.reserve(beatFrames.size());
    for (int frame : beatFrames) {
        // Ensure index is within bounds
        size_t scoreIndex = frame / hopSize;
        if (scoreIndex < localScore.size()) {
            beatScores.push_back(localScore[scoreIndex]);
        }
    }
    
    if (beatScores.empty()) return;

    // Create a 5-point Hanning window
    std::vector<float> hanningWindow(5);
    for(int i=0; i<5; ++i) {
        hanningWindow[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / 4.0f));
    }

    // Convolve beatScores with Hanning window (mode='same')
    std::vector<float> smoothedScores(beatScores.size(), 0.0f);
    int halfWindow = hanningWindow.size() / 2;
    for(size_t i=0; i<beatScores.size(); ++i) {
        for(size_t j=0; j<hanningWindow.size(); ++j) {
            int scoreIndex = static_cast<int>(i) + static_cast<int>(j) - halfWindow;
            if (scoreIndex >= 0 && scoreIndex < static_cast<int>(beatScores.size())) {
                smoothedScores[i] += beatScores[scoreIndex] * hanningWindow[j];
            }
        }
    }
    
    // Calculate RMS of smoothed scores
    double rms_sq_sum = 0.0;
    for(float score : smoothedScores) {
        rms_sq_sum += score * score;
    }
    double rms = smoothedScores.empty() ? 0.0 : std::sqrt(rms_sq_sum / smoothedScores.size());
    
    double threshold = 0.5 * rms;
    
    // Trim leading beats
    auto firstBeat = beatFrames.begin();
    while (firstBeat != beatFrames.end()) {
        if (localScore[*firstBeat / hopSize] < threshold) {
            firstBeat = beatFrames.erase(firstBeat);
        } else {
            break;
        }
    }
    
    // Trim trailing beats
    while (!beatFrames.empty()) {
        if (localScore[beatFrames.back() / hopSize] < threshold) {
            beatFrames.pop_back();
        } else {
            break;
        }
    }
}


std::vector<double> AudioAnalysis::findBeats(const std::vector<float>& onsetEnvelope, float bpm, float sampleRate, int hopSize, double tightness)
{
    if (onsetEnvelope.empty() || bpm <= 0)
    {
        return {};
    }
    // Convert BPM to period in frames
    double framesPerSecond = sampleRate / hopSize;
    double period = framesPerSecond * 60.0 / bpm;

    // --- Create a tempo-synchronized local score (Librosa's key improvement) ---
    std::vector<float> localScore = createLocalScore(onsetEnvelope, period);
    if (localScore.empty())
    {
        return {};
    }

    std::vector<double> cumulativeScore(localScore.size());
    std::vector<int> backlink(localScore.size(), -1);

    // DP loop (operates on localScore now)
    int searchStart = static_cast<int>(std::floor(period / 2.0));
    int searchEnd = static_cast<int>(std::ceil(period * 2.0));
    searchStart = std::max(1, searchStart); // Ensure searchStart is at least 1

    for (int i = 0; i < (int)localScore.size(); ++i)
    {
        double maxScoreFromPrev = -std::numeric_limits<double>::infinity();
        int bestPrev = -1;

        // Search for best predecessor
        int p_start = i - searchEnd;
        int p_end = i - searchStart;

        for (int j = p_start; j <= p_end; ++j)
        {
            if (j < 0) continue;
            
            int p = i - j;

            // Use natural log for penalty, matching librosa
            double penalty = -tightness * std::pow(std::log((double)p / period), 2.0);
            
            double score = cumulativeScore[j] + penalty;

            if (score > maxScoreFromPrev)
            {
                maxScoreFromPrev = score;
                bestPrev = j;
            }
        }
        
        if (bestPrev != -1)
        {
            cumulativeScore[i] = localScore[i] + maxScoreFromPrev;
            backlink[i] = bestPrev;
        }
        else
        {
            cumulativeScore[i] = localScore[i];
            backlink[i] = -1; // No valid predecessor found
        }
    }

    // Backtrack from the best ending point
    int currentFrame = std::distance(cumulativeScore.begin(), std::max_element(cumulativeScore.begin(), cumulativeScore.end()));
    
    std::vector<int> beatFrames; // This will be in onset envelope frame indices
    while(currentFrame >= 0)
    {
        beatFrames.push_back(currentFrame);
        currentFrame = backlink[currentFrame];
    }
    std::reverse(beatFrames.begin(), beatFrames.end());

    // Convert onset frames to sample frames
    std::vector<int> beatSampleFrames;
    for (int frame : beatFrames) {
        beatSampleFrames.push_back(frame * hopSize);
    }
    
    // --- Trim weak leading/trailing beats ---
    trimBeats(localScore, beatSampleFrames, hopSize);

    // Convert frames to seconds
    std::vector<double> beatTimes;
    for(int frame : beatSampleFrames)
    {
        beatTimes.push_back(static_cast<double>(frame) / sampleRate);
    }

    return beatTimes;
}

std::vector<float> AudioAnalysis::getTempoFrequencies(int numBins, float sampleRate, int hopSize)
{
    std::vector<float> freqs(numBins);
    float framesPerSecond = sampleRate / hopSize;
    for (int i = 0; i < numBins; ++i)
    {
        if (i == 0)
        {
            freqs[i] = 0.0f; // Corresponds to infinite period
        }
        else
        {
            // Conversion from lag (in frames) to BPM
            freqs[i] = 60.0f * framesPerSecond / i;
        }
    }
    return freqs;
}

void AudioAnalysis::performDCT(std::vector<float>& input)
{
    const int N = input.size();
    if (N == 0) return;

    // --- 1. Setup FFT for DCT calculation ---
    // We use the relation DCT-II[x] = Re{exp(-j*pi*k/2N) * FFT_2N[x_padded]}
    // where x is padded with N zeros to make a 2N-length sequence.
    // The FFT size must be a power of two.
    int fftSize = juce::nextPowerOfTwo(N * 2);
    int fftOrder = static_cast<int>(std::log2(fftSize));

    // Create or resize FFT object and workspace if necessary to be efficient
    if (fftOrder != dctfftOrder)
    {
        dctfft = std::make_unique<juce::dsp::FFT>(fftOrder);
        dctfftOrder = fftOrder;
        dctfftWorkspace.resize(fftSize);
    }
    
    // --- 2. Prepare input buffer for FFT ---
    std::fill(dctfftWorkspace.begin(), dctfftWorkspace.end(), juce::dsp::Complex<float>{0.0f, 0.0f});
    for(int i = 0; i < N; ++i)
    {
        dctfftWorkspace[i].real(input[i]);
    }

    // --- 3. Perform FFT ---
    dctfft->perform(dctfftWorkspace.data(), dctfftWorkspace.data(), false); // false = forward transform

    // --- 4. Post-twiddle and extract real part to get the DCT sum ---
    float pi_over_2N = M_PI / (2.0f * N);
    
    for (int k = 0; k < N; ++k)
    {
        float angle = k * pi_over_2N;
        // Calculation is: Re{ (FFT_real + j*FFT_imag) * (cos(angle) - j*sin(angle)) }
        // which simplifies to: FFT_real * cos(angle) + FFT_imag * sin(angle)
        float real = dctfftWorkspace[k].real();
        float imag = dctfftWorkspace[k].imag();
        input[k] = real * std::cos(angle) + imag * std::sin(angle);
    }

    // --- 5. Apply ortho-normalization to match original implementation ---
    float sqrt_1_N = std::sqrt(1.0f / N);
    float sqrt_2_N = std::sqrt(2.0f / N);

    if (N > 0)
    {
        input[0] *= sqrt_1_N;
    }
    for (int k = 1; k < N; ++k)
    {
        input[k] *= sqrt_2_N;
    }
}


std::vector<std::vector<float>> AudioAnalysis::calculateMFCCs(const juce::AudioBuffer<float>& buffer,
                                                              double sampleRate,
                                                              int numCoefficients,
                                                              int fftSize,
                                                              int hopSize)
{
    // --- 1. Get the power Mel spectrogram (from cache or new calculation) ---
    const auto& powerMelSpectrogram = getPowerMelSpectrogram(buffer, sampleRate, fftSize, hopSize);
    
    // --- 2. We now have a power Mel spectrogram. Take the log. ---
    auto logMelSpectrogram = powerMelSpectrogram; // Make a mutable copy
    for (auto& frame : logMelSpectrogram)
    {
        for (auto& val : frame)
        {
            val = std::log(val + 1e-6f);
        }
    }

    // --- 3. Perform DCT on each frame to get MFCCs ---
    std::vector<std::vector<float>> allMfccs;
    allMfccs.reserve(logMelSpectrogram.size());

    for (auto& frame : logMelSpectrogram)
    {
        performDCT(frame);
        frame.resize(numCoefficients);
        allMfccs.push_back(frame);
    }
    
    return allMfccs;
}

std::vector<std::vector<float>> AudioAnalysis::concatenateMFCCsByBeats(const std::vector<std::vector<float>>& allMfccs, const std::vector<double>& beats, double sampleRate, int hopSize)
{
    std::vector<std::vector<float>> concatenatedMfccs;
    if (beats.size() < 2 || allMfccs.empty())
    {
        return concatenatedMfccs;
    }

    auto timeToFrame = [&](double time) {
        return static_cast<int>(time * sampleRate / hopSize);
    };

    std::vector<int> beatFrames;
    for (const auto& beatTime : beats)
    {
        beatFrames.push_back(timeToFrame(beatTime));
    }

    int min_dist = -1;
    for (size_t i = 1; i < beatFrames.size(); ++i)
    {
        int dist = beatFrames[i] - beatFrames[i-1];
        if (min_dist == -1 || dist < min_dist)
        {
            min_dist = dist;
        }
    }

    if (min_dist <= 0)
    {
        // Cannot proceed with non-positive min_dist
        return concatenatedMfccs;
    }

    for (size_t i = 0; i < beats.size() - 1; ++i)
    {
        int startFrame = beatFrames[i];
        int endFrame = startFrame + min_dist;

        if (endFrame > (int)allMfccs.size())
        {
            // Not enough frames for a full slice, skip this beat to avoid errors
            continue;
        }

        std::vector<float> one_mfcc;
        // Reserve space for efficiency
        one_mfcc.reserve(allMfccs[0].size() * min_dist);

        for (int frame = startFrame; frame < endFrame; ++frame)
        {
            one_mfcc.insert(one_mfcc.end(), allMfccs[frame].begin(), allMfccs[frame].end());
        }
        
        concatenatedMfccs.push_back(one_mfcc);
    }

    return concatenatedMfccs;
}

// Helper to calculate Pearson correlation, assuming inputs are already ranked for Spearman
float calculatePearsonCorrelationOnRanks(const std::vector<float>& x_ranks, const std::vector<float>& y_ranks)
{
    if (x_ranks.empty()) return 0.0f;

    float sum_x = 0.0f, sum_y = 0.0f, sum_xy = 0.0f;
    float sum_x2 = 0.0f, sum_y2 = 0.0f;
    int n = x_ranks.size();

    for (int i = 0; i < n; ++i)
    {
        sum_x += x_ranks[i];
        sum_y += y_ranks[i];
        sum_xy += x_ranks[i] * y_ranks[i];
        sum_x2 += x_ranks[i] * x_ranks[i];
        sum_y2 += y_ranks[i] * y_ranks[i];
    }

    float numerator = n * sum_xy - sum_x * sum_y;
    float denominator = std::sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));

    if (denominator == 0) return 0.0f;

    return numerator / denominator;
}

std::vector<std::vector<float>> AudioAnalysis::createSimilarityMatrix(const std::vector<std::vector<float>>& mfccs)
{
    int numSlices = mfccs.size();
    if (numSlices == 0) return {};

    DBG("--- createSimilarityMatrix breakdown ---");
    auto timer = juce::Time::getMillisecondCounterHiRes();

    // --- 1. Pre-calculate ranks for all MFCC vectors to avoid redundant work ---
    auto getRanks = [](const std::vector<float>& data) -> std::vector<float> {
        std::vector<int> indices(data.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                  [&](int a, int b) { return data[a] < data[b]; });
        std::vector<float> ranks(data.size());
        for (size_t i = 0; i < data.size(); ++i)
        {
            ranks[indices[i]] = i + 1;
        }
        return ranks;
    };

    std::vector<std::vector<float>> rankedMfccs;
    rankedMfccs.reserve(numSlices);
    for (const auto& vec : mfccs)
    {
        rankedMfccs.push_back(getRanks(vec));
    }

    DBG("  1. Pre-calculate Ranks: " << juce::String(juce::Time::getMillisecondCounterHiRes() - timer, 2) << " ms");
    timer = juce::Time::getMillisecondCounterHiRes();

    // --- 2. Calculate similarity using the pre-ranked data ---
    std::vector<std::vector<float>> similarityMatrix(numSlices, std::vector<float>(numSlices, 0.0f));
    
    // Determine the number of threads to use
    const int numThreads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> threads;
    threads.reserve(numThreads);

    // Define a worker lambda to calculate a portion of the matrix
    auto worker = [&](int start, int end) {
        for (int i = start; i < end; ++i)
        {
            for (int j = i; j < numSlices; ++j)
            {
                float correlation = calculatePearsonCorrelationOnRanks(rankedMfccs[i], rankedMfccs[j]);
                similarityMatrix[i][j] = correlation;
                similarityMatrix[j][i] = correlation; // Matrix is symmetric
            }
        }
    };

    // Dispatch the work to the threads
    int sliceSize = (numSlices + numThreads - 1) / numThreads;
    for (int i = 0; i < numThreads; ++i)
    {
        int start = i * sliceSize;
        int end = std::min(start + sliceSize, numSlices);
        if (start < end)
        {
            threads.emplace_back(worker, start, end);
        }
    }

    // Wait for all threads to complete
    for (auto& thread : threads)
    {
        thread.join();
    }

    DBG("  2. Calculate Correlation Matrix: " << juce::String(juce::Time::getMillisecondCounterHiRes() - timer, 2) << " ms");
    timer = juce::Time::getMillisecondCounterHiRes();
    
    // --- 3. Apply temporal smoothing ---
    auto smoothedMatrix = smoothSimilarityMatrix(similarityMatrix, 2);
    
    DBG("  3. Temporal Smoothing: " << juce::String(juce::Time::getMillisecondCounterHiRes() - timer, 2) << " ms");

    return smoothedMatrix;
}

std::vector<std::vector<float>> AudioAnalysis::smoothSimilarityMatrix(const std::vector<std::vector<float>>& matrix, int windowSize)
{
    int n = matrix.size();
    if (n == 0 || windowSize <= 1) return matrix;

    auto smoothedMatrix = matrix; // Start with a copy

    // Iterate over each diagonal. A diagonal is defined by a constant difference between row and column indices (i - j = d).
    // The difference 'd' ranges from -(n-1) (top-right) to (n-1) (bottom-left).
    for (int d = -(n - 1); d < n; ++d)
    {
        // 1. Extract the values and coordinates of the current diagonal
        std::vector<float> diagonalValues;
        std::vector<std::pair<int, int>> diagonalCoords;
        for (int i = 0; i < n; ++i)
        {
            int j = i - d;
            if (j >= 0 && j < n)
            {
                diagonalValues.push_back(matrix[i][j]);
                diagonalCoords.emplace_back(i, j);
            }
        }

        if (diagonalValues.size() < 2) continue; // Not enough elements to smooth

        // 2. Apply a simple moving average with the given window size
        std::vector<float> smoothedDiagonal = diagonalValues; // Copy to store results
        for (size_t i = 0; i < diagonalValues.size(); ++i)
        {
            float sum = 0.0f;
            int count = 0;
            // The window extends from (i - windowSize + 1) to i
            for (int k = 0; k < windowSize; ++k)
            {
                int index = (int)i - k;
                if (index >= 0 && index < (int)diagonalValues.size())
                {
                    sum += diagonalValues[index];
                    count++;
                }
            }
            if (count > 0)
            {
                smoothedDiagonal[i] = sum / count;
            }
        }

        // 3. Place the smoothed values back into the result matrix
        for (size_t i = 0; i < diagonalCoords.size(); ++i)
        {
            smoothedMatrix[diagonalCoords[i].first][diagonalCoords[i].second] = smoothedDiagonal[i];
        }
    }

    return smoothedMatrix;
}

float AudioAnalysis::calculateSpearmanCorrelation(const std::vector<float>& x, const std::vector<float>& y)
{
    if (x.size() != y.size() || x.empty())
    {
        return 0.0f;
    }

    auto getRanks = [](const std::vector<float>& data) -> std::vector<float> {
        std::vector<int> indices(data.size());
        std::iota(indices.begin(), indices.end(), 0);

        std::sort(indices.begin(), indices.end(),
                  [&](int a, int b) { return data[a] < data[b]; });

        std::vector<float> ranks(data.size());
        for (size_t i = 0; i < data.size(); ++i)
        {
            ranks[indices[i]] = i + 1;
        }
        return ranks;
    };

    std::vector<float> x_ranks = getRanks(x);
    std::vector<float> y_ranks = getRanks(y);

    return calculatePearsonCorrelationOnRanks(x_ranks, y_ranks);
}
