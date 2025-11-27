#include "AudioAnalysis.h"
#include <juce_dsp/juce_dsp.h>
#include <cmath>
#include <numeric>
#include <limits>

AudioAnalysis::AudioAnalysis()
{
    // Constructor is now empty. Filterbank is created in the new prepare() method
    // to ensure the correct sample rate is used.
}

void AudioAnalysis::prepare(double sampleRate, int fftSize)
{
    this->sampleRate = sampleRate;
    this->fftSize = fftSize;
    this->hopSize = fftSize / 4; // Common setting for good time/freq resolution

    // Pre-calculate the filterbank with parameters matching Librosa's defaults for onset detection.
    createMelFilterbank(128, this->fftSize, this->sampleRate);
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
        
        for (int j = startBin; j < centerBin; ++j)
        {
            if (j < fftSize / 2 + 1)
            {
                float weight = (float)(j - startBin) / (centerBin - startBin);
                melFilterbank[i].push_back({j, weight});
            }
        }
        for (int j = centerBin; j < endBin; ++j)
        {
            if (j < fftSize / 2 + 1)
            {
                float weight = (float)(endBin - j) / (endBin - centerBin);
                melFilterbank[i].push_back({j, weight});
            }
        }
    }
}


std::vector<float> AudioAnalysis::getOnsetStrengthEnvelope(const juce::AudioBuffer<float>& buffer)
{
    // These are now class members, initialized in prepare()
    // const int fftSize = 2048;
    // const int hopSize = 512;
    
    // --- Librosa Alignment: Centered frames ---
    // Create a padded version of the buffer to emulate center=true in librosa's STFT
    int padding = fftSize / 2;
    // The padded buffer is zero-initialized by its constructor.
    juce::AudioBuffer<float> paddedBuffer(buffer.getNumChannels(), buffer.getNumSamples() + 2 * padding);
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
    {
        // Just copy the original data into the center of the zero-padded buffer.
        // This is a safe way to implement centered framing.
        paddedBuffer.copyFrom(ch, padding, buffer, ch, 0, buffer.getNumSamples());
    }
    
    auto* audioData = paddedBuffer.getReadPointer(0); // Analyze mono
    int numSamples = paddedBuffer.getNumSamples();

    // --- First Pass: Calculate all Mel Power Spectrograms and find global max ---
    std::vector<std::vector<float>> allMelSpectrums;
    float globalMax = 0.0f;
    
    juce::dsp::FFT fft(log2(fftSize));
    std::vector<float> window(fftSize);
    for(int j=0; j<fftSize; ++j) {
        window[j] = 0.5f - 0.5f * std::cos(2.0f * M_PI * j / (fftSize - 1));
    }

    for (int i = 0; i + fftSize <= numSamples; i += hopSize)
    {
        std::vector<float> windowedData(fftSize);
        for(int j=0; j<fftSize; ++j) {
            windowedData[j] = audioData[i+j] * window[j];
        }

        std::vector<float> fftBuffer(fftSize * 2, 0.0f);
        std::copy(windowedData.begin(), windowedData.end(), fftBuffer.begin());
        fft.performFrequencyOnlyForwardTransform(fftBuffer.data());

        std::vector<float> melPower(melFilterbank.size(), 0.0f);
        for (size_t band = 0; band < melFilterbank.size(); ++band)
        {
            for (const auto& binWeightPair : melFilterbank[band])
            {
                melPower[band] += fftBuffer[binWeightPair.first];
            }
            melPower[band] = melPower[band] * melPower[band]; // Convert to power
            globalMax = std::max(globalMax, melPower[band]);
        }
        allMelSpectrums.push_back(melPower);
    }
    
    // --- Second Pass: Convert to dB relative to global max ---
    std::vector<std::vector<float>> S_log_power;
    float amin = 1e-10f; // Floor for log calculation, similar to librosa

    for(const auto& melPowerFrame : allMelSpectrums)
    {
        std::vector<float> currentMelSpectrum(melFilterbank.size());
        for(size_t band = 0; band < melPowerFrame.size(); ++band)
        {
            float power = melPowerFrame[band];
            if (power < amin) power = amin;
            // Librosa's power_to_db: 10.0 * log10(S / ref) with ref=globalMax
            currentMelSpectrum[band] = 10.0f * std::log10(power / globalMax);
        }
        S_log_power.push_back(currentMelSpectrum);
    }
    
    // --- Third Pass: Calculate onset strength envelope from S_log_power (a la onset_strength_multi) ---
    const int lag = 1;
    const int max_size = 1; // librosa default
    const int num_frames = S_log_power.size();
    const int num_bands = num_frames > 0 ? S_log_power[0].size() : 0;

    // 1. Create the reference spectrogram
    auto& ref_log_power = S_log_power; // By default, ref is S (max_size=1)
    std::vector<std::vector<float>> S_filtered;
    if (max_size > 1)
    {
        S_filtered.resize(num_frames, std::vector<float>(num_bands));
        for (int t = 0; t < num_frames; ++t) {
            for (int b = 0; b < num_bands; ++b) {
                float max_val = -std::numeric_limits<float>::infinity();
                int start = std::max(0, b - max_size / 2);
                int end = std::min(num_bands, b + max_size / 2 + 1);
                for (int k = start; k < end; ++k) {
                    max_val = std::max(max_val, S_log_power[t][k]);
                }
                S_filtered[t][b] = max_val;
            }
        }
        ref_log_power = S_filtered;
    }

    // 2. Compute raw spectral flux
    std::vector<float> raw_flux;
    raw_flux.reserve(num_frames);
    for (size_t t = lag; t < num_frames; ++t)
    {
        float sumOfDifferences = 0.0f;
        int countOfPositiveDifferences = 0;
        for (size_t band = 0; band < num_bands; ++band) {
            float spectralDifference = S_log_power[t][band] - ref_log_power[t - lag][band];
            if (spectralDifference > 0)
            {
                sumOfDifferences += spectralDifference;
                countOfPositiveDifferences++;
            }
        }
        
        float flux = 0.0f;
        if (countOfPositiveDifferences > 0)
        {
            flux = sumOfDifferences / countOfPositiveDifferences;
        }
        raw_flux.push_back(flux);
    }

    // 3. Compensate for lag and centering, then pad and trim
    int pad_width = lag;
    pad_width += fftSize / (2 * hopSize); // center=true compensation

    std::vector<float> onsetEnvelope(num_frames, 0.0f);
    int start_frame = pad_width;
    int end_frame = std::min((int)onsetEnvelope.size(), start_frame + (int)raw_flux.size());

    for(int i = start_frame; i < end_frame; ++i)
    {
        onsetEnvelope[i] = raw_flux[i - start_frame];
    }

    return onsetEnvelope;
}

std::vector<int> AudioAnalysis::onsetBacktrack(const std::vector<int>& events, const std::vector<float>& energy)
{
    // Find points where energy is non-increasing then increasing
    std::vector<int> minima;
    for (size_t i = 1; i < energy.size() - 1; ++i)
    {
        if (energy[i] <= energy[i - 1] && energy[i] < energy[i + 1])
        {
            minima.push_back(i);
        }
    }

    if (minima.empty())
    {
        return events; // No minima found, return original events
    }
    
    std::vector<int> backtrackedEvents;
    backtrackedEvents.reserve(events.size());

    for (int eventFrame : events)
    {
        // Find the closest minimum *before* the event
        auto it = std::lower_bound(minima.begin(), minima.end(), eventFrame);
        if (it != minima.begin())
        {
            --it; // Move to the last element that is not greater than eventFrame
            backtrackedEvents.push_back(*it);
        }
        else
        {
            // No preceding minimum found, use the first minimum as a fallback
            // or you could decide to keep the original event frame.
            // Librosa's `match_events` would effectively do this.
            backtrackedEvents.push_back(minima[0]);
        }
    }
    
    return backtrackedEvents;
}


std::vector<int> AudioAnalysis::detectOnsets(const std::vector<float>& onsetEnvelope)
{
    if (onsetEnvelope.empty())
    {
        return {};
    }

    // --- 1. Normalize onset envelope to [0, 1] ---
    std::vector<float> normalizedEnvelope = onsetEnvelope;
    float minVal = std::numeric_limits<float>::max();
    float maxVal = std::numeric_limits<float>::lowest();
    for (float val : normalizedEnvelope)
    {
        if (val < minVal) minVal = val;
        if (val > maxVal) maxVal = val;
    }

    float range = maxVal - minVal;
    if (range > 1e-8) // Avoid division by zero if envelope is flat
    {
        for (auto& val : normalizedEnvelope)
        {
            val = (val - minVal) / range;
        }
    }

    // --- 2. Set up parameters, mirroring librosa's defaults ---
    const float pre_max_s = 0.03f;
    const float post_max_s = 0.0f;
    const float pre_avg_s = 0.10f;
    const float post_avg_s = 0.10f;
    const float wait_s = 0.03f;
    const float delta = 0.07f;

    int pre_max = static_cast<int>(std::round(pre_max_s * sampleRate / hopSize));
    int post_max = static_cast<int>(std::round(post_max_s * sampleRate / hopSize)) + 1;
    int pre_avg = static_cast<int>(std::round(pre_avg_s * sampleRate / hopSize));
    int post_avg = static_cast<int>(std::round(post_avg_s * sampleRate / hopSize)) + 1;
    int wait = static_cast<int>(std::round(wait_s * sampleRate / hopSize));

    // --- 3. Peak picking ---
    std::vector<int> peaks;
    const int num_frames = normalizedEnvelope.size();
    
    for (int i = 0; i < num_frames; ++i)
    {
        // a. Check for local maximum
        float current_val = normalizedEnvelope[i];
        bool is_max = true;
        int max_start = std::max(0, i - pre_max);
        int max_end = std::min(num_frames, i + post_max);
        for (int j = max_start; j < max_end; ++j)
        {
            if (normalizedEnvelope[j] > current_val)
            {
                is_max = false;
                break;
            }
        }
        
        if (!is_max)
        {
            continue;
        }

        // b. Check against local average
        float avg = 0.0f;
        int avg_start = std::max(0, i - pre_avg);
        int avg_end = std::min(num_frames, i + post_avg);
        for (int j = avg_start; j < avg_end; ++j)
        {
            avg += normalizedEnvelope[j];
        }
        avg /= (avg_end - avg_start);

        if (current_val >= avg + delta)
        {
            peaks.push_back(i);
        }
    }
    
    // --- 4. Post-processing: apply wait time ---
    if (peaks.empty() || wait <= 0)
    {
        // Convert frames to samples for the final output
        std::vector<int> onset_samples;
        onset_samples.reserve(peaks.size());
        for (int frame : peaks)
        {
            onset_samples.push_back(frame * hopSize);
        }
        return onset_samples;
    }

    std::vector<int> final_peaks;
    final_peaks.push_back(peaks[0]);

    for (size_t i = 1; i < peaks.size(); ++i)
    {
        if (peaks[i] - final_peaks.back() > wait)
        {
            final_peaks.push_back(peaks[i]);
        }
        else if (normalizedEnvelope[peaks[i]] > normalizedEnvelope[final_peaks.back()])
        {
            // If a new peak is found within the refractory period,
            // replace the previous one if it's larger
            final_peaks.back() = peaks[i];
        }
    }
    
    // Convert final peak frames to samples
    std::vector<int> onset_samples;
    onset_samples.reserve(final_peaks.size());
    for (int frame : final_peaks)
    {
        onset_samples.push_back(frame * hopSize);
    }
    
    return onset_samples;
}


float AudioAnalysis::estimateTempo(const std::vector<float>& onsetEnvelope, float start_bpm, float std_bpm, float ac_size, float max_tempo)
{
    if (onsetEnvelope.empty()) return 120.0f;

    // --- Dynamic Window Size Calculation (matches librosa.time_to_frames) ---
    int acWinSizeFrames = static_cast<int>(std::round(ac_size * sampleRate / hopSize));
    lastTempogram = calculateTempogram(onsetEnvelope, acWinSizeFrames);

    if (lastTempogram.empty()) return 120.0f;

    // --- Aggregate Tempogram (matches aggregate=np.mean) ---
    const size_t numLags = lastTempogram.size();
    const size_t numFrames = numLags > 0 ? lastTempogram[0].size() : 0;
    
    std::vector<float> globalAcf(numLags, 0.0f);
    if (numFrames > 0)
    {
        for(size_t lag = 0; lag < numLags; ++lag)
        {
            for(size_t t = 0; t < numFrames; ++t)
            {
                globalAcf[lag] += lastTempogram[lag][t];
            }
            globalAcf[lag] /= (float)numFrames;
        }
    }
    
    lastGlobalAcf = globalAcf; // Store for UI

    // --- Librosa-style psychoacoustic tempo weighting (Static Prior) ---
    // (matches librosa.tempo_frequencies)
    std::vector<float> bpms(numLags, 0.0f);
    for(size_t i = 1; i < numLags; ++i)
    {
        bpms[i] = 60.0f * (float)sampleRate / ((float)hopSize * i);
    }

    // Create a log-normal prior centered at start_bpm (matches librosa default prior)
    std::vector<float> logprior(numLags, -std::numeric_limits<float>::infinity());
    for(size_t i = 1; i < numLags; ++i)
    {
        if (bpms[i] > 0)
        {
            float log2_diff = std::log2(bpms[i]) - std::log2(start_bpm);
            logprior[i] = -0.5f * std::pow(log2_diff / std_bpm, 2.0f);
        }
    }
    
    // Kill everything above max_tempo
    if (max_tempo > 0)
    {
        for (size_t i = 1; i < numLags; ++i)
        {
            if (bpms[i] >= max_tempo)
            {
                logprior[i] = -std::numeric_limits<float>::infinity();
            }
        }
    }
    
    // Find the peak in the aggregated ACF, weighted by the prior.
    // (matches np.log1p(1e6 * tg) + logprior)
    int bestLag = -1;
    float maxWeightedVal = -std::numeric_limits<float>::infinity();

    for (size_t lag = 1; lag < numLags; ++lag) // Start from 1, lag 0 is not useful for tempo
    {
        // The tempogram is already normalized per-frame in calculateTempogram,
        // so we use the aggregated value directly without re-normalizing.
        float weighted_val = std::log1p(globalAcf[lag] * 1e6f) + logprior[lag];

        if (weighted_val > maxWeightedVal)
        {
            maxWeightedVal = weighted_val;
            bestLag = lag;
        }
    }

    if (bestLag > 0)
    {
        return bpms[bestLag];
    }

    return 120.0f; // Fallback tempo
}

std::vector<std::vector<float>> AudioAnalysis::calculateTempogram(const std::vector<float>& onsetEnvelope, int winLength)
{
    // A 1:1 port of librosa.feature.tempogram
    if (onsetEnvelope.empty()) return {};

    const int n = onsetEnvelope.size();
    const int padding = winLength / 2;

    // 1. Pad the onset envelope with a linear ramp (as numpy.pad with end_values=0)
    std::vector<float> paddedEnvelope(n + 2 * padding, 0.0f);
    // Left pad (np.linspace(0, oenv[0], num=padding, endpoint=False))
    float firstVal = onsetEnvelope.empty() ? 0.0f : onsetEnvelope[0];
    for (int i = 0; i < padding; ++i)
    {
        paddedEnvelope[i] = firstVal * (float)i / (float)padding;
    }

    // Copy original data
    std::copy(onsetEnvelope.begin(), onsetEnvelope.end(), paddedEnvelope.begin() + padding);

    // Right pad (np.linspace(oenv[-1], 0, num=padding, endpoint=False))
    float lastVal = onsetEnvelope.empty() ? 0.0f : onsetEnvelope.back();
    for (int i = 0; i < padding; ++i)
    {
        paddedEnvelope[padding + n + i] = lastVal * (float)(padding - i) / (float)padding;
    }

    // 2. Create the autocorrelation window (Hann)
    std::vector<float> acWindow(winLength);
    for(int i = 0; i < winLength; ++i) {
        acWindow[i] = 0.5f - 0.5f * std::cos(2.0f * M_PI * i / (winLength - 1));
    }

    // Prepare for autocorrelation via FFT
    const int fftSize = juce::nextPowerOfTwo(winLength) * 2;
    juce::dsp::FFT fft(log2(fftSize));
    
    // Librosa's tempogram is [lag][time], so we match that layout
    std::vector<std::vector<float>> tempogram(winLength, std::vector<float>(n, 0.0f));

    // 3. Slide over the padded envelope, frame by frame
    for (int i = 0; i < n; ++i)
    {
        // Use a complex buffer for the FFT. JUCE expects [real, imag, real, imag, ...]
        std::vector<float> complexFrame(fftSize * 2, 0.0f);
        
        // 4. Extract and window the frame, placing it in the real part of the complex buffer
        for(int j=0; j < winLength; ++j) {
            complexFrame[j*2] = paddedEnvelope[i+j] * acWindow[j];
        }

        // 5. Autocorrelation via FFT
        // a) Forward FFT
        fft.perform(reinterpret_cast<juce::dsp::Complex<float>*>(complexFrame.data()),
                    reinterpret_cast<juce::dsp::Complex<float>*>(complexFrame.data()),
                    false);
        
        // b) Compute power spectrum (S * conj(S) = |S|^2) and store it as a real signal for IFFT
        for (int j = 0; j < fftSize; ++j)
        {
            float real = complexFrame[j*2];
            float imag = complexFrame[j*2+1];
            complexFrame[j*2] = real * real + imag * imag; // Magnitude squared in the real part
            complexFrame[j*2+1] = 0.0f;                   // Zero out the imaginary part
        }

        // c) Inverse FFT of the power spectrum gives the autocorrelation
        fft.perform(reinterpret_cast<juce::dsp::Complex<float>*>(complexFrame.data()),
                    reinterpret_cast<juce::dsp::Complex<float>*>(complexFrame.data()),
                    true);
        
        // 6. Normalize and store
        // The result of the IFFT is in the real parts of the complex data.
        float maxVal = 0.0f;
        for (int j = 0; j < winLength; ++j) {
            maxVal = std::max(maxVal, std::abs(complexFrame[j*2]));
        }

        if (maxVal > 0)
        {
            for (int j = 0; j < winLength; ++j)
            {
                tempogram[j][i] = complexFrame[j*2] / maxVal;
            }
        }
    }

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


std::vector<double> AudioAnalysis::findBeats(const std::vector<float>& onsetEnvelope, float bpm, double tightness, bool trim)
{
    if (onsetEnvelope.empty() || bpm <= 0)
    {
        return {};
    }

    // Check if there's any signal in the onset envelope (matches !onset_envelope.any())
    float onset_sum = 0.0f;
    for (float val : onsetEnvelope) {
        onset_sum += val;
    }
    if (onset_sum < 1e-8) {
        return {};
    }

    // Convert BPM to period in frames
    double framesPerSecond = this->sampleRate / this->hopSize;
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
        beatSampleFrames.push_back(frame * this->hopSize);
    }
    
    // --- Trim weak leading/trailing beats (conditional) ---
    if (trim)
    {
        trimBeats(localScore, beatSampleFrames, this->hopSize);
    }

    // Convert frames to seconds
    std::vector<double> beatTimes;
    for(int frame : beatSampleFrames)
    {
        beatTimes.push_back(static_cast<double>(frame) / this->sampleRate);
    }

    return beatTimes;
}

void AudioAnalysis::performDCT(std::vector<float>& input)
{
    int N = input.size();
    if (N == 0) return;

    std::vector<float> result(N);
    float c_k_factor = M_PI / N;
    float sqrt_1_N = std::sqrt(1.0f / N);
    float sqrt_2_N = std::sqrt(2.0f / N);

    for (int k = 0; k < N; ++k)
    {
        float sum = 0.0f;
        for (int n = 0; n < N; ++n)
        {
            sum += input[n] * std::cos(c_k_factor * (n + 0.5f) * k);
        }
        float c_k = (k == 0) ? sqrt_1_N : sqrt_2_N;
        result[k] = c_k * sum;
    }

    input = result;
}

std::vector<std::vector<float>> AudioAnalysis::calculateMFCCs(const juce::AudioBuffer<float>& buffer, double sampleRate, int numCoefficients)
{
    std::vector<std::vector<float>> allMfccs;
    const int fftSize = 1024;
    const int hopSize = 256;
    juce::dsp::FFT fft(log2(fftSize));
    auto* audioData = buffer.getReadPointer(0);
    int numSamples = buffer.getNumSamples();

    // Pre-allocate buffers outside the loop for performance
    std::vector<float> window(fftSize);
    std::vector<float> windowedData(fftSize);
    std::vector<float> fftBuffer(fftSize * 2, 0.0f);

    for(int j=0; j<fftSize; ++j) {
        window[j] = 0.5f - 0.5f * std::cos(2.0f * M_PI * j / (fftSize - 1));
    }

    for (int i = 0; i + fftSize <= numSamples; i += hopSize)
    {
        for(int j=0; j<fftSize; ++j) {
            windowedData[j] = audioData[i+j] * window[j];
        }

        std::fill(fftBuffer.begin(), fftBuffer.end(), 0.0f);
        std::copy(windowedData.begin(), windowedData.end(), fftBuffer.begin());

        fft.performFrequencyOnlyForwardTransform(fftBuffer.data());

        std::vector<float> melEnergies(melFilterbank.size(), 0.0f);
        for (size_t band = 0; band < melFilterbank.size(); ++band)
        {
            for (const auto& binWeightPair : melFilterbank[band])
            {
                melEnergies[band] += fftBuffer[binWeightPair.first] * binWeightPair.second;
            }
            melEnergies[band] = std::log(melEnergies[band] + 1e-6);
        }
        
        performDCT(melEnergies);

        melEnergies.resize(numCoefficients);
        allMfccs.push_back(melEnergies);
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

    // --- 2. Calculate similarity using the pre-ranked data ---
    std::vector<std::vector<float>> similarityMatrix(numSlices, std::vector<float>(numSlices, 0.0f));
    
    for (int i = 0; i < numSlices; ++i)
    {
        for (int j = i; j < numSlices; ++j)
        {
            float correlation = calculatePearsonCorrelationOnRanks(rankedMfccs[i], rankedMfccs[j]);
            similarityMatrix[i][j] = correlation;
            similarityMatrix[j][i] = correlation; // Matrix is symmetric
        }
    }

    return similarityMatrix;
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
