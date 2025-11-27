#include "AudioAnalysis.h"
#include <juce_dsp/juce_dsp.h>
#include <cmath>
#include <numeric>
#include <limits>

AudioAnalysis::AudioAnalysis()
{
    // Pre-calculate the filterbank for a standard FFT size and sample rate.
    // A more advanced implementation might recreate this if the sample rate changes.
    createMelFilterbank(40, 1024, 44100.0);
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
    const int fftSize = 1024;
    const int hopSize = 256;
    juce::dsp::FFT fft(log2(fftSize));
    auto* audioData = buffer.getReadPointer(0); // Analyze mono
    int numSamples = buffer.getNumSamples();
    std::vector<float> onsetEnvelope;

    // Pre-allocate buffers outside the loop for performance
    std::vector<float> window(fftSize);
    std::vector<float> windowedData(fftSize);
    std::vector<float> fftBuffer(fftSize * 2, 0.0f);
    std::vector<float> currentMelSpectrum(melFilterbank.size(), 0.0f);
    std::vector<float> lastMelSpectrum(melFilterbank.size(), 0.0f);

    for(int j=0; j<fftSize; ++j) {
        window[j] = 0.5f - 0.5f * cos(2.0f * M_PI * j / (fftSize - 1));
    }

    for (int i = 0; i + fftSize < numSamples; i += hopSize)
    {
        for(int j=0; j<fftSize; ++j) {
            windowedData[j] = audioData[i+j] * window[j];
        }

        std::fill(fftBuffer.begin(), fftBuffer.end(), 0.0f);
        std::copy(windowedData.begin(), windowedData.end(), fftBuffer.begin());

        fft.performFrequencyOnlyForwardTransform(fftBuffer.data());
        
        // Apply the Mel filterbank to the FFT output (sparse multiplication)
        std::fill(currentMelSpectrum.begin(), currentMelSpectrum.end(), 0.0f);
        for (size_t band = 0; band < melFilterbank.size(); ++band)
        {
            for (const auto& binWeightPair : melFilterbank[band])
            {
                currentMelSpectrum[band] += fftBuffer[binWeightPair.first] * binWeightPair.second;
            }
        }

        // // --- Logarithmic Compression of the Mel Spectrum ---
        // // This reduces the dynamic range, making onsets in quiet sections
        // // more comparable to onsets in loud sections.
        // for(auto& val : currentMelSpectrum) {
        //     val = std::log1p(val * 100.0f); // Scale factor can be tuned
        // }

        // Calculate spectral flux on the Mel spectrum
        float flux = 0.0f;
        for (size_t band = 0; band < melFilterbank.size(); ++band) {
            float spectralDifference = currentMelSpectrum[band] - lastMelSpectrum[band];
            if (spectralDifference > 0)
            {
                // Weight the flux by the Mel band index. This emphasizes changes
                // in higher frequencies, which often correspond to sharper transients,
                // providing a clearer rhythmic pulse.
                float weight = static_cast<float>(band + 1);
                flux += spectralDifference * weight;
            }
        }
        
        onsetEnvelope.push_back(flux);
        lastMelSpectrum = currentMelSpectrum;
    }

    return onsetEnvelope;
}

std::vector<int> AudioAnalysis::detectOnsets(const std::vector<float>& onsetEnvelope)
{
    std::vector<int> onsets;
    const int hopSize = 256; // Must match the hop size from getOnsetStrengthEnvelope
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


float AudioAnalysis::estimateTempo(const std::vector<float>& onsetEnvelope, float sampleRate, int hopSize)
{
    if (onsetEnvelope.empty()) return 120.0f;

    // // --- Preprocessing: Median filtering to enhance the main pulse ---
    // std::vector<float> processedEnvelope = onsetEnvelope;
    // const int medianFilterSize = 7; // A small, odd-sized window
    // if (processedEnvelope.size() > medianFilterSize)
    // {
    //     std::vector<float> temp(medianFilterSize);
    //     for (size_t i = medianFilterSize / 2; i < processedEnvelope.size() - medianFilterSize / 2; ++i)
    //     {
    //         for (int j = 0; j < medianFilterSize; ++j)
    //         {
    //             temp[j] = onsetEnvelope[i + j - medianFilterSize / 2];
    //         }
    //         std::sort(temp.begin(), temp.end());
    //         processedEnvelope[i] = temp[medianFilterSize / 2];
    //     }
    // }

    std::vector<float> processedEnvelope = onsetEnvelope;

    // --- Tempogram Calculation and Aggregation ---
    // Instead of a single global autocorrelation, we compute a tempogram (local
    // autocorrelations over time) and average it. This gives a much more
    // stable representation of the song's overall rhythm.
    const int acWinSizeFrames = 384; // Corresponds to ~8.9s, librosa's default
    lastTempogram = calculateTempogram(processedEnvelope, acWinSizeFrames);

    if (lastTempogram.empty()) return 120.0f;

    std::vector<float> globalAcf(acWinSizeFrames, 0.0f);
    for (const auto& frameAcf : lastTempogram)
    {
        for (int i = 0; i < acWinSizeFrames; ++i)
        {
            globalAcf[i] += frameAcf[i];
        }
    }
    float numFrames = static_cast<float>(lastTempogram.size());
    for (float& val : globalAcf)
    {
        val /= numFrames;
    }
    lastGlobalAcf = globalAcf; // Store for UI

    // --- Tempo Estimation via raw autocorrelation peak picking on the aggregated ACF ---
    const int originalSize = acWinSizeFrames;

    // 1. Define plausible tempo range in terms of lag frames
    float framesPerSecond = sampleRate / hopSize;
    int minLag = static_cast<int>(framesPerSecond * 60.0f / 200.0f); // Max tempo 200 BPM
    int maxLag = static_cast<int>(framesPerSecond * 60.0f / 55.0f);   // Min tempo 55 BPM
    maxLag = std::min(maxLag, originalSize - 1);
    minLag = std::max(1, minLag);

    if (minLag >= maxLag) return 120.0f;

    float acf_norm = globalAcf[0];
    if (acf_norm <= 0) acf_norm = 1.0f;

    // --- Find the strongest peak in the aggregated autocorrelation ---
    int bestLag = -1;
    float maxAcfVal = -1.0f;
    for (int lag = minLag; lag <= maxLag; ++lag)
    {
        float currentVal = globalAcf[lag] / acf_norm;
        if (currentVal > maxAcfVal)
        {
            maxAcfVal = currentVal;
            bestLag = lag;
        }
    }

    if (bestLag > 0)
    {
        float tempo = 60.0f * framesPerSecond / bestLag;
        return tempo;
    }

    return 120.0f; // Fallback tempo
}

std::vector<std::vector<float>> AudioAnalysis::calculateTempogram(const std::vector<float>& onsetEnvelope, int winLength)
{
    if (onsetEnvelope.empty() || winLength < 1) return {};

    // 1. Pad the onset envelope to center the analysis windows (zero-padding)
    int padding = winLength / 2;
    std::vector<float> paddedEnvelope(onsetEnvelope.size() + 2 * padding, 0.0f);
    std::copy(onsetEnvelope.begin(), onsetEnvelope.end(), paddedEnvelope.begin() + padding);

    // 2. Create the windowing function
    std::vector<float> hanningWindow(winLength);
    for (int i = 0; i < winLength; ++i)
    {
        hanningWindow[i] = 0.5f - 0.5f * std::cos(2.0f * M_PI * i / (winLength - 1));
    }

    // 3. Slide over the padded envelope and compute local autocorrelation for each frame
    std::vector<std::vector<float>> tempogram;
    int numFrames = onsetEnvelope.size();
    tempogram.reserve(numFrames);

    const int fftSize = juce::nextPowerOfTwo(winLength * 2);
    juce::dsp::FFT fft(log2(fftSize));

    for (int i = 0; i < numFrames; ++i)
    {
        // The frame starts at i in the padded envelope, which corresponds to the
        // original onset frame being at the center of the window.
        std::vector<float> frame(winLength);
        for(int j=0; j<winLength; ++j)
        {
            frame[j] = paddedEnvelope[i+j] * hanningWindow[j];
        }

        // --- Autocorrelation via FFT ---
        using Complex = juce::dsp::Complex<float>;
        std::vector<Complex> fftInput(fftSize, Complex{0.0f, 0.0f});
        std::vector<Complex> fftOutput(fftSize);

        for(size_t j = 0; j < frame.size(); ++j)
            fftInput[j].real(frame[j]);

        fft.perform(fftInput.data(), fftOutput.data(), false);

        for(int j = 0; j < fftSize; ++j)
            fftOutput[j] = fftOutput[j] * std::conj(fftOutput[j]);
        
        fft.perform(fftOutput.data(), fftInput.data(), true);

        std::vector<float> acf(winLength);
        for(int j=0; j<winLength; ++j)
        {
            acf[j] = fftInput[j].real();
        }
        tempogram.push_back(acf);
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
