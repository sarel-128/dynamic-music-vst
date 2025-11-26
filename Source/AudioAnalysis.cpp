#include "AudioAnalysis.h"
#include <juce_dsp/juce_dsp.h>
#include <cmath>
#include <numeric>

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
    const int hopSize = 512;
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

        // Calculate spectral flux on the Mel spectrum
        float flux = 0.0f;
        for (size_t band = 0; band < melFilterbank.size(); ++band) {
            float spectralDifference = currentMelSpectrum[band] - lastMelSpectrum[band];
            if (spectralDifference > 0) flux += spectralDifference;
        }
        
        onsetEnvelope.push_back(flux);
        lastMelSpectrum = currentMelSpectrum;
    }

    return onsetEnvelope;
}

std::vector<int> AudioAnalysis::detectOnsets(const std::vector<float>& onsetEnvelope)
{
    std::vector<int> onsets;
    const int hopSize = 512; // Must match the hop size from getOnsetStrengthEnvelope
    
    // Guard against envelopes that are too small to have a peak
    if (onsetEnvelope.size() < 3)
    {
        return onsets;
    }

    // Simple peak picking on the onset envelope
    // A more robust solution would use a moving average threshold
    for (size_t i = 1; i < onsetEnvelope.size() - 1; ++i)
    {
        if (onsetEnvelope[i] > onsetEnvelope[i - 1] && onsetEnvelope[i] > onsetEnvelope[i + 1] && onsetEnvelope[i] > 0.1)
        {
            onsets.push_back(i * hopSize);
        }
    }
    return onsets;
}


float AudioAnalysis::estimateTempo(const std::vector<float>& onsetEnvelope, float sampleRate, int hopSize)
{
    if (onsetEnvelope.empty()) return 120.0f;

    // Use FFT for fast autocorrelation (O(N log N) complexity)
    const int originalSize = (int)onsetEnvelope.size();
    const int fftSize = juce::nextPowerOfTwo(originalSize * 2);

    // Prepare complex buffers for FFT
    using Complex = juce::dsp::Complex<float>;
    std::vector<Complex> fftInput(fftSize, Complex{0.0f, 0.0f});
    std::vector<Complex> fftOutput(fftSize);

    // Copy real signal into complex buffer
    for(int i = 0; i < originalSize; ++i)
        fftInput[i].real(onsetEnvelope[i]);

    // Perform forward FFT
    juce::dsp::FFT fft(log2(fftSize));
    fft.perform(fftInput.data(), fftOutput.data(), false);

    // Compute the power spectrum (complex conjugate multiplication)
    for(int i = 0; i < fftSize; ++i)
        fftOutput[i] = fftOutput[i] * std::conj(fftOutput[i]);
    
    // Perform inverse FFT to get the autocorrelation
    fft.perform(fftOutput.data(), fftInput.data(), true);
    
    // The result in fftInput.real() is now the Autocorrelation Function (ACF)
    
    // Find the peak in the ACF in a plausible tempo range
    float framesPerSecond = sampleRate / hopSize;
    int minLag = static_cast<int>(framesPerSecond * 60.0f / 200.0f); // 200 BPM
    int maxLag = static_cast<int>(framesPerSecond * 60.0f / 60.0f);   // 60 BPM

    maxLag = std::min(maxLag, originalSize - 1);
    minLag = std::max(1, minLag);

    if (minLag >= maxLag)
    {
        return 120.0f; // Default if range is invalid
    }

    int bestLag = -1;
    float maxVal = -1.0f;
    for (int lag = minLag; lag <= maxLag; ++lag)
    {
        // We divide by acf[0] to normalize the ACF, which can improve peak picking.
        float currentVal = fftInput[lag].real();
        if (fftInput[0].real() > 0 && currentVal / fftInput[0].real() > maxVal)
        {
            maxVal = currentVal / fftInput[0].real();
            bestLag = lag;
        }
    }

    if (bestLag > 0)
    {
        float tempo = 60.0f * framesPerSecond / bestLag;

        // Normalize tempo to be within a plausible range (e.g., 65-200 BPM)
        while (tempo < 65.0f) tempo *= 2.0f;
        while (tempo > 200.0f) tempo /= 2.0f;
        
        return tempo;
    }

    return 120.0f; // Default tempo
}

std::vector<double> AudioAnalysis::findBeats(const std::vector<float>& onsetEnvelope, float bpm, float sampleRate, int hopSize, double tightness, double inertia, double onsetWeight, double activationThreshold)
{
    if (onsetEnvelope.empty() || bpm <= 0)
    {
        return {};
    }

    // --- Normalize the onset envelope ---
    std::vector<float> normalizedEnvelope = onsetEnvelope;
    float maxVal = *std::max_element(normalizedEnvelope.begin(), normalizedEnvelope.end());
    if (maxVal > 0)
    {
        for (auto& val : normalizedEnvelope)
        {
            val /= maxVal;
        }
    }

    // --- Apply Activation Threshold ---
    for (auto& val : normalizedEnvelope)
    {
        val = std::max(0.0f, val - (float)activationThreshold);
    }


    // Convert BPM to period in frames
    double framesPerSecond = sampleRate / hopSize;
    double period = framesPerSecond * 60.0 / bpm;

    std::vector<double> cumulativeScore(normalizedEnvelope.size());
    std::vector<int> backlink(normalizedEnvelope.size(), -1);

    // Initialize scores, now weighted by the onsetWeight parameter
    for (size_t i = 0; i < normalizedEnvelope.size(); ++i)
    {
        cumulativeScore[i] = onsetWeight * normalizedEnvelope[i];
    }

    // DP loop
    int searchStart = static_cast<int>(std::floor(period / 2.0));
    int searchEnd = static_cast<int>(std::ceil(period * 2.0));

    // Store the period of the last chosen step for each frame
    std::vector<int> backlinkPeriod(onsetEnvelope.size(), 0);

    for (int i = 0; i < (int)normalizedEnvelope.size(); ++i)
    {
        double maxScore = -1.0;
        int bestPrev = -1;
        int bestPeriod = 0;

        for (int p = searchStart; p < searchEnd; ++p)
        {
            int j = i - p;
            if (j < 0) continue;

            // Global tempo penalty (original)
            double globalPenalty = -tightness * std::pow(std::log2((double)p / period), 2.0);

            // Local tempo consistency penalty
            double localPenalty = 0.0;
            if (backlink[j] > 0) // If the previous frame has a valid predecessor
            {
                int prevPeriod = backlinkPeriod[j];
                if (prevPeriod > 0)
                {
                    // Penalize deviation from the previous beat's interval.
                    localPenalty = -inertia * std::pow(std::log2((double)p / prevPeriod), 2.0);
                }
            }
            
            double score = cumulativeScore[j] + globalPenalty + localPenalty;

            if (score > maxScore)
            {
                maxScore = score;
                bestPrev = j;
                bestPeriod = p;
            }
        }
        
        if (bestPrev != -1)
        {
            cumulativeScore[i] += maxScore;
            backlink[i] = bestPrev;
            backlinkPeriod[i] = bestPeriod;
        }
    }

    // Backtrack from the best ending point
    int currentFrame = std::distance(cumulativeScore.begin(), std::max_element(cumulativeScore.begin(), cumulativeScore.end()));
    
    std::vector<int> beatFrames;
    while(currentFrame >= 0)
    {
        beatFrames.push_back(currentFrame);
        currentFrame = backlink[currentFrame];
    }
    std::reverse(beatFrames.begin(), beatFrames.end());

    // Convert frames to seconds
    std::vector<double> beatTimes;
    for(int frame : beatFrames)
    {
        beatTimes.push_back(static_cast<double>(frame * hopSize) / sampleRate);
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
    const int hopSize = 512;
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
