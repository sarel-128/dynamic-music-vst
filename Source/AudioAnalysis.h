#pragma once

#include <juce_audio_basics/juce_audio_basics.h>
#include <vector>

class AudioAnalysis
{
public:
    AudioAnalysis();

    std::vector<float> getOnsetStrengthEnvelope(const juce::AudioBuffer<float>& buffer);
    std::vector<int> detectOnsets(const std::vector<float>& onsetEnvelope);
    float estimateTempo(const std::vector<float>& onsetEnvelope, float sampleRate, int hopSize);
    std::vector<double> findBeats(const std::vector<float>& onsetEnvelope, float bpm, float sampleRate, int hopSize, double tightness = 1.0, double inertia = 0.5, double onsetWeight = 1.0, double activationThreshold = 0.1);
    std::vector<std::vector<float>> calculateMFCCs(const juce::AudioBuffer<float>& buffer, double sampleRate, int numCoefficients = 40);
    std::vector<std::vector<float>> concatenateMFCCsByBeats(const std::vector<std::vector<float>>& allMfccs, const std::vector<double>& beats, double sampleRate, int hopSize);
    std::vector<std::vector<float>> createSimilarityMatrix(const std::vector<std::vector<float>>& mfccs);

private:
    float calculateSpearmanCorrelation(const std::vector<float>& x, const std::vector<float>& y);
    void performDCT(std::vector<float>& input);
    double hzToMel(double hz);
    double melToHz(double mel);
    void createMelFilterbank(int numBands, int fftSize, double sampleRate);

    // Store the filterbank as a sparse representation for efficiency
    std::vector<std::vector<std::pair<int, float>>> melFilterbank;
};
