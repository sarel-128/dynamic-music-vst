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
    std::vector<double> findBeats(const std::vector<float>& onsetEnvelope, float bpm, float sampleRate, int hopSize, double tightness = 100.0);
    std::vector<std::vector<float>> calculateTempogram(const std::vector<float>& onsetEnvelope, int winLength);
    std::vector<std::vector<float>> calculateMFCCs(const juce::AudioBuffer<float>& buffer, double sampleRate, int numCoefficients = 40);
    std::vector<std::vector<float>> concatenateMFCCsByBeats(const std::vector<std::vector<float>>& allMfccs, const std::vector<double>& beats, double sampleRate, int hopSize);
    std::vector<std::vector<float>> createSimilarityMatrix(const std::vector<std::vector<float>>& mfccs);

    const std::vector<std::vector<float>>& getLastTempogram() const { return lastTempogram; }
    const std::vector<float>& getLastGlobalAcf() const { return lastGlobalAcf; }

private:
    float calculateSpearmanCorrelation(const std::vector<float>& x, const std::vector<float>& y);
    void performDCT(std::vector<float>& input);
    double hzToMel(double hz);
    double melToHz(double mel);
    void createMelFilterbank(int numBands, int fftSize, double sampleRate);
    std::vector<float> createLocalScore(const std::vector<float>& onsetEnvelope, double period);
    void trimBeats(const std::vector<float>& localScore, std::vector<int>& beatFrames, int hopSize);

    // Store the filterbank as a sparse representation for efficiency
    std::vector<std::vector<std::pair<int, float>>> melFilterbank;
    std::vector<std::vector<float>> lastTempogram;
    std::vector<float> lastGlobalAcf;
};
