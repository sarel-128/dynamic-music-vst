#pragma once

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_core/juce_core.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <vector>
#include <thread>

enum class AggregationMethod {
    Mean,
    Median
};

class AudioAnalysis
{
public:
    AudioAnalysis();
    
    // Onset detection
    std::vector<float> getOnsetStrengthEnvelope(const juce::AudioBuffer<float>& buffer,
                                                double sampleRate,
                                                int hopSize = 512,
                                                int fftSize = 2048,
                                                int nBands = 128,
                                                int lag = 1,
                                                int max_size = 1,
                                                AggregationMethod aggregate = AggregationMethod::Median);
    std::vector<int> detectOnsets(const std::vector<float>& onsetEnvelope, int hopSize);
    
    // Tempo and Beat tracking
    float estimateTempo(const std::vector<float>& onsetEnvelope,
                        float sampleRate,
                        int hopSize,
                        float start_bpm = 120.0f,
                        float std_bpm = 1.0f,
                        float ac_size = 8.0f,
                        float max_tempo = 320.0f);
    std::vector<std::vector<float>> calculateTempogram(const std::vector<float>& onsetEnvelope, int winLength);
    std::vector<double> findBeats(const std::vector<float>& onsetEnvelope, float bpm, float sampleRate, int hopSize, double tightness = 100.0);

    // MFCC and Similarity
    std::vector<std::vector<float>> calculateMFCCs(const juce::AudioBuffer<float>& buffer,
                                                   double sampleRate,
                                                   int numCoefficients = 40,
                                                   int fftSize = 2048,
                                                   int hopSize = 512);
    std::vector<std::vector<float>> concatenateMFCCsByBeats(const std::vector<std::vector<float>>& allMfccs, const std::vector<double>& beats, double sampleRate, int hopSize);
    std::vector<std::vector<float>> createSimilarityMatrix(const std::vector<std::vector<float>>& mfccs);
    float calculateSpearmanCorrelation(const std::vector<float>& x, const std::vector<float>& y);

    // Public getter for the last calculated tempogram
    const std::vector<std::vector<float>>& getLastTempogram() const { return lastTempogram; }
    const std::vector<float>& getLastGlobalAcf() const { return lastGlobalAcf; }

private:
    // Store last used params for the Mel filterbank to avoid recalculating it unnecessarily
    int lastNbands = 0;
    int lastFftSize = 0;
    double lastSampleRate = 0.0;

    std::vector<float> getTempoFrequencies(int numBins, float sampleRate, int hopSize);
    void performDCT(std::vector<float>& input);
    double hzToMel(double hz);
    double melToHz(double mel);
    void createMelFilterbank(int numBands, int fftSize, double sampleRate);
    std::vector<float> createLocalScore(const std::vector<float>& onsetEnvelope, double period);
    void trimBeats(const std::vector<float>& localScore, std::vector<int>& beatFrames, int hopSize);
    const std::vector<std::vector<float>>& getPowerMelSpectrogram(const juce::AudioBuffer<float>& buffer, double sampleRate, int fftSize, int hopSize);

    // Store the filterbank as a sparse representation for efficiency
    std::vector<std::vector<std::pair<int, float>>> melFilterbank;
    std::vector<std::vector<float>> lastTempogram;
    std::vector<float> lastGlobalAcf;

    // --- For FFT-based DCT ---
    std::unique_ptr<juce::dsp::FFT> dctfft;
    int dctfftOrder = 0;
    std::vector<juce::dsp::Complex<float>> dctfftWorkspace;

    // --- For Tempogram Calculation ---
    std::unique_ptr<juce::dsp::FFT> tempogramfft;
    int tempogramfftOrder = 0;
    std::vector<juce::dsp::Complex<float>> tempogramfftWorkspace1;
    std::vector<juce::dsp::Complex<float>> tempogramfftWorkspace2;

    // --- Mel Spectrogram Cache ---
    // We cache the raw power spectrogram to share between onset and MFCC calculations
    std::vector<std::vector<float>> cachedPowerMelSpectrogram;
    int cachedFftSize = 0;
    int cachedHopSize = 0;
    double cachedSampleRate = 0.0;
    // Keep track of the buffer's pointer and size to detect audio changes
    const float* cachedAudioDataPtr = nullptr;
    int cachedNumSamples = 0;
};
