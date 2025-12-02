#pragma once

#include <juce_core/juce_core.h>
#include <vector>

// Forward declaration
struct ConstraintPoint;

class AudioRetargeter
{
public:
    AudioRetargeter();
    ~AudioRetargeter();

    struct RetargetResult
    {
        std::vector<int> path;
        float cost;
    };

    // Full retargeting - processes entire timeline
    RetargetResult retargetDuration(const std::vector<std::vector<float>>& similarityMatrix,
                                      const std::vector<float>& beats,
                                      float targetDuration,
                                      float similarityPenalty,
                                      float backwardJumpPenalty,
                                      float timeContinuityPenalty,
                                      float constraintTimePenalty,
                                      float constraintBeatPenalty,
                                      const std::vector<std::pair<float, float>>& timeConstraints);

    // Segmented retargeting - processes only a slice of the timeline between anchors
    RetargetResult retargetSegment(const std::vector<std::vector<float>>& similarityMatrix,
                                    const std::vector<float>& beats,
                                    int startStep,           // Time boundary start (in steps)
                                    int endStep,             // Time boundary end (in steps)
                                    int startBeat,           // Fixed anchor beat at startStep
                                    int endBeat,             // Fixed anchor beat at endStep
                                    float secondsPerBeat,    // For time calculations
                                    const std::vector<ConstraintPoint>& constraintsInSegment,
                                    float similarityPenalty,
                                    float backwardJumpPenalty,
                                    float timeContinuityPenalty,
                                    float constraintTimePenalty,
                                    float constraintBeatPenalty);

private:
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AudioRetargeter)
};
