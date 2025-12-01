#pragma once

#include <juce_core/juce_core.h>
#include <vector>

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

    RetargetResult retargetDuration(const std::vector<std::vector<float>>& similarityMatrix,
                                      const std::vector<float>& beats,
                                      float targetDuration,
                                      float similarityPenalty,
                                      float backwardJumpPenalty,
                                      float timeContinuityPenalty,
                                      float constraintTimePenalty,
                                      float constraintBeatPenalty,
                                      const std::vector<std::pair<float, float>>& timeConstraints);

private:
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AudioRetargeter)
};
