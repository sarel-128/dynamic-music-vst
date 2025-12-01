#include "AudioRetarget.h"
#include <future>
#include <thread>
#include <vector>

AudioRetargeter::AudioRetargeter()
{
}

AudioRetargeter::~AudioRetargeter()
{
}

AudioRetargeter::RetargetResult AudioRetargeter::retargetDuration(const std::vector<std::vector<float>>& similarityMatrix,
                                                   const std::vector<float>& beats,
                                                   float targetDuration,
                                                   float similarityPenalty,
                                                   float backwardJumpPenalty,
                                                   float timeContinuityPenalty,
                                                   float constraintTimePenalty,
                                                   float constraintBeatPenalty,
                                                   const std::vector<std::pair<float, float>>& timeConstraints)
{
    if (beats.empty() || targetDuration <= 0.0f)
    {
        return { {}, std::numeric_limits<float>::infinity() };
    }
    
    DBG(" ");
    DBG("--- Inside retargetDuration ---");
    auto functionStartTime = juce::Time::getMillisecondCounterHiRes();


    // Calculate target number of beats
    float secondsPerBeat = 0.0f;
    if (beats.size() > 1)
    {
        secondsPerBeat = (beats.back() - beats.front()) / (float)(beats.size() - 1);
    }
    else
    {
        secondsPerBeat = 1.0f; 
    }

    int targetNumBeats = static_cast<int>(targetDuration / secondsPerBeat);
    if (targetNumBeats <= 0) return { {}, std::numeric_limits<float>::infinity() };

    int numBeats = static_cast<int>(similarityMatrix.size());
    if (numBeats == 0) return { {}, std::numeric_limits<float>::infinity() };

    // Parse constraints into a list of points
    auto stepStartTime = juce::Time::getMillisecondCounterHiRes();
    struct ConstraintPoint { int step; int beat; };
    std::vector<ConstraintPoint> constraintPoints;

    if (!timeConstraints.empty())
    {
        for (const auto& c : timeConstraints)
        {
            float srcTime = c.first;
            float tgtTime = c.second;

            // Find closest beat index for source time
            int bestBeatIdx = 0;
            float minDiff = std::numeric_limits<float>::max();
            for (int b = 0; b < (int)beats.size(); ++b)
            {
                float diff = std::abs(beats[b] - srcTime);
                if (diff < minDiff)
                {
                    minDiff = diff;
                    bestBeatIdx = b;
                }
            }
            if (bestBeatIdx >= numBeats) bestBeatIdx = numBeats - 1;

            // Find step index for target time
            int stepIdx = static_cast<int>(tgtTime / secondsPerBeat);
            if (stepIdx < 0) stepIdx = 0;
            if (stepIdx >= targetNumBeats) stepIdx = targetNumBeats - 1;

            constraintPoints.push_back({stepIdx, bestBeatIdx});
        }
    }
    else
    {
        // Default constraints if none provided: Start -> Start, End -> End
        constraintPoints.push_back({0, 0});
        constraintPoints.push_back({targetNumBeats - 1, numBeats - 1});
    }
    DBG("1. Constraint Parsing: " << juce::String(juce::Time::getMillisecondCounterHiRes() - stepStartTime, 2) << " ms");

    // DP table for costs
    std::vector<std::vector<float>> cost(targetNumBeats, std::vector<float>(numBeats, std::numeric_limits<float>::infinity()));
    // DP table for path backtracking
    std::vector<std::vector<int>> path(targetNumBeats, std::vector<int>(numBeats, -1));

    // Initialization
    stepStartTime = juce::Time::getMillisecondCounterHiRes();
    for (int j = 0; j < numBeats; ++j)
    {
        // Calculate initial cost based on constraints (soft start)
        float minWeightedDist = std::numeric_limits<float>::max();
        for (const auto& pt : constraintPoints)
        {
             // Distance at step 0
            float dTime = std::abs((float)0 - pt.step) / (float)targetNumBeats;
            float dBeat = std::abs((float)j - pt.beat) / (float)numBeats;
            
            float wDist = std::sqrt(std::pow(constraintTimePenalty * dTime, 2) + std::pow(constraintBeatPenalty * dBeat, 2));
            if (wDist < minWeightedDist) minWeightedDist = wDist;
        }
        cost[0][j] = minWeightedDist;
    }
    DBG("2. DP Initialization: " << juce::String(juce::Time::getMillisecondCounterHiRes() - stepStartTime, 2) << " ms");

    // Fill DP table
    stepStartTime = juce::Time::getMillisecondCounterHiRes();
    const int searchWindowRadius = 50; // Only search +/- this many beats from current

    for (int i = 1; i < targetNumBeats; ++i)
    {
        // Calculate the ideal beat index for this time step to enforce global structure (Linear Time Scaling Baseline)
        float progress = (float)i / (float)(targetNumBeats - 1);
        float idealBeatIndex = progress * (numBeats - 1);

        auto process_j_range = [&](int start_j, int end_j)
        {
            for (int j = start_j; j < end_j; ++j)
            {
                float minCost = std::numeric_limits<float>::infinity();
                int bestPrevBeat = -1;

                // Penalty for deviating from the ideal time curve (Global Linear Guide)
                float timeDeviationCost = timeContinuityPenalty * std::abs((float)j - idealBeatIndex) / (float)numBeats;
                
                // Soft Constraint Penalty (Local Attractors)
                float constraintCost = 0.0f;
                if (!constraintPoints.empty())
                {
                    float minWeightedDist = std::numeric_limits<float>::max();
                    for (const auto& pt : constraintPoints)
                    {
                        float dTime = std::abs((float)i - pt.step) / (float)targetNumBeats;
                        float dBeat = std::abs((float)j - pt.beat) / (float)numBeats;
                        
                        float wDist = std::sqrt(std::pow(constraintTimePenalty * dTime, 2) + std::pow(constraintBeatPenalty * dBeat, 2));
                        if (wDist < minWeightedDist) minWeightedDist = wDist;
                    }
                    constraintCost = minWeightedDist;
                }

                for (int k = std::max(0, j - searchWindowRadius); k < std::min(numBeats, j + searchWindowRadius); ++k)
                {
                    if (k + 1 >= numBeats) continue;
                    if (cost[i - 1][k] == std::numeric_limits<float>::infinity()) continue;

                    float dist = 1.0f - similarityMatrix[k + 1][j];
                    float transitionCost = similarityPenalty * dist;

                    if (j < k)
                    {
                        transitionCost += backwardJumpPenalty;
                    }

                    float currentCost = cost[i - 1][k] + transitionCost + timeDeviationCost + constraintCost;

                    if (currentCost < minCost)
                    {
                        minCost = currentCost;
                        bestPrevBeat = k;
                    }
                }
                cost[i][j] = minCost;
                path[i][j] = bestPrevBeat;
            }
        };
        
        unsigned int num_threads = std::max(1u, std::thread::hardware_concurrency());
        std::vector<std::future<void>> futures;
        int chunk_size = numBeats / num_threads;

        for (unsigned int t = 0; t < num_threads; ++t)
        {
            int start_j = t * chunk_size;
            int end_j = (t == num_threads - 1) ? numBeats : start_j + chunk_size;
            if (start_j < end_j)
                futures.push_back(std::async(std::launch::async, process_j_range, start_j, end_j));
        }

        for (auto& f : futures)
        {
            f.get();
        }
    }
    DBG("3. Fill DP Table: " << juce::String(juce::Time::getMillisecondCounterHiRes() - stepStartTime, 2) << " ms");

    // Backtrack to find the best path
    stepStartTime = juce::Time::getMillisecondCounterHiRes();
    std::vector<int> resultPath(targetNumBeats);
    float finalCost = std::numeric_limits<float>::infinity();
    int endBeat = -1;

    // Find the minimal cost beat at the last step
    for (int j = 0; j < numBeats; ++j)
    {
        if (cost[targetNumBeats - 1][j] < finalCost)
        {
            finalCost = cost[targetNumBeats - 1][j];
            endBeat = j;
        }
    }

    if (targetNumBeats > 0 && numBeats > 0)
    {
        if (finalCost != std::numeric_limits<float>::infinity() && endBeat != -1)
        {
            resultPath[targetNumBeats - 1] = endBeat;

            for (int i = targetNumBeats - 1; i > 0; --i)
            {
                resultPath[i - 1] = path[i][resultPath[i]];
            }
        }
        else
        {
             return { {}, std::numeric_limits<float>::infinity() };
        }
    }
    DBG("4. Backtracking: " << juce::String(juce::Time::getMillisecondCounterHiRes() - stepStartTime, 2) << " ms");
    DBG("Total internal time: " << juce::String(juce::Time::getMillisecondCounterHiRes() - functionStartTime, 2) << " ms");
    DBG("-----------------------------");


    // Return average cost per beat
    return { resultPath, finalCost / (float)targetNumBeats };
}
