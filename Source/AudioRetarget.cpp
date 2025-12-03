#include "AudioRetarget.h"
#include "PluginProcessor.h"  // For ConstraintPoint struct
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
                                                   const std::vector<std::pair<float, float>>& timeConstraints)
{
    if (beats.empty() || targetDuration <= 0.0f)
    {
        return { {}, std::numeric_limits<float>::infinity(), -1, -1 };
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
    if (targetNumBeats <= 0) return { {}, std::numeric_limits<float>::infinity(), -1, -1 };

    int numBeats = static_cast<int>(similarityMatrix.size());
    if (numBeats == 0) return { {}, std::numeric_limits<float>::infinity(), -1, -1 };

    auto stepStartTime = juce::Time::getMillisecondCounterHiRes();

    // DP table for costs
    std::vector<std::vector<float>> cost(targetNumBeats, std::vector<float>(numBeats, std::numeric_limits<float>::infinity()));
    // DP table for path backtracking
    std::vector<std::vector<int>> path(targetNumBeats, std::vector<int>(numBeats, -1));

    // Initialization
    for (int j = 0; j < numBeats; ++j)
    {
        cost[0][j] = 0.0f;
    }
    DBG("1. DP Initialization: " << juce::String(juce::Time::getMillisecondCounterHiRes() - stepStartTime, 2) << " ms");

    // Fill DP table
    stepStartTime = juce::Time::getMillisecondCounterHiRes();
    const int searchWindowRadius = numBeats; // Full search range allowed

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

                for (int k = std::max(0, j - searchWindowRadius); k < std::min(numBeats, j + searchWindowRadius); ++k)
                {
                    if (k + 1 >= numBeats) continue;
                    if (cost[i - 1][k] == std::numeric_limits<float>::infinity()) continue;

                    float dist = 1.0f - similarityMatrix[k + 1][j];
                    float transitionCost = similarityPenalty * dist;

                    // Penalize backward jumps and staying at the same beat (no forward progress)
                    if (j <= k)
                    {
                        transitionCost += backwardJumpPenalty;
                    }

                    float currentCost = cost[i - 1][k] + transitionCost + timeDeviationCost;

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
    DBG("2. Fill DP Table: " << juce::String(juce::Time::getMillisecondCounterHiRes() - stepStartTime, 2) << " ms");

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
             return { {}, std::numeric_limits<float>::infinity(), -1, -1 };
        }
    }
    DBG("3. Backtracking: " << juce::String(juce::Time::getMillisecondCounterHiRes() - stepStartTime, 2) << " ms");
    DBG("Total internal time: " << juce::String(juce::Time::getMillisecondCounterHiRes() - functionStartTime, 2) << " ms");
    DBG("-----------------------------");


    // Return average cost per beat
    int actualStart = resultPath.empty() ? -1 : resultPath[0];
    int actualEnd = resultPath.empty() ? -1 : resultPath[targetNumBeats - 1];
    return { resultPath, finalCost / (float)targetNumBeats, actualStart, actualEnd };
}

AudioRetargeter::RetargetResult AudioRetargeter::retargetSegment(
    const std::vector<std::vector<float>>& similarityMatrix,
    const std::vector<float>& beats,
    int startStep,
    int endStep,
    int startBeat,
    int endBeat,
    float secondsPerBeat,
    float similarityPenalty,
    float backwardJumpPenalty,
    float timeContinuityPenalty,
    int startBeatTolerance,
    int endBeatTolerance)
{
    int numSteps = endStep - startStep;
    if (numSteps <= 0)
    {
        return { {}, 0.0f, startBeat, endBeat };
    }

    int numBeats = static_cast<int>(similarityMatrix.size());
    if (numBeats == 0 || startBeat < 0 || endBeat < 0 || startBeat >= numBeats || endBeat >= numBeats)
    {
        return { {}, std::numeric_limits<float>::infinity(), -1, -1 };
    }

    DBG(" ");
    DBG("--- Inside retargetSegment ---");
    DBG("Segment: steps " << startStep << "-" << endStep << ", beats " << startBeat << "->" << endBeat 
        << " (start tolerance: ±" << startBeatTolerance << ", end tolerance: ±" << endBeatTolerance << " beats)");
    auto functionStartTime = juce::Time::getMillisecondCounterHiRes();

    // DP table sized for this segment
    std::vector<std::vector<float>> cost(numSteps, std::vector<float>(numBeats, std::numeric_limits<float>::infinity()));
    std::vector<std::vector<int>> path(numSteps, std::vector<int>(numBeats, -1));

    // Initialization: start beat handling with tolerance
    int actualStartBeat = startBeat;
    
    if (startBeatTolerance > 0)
    {
        // Allow starting within tolerance range of startBeat
        int startBeatMin = std::max(0, startBeat - startBeatTolerance);
        int startBeatMax = std::min(numBeats - 1, startBeat + startBeatTolerance);
        
        for (int j = startBeatMin; j <= startBeatMax; ++j)
        {
            // Small penalty proportional to distance from ideal anchor
            float distancePenalty = std::abs(j - startBeat) * 0.01f;
            cost[0][j] = distancePenalty;
        }
    }
    else
    {
        // No tolerance - must start at exact beat (locked from previous segment)
        cost[0][startBeat] = 0.0f;
    }

    const int searchWindowRadius = numBeats;

    // Fill DP table
    auto stepStartTime = juce::Time::getMillisecondCounterHiRes();
    
    for (int i = 1; i < numSteps; ++i)
    {
        // Calculate ideal beat for linear interpolation between anchors
        float progress = (float)i / (float)(numSteps - 1);
        float idealBeatIndex = startBeat + progress * (endBeat - startBeat);
        
        // For backward segments, disable time continuity to avoid encouraging gradual zigzag
        bool isBackwardSegment = (endBeat < startBeat);

        auto process_j_range = [&](int start_j, int end_j)
        {
            for (int j = start_j; j < end_j; ++j)
            {
                float minCost = std::numeric_limits<float>::infinity();
                int bestPrevBeat = -1;

                // Time deviation from ideal linear path
                // Disable for backward segments to avoid encouraging gradual zigzag movement
                float timeDeviationCost = isBackwardSegment ? 0.0f : 
                    timeContinuityPenalty * std::abs((float)j - idealBeatIndex) / (float)numBeats;

                for (int k = std::max(0, j - searchWindowRadius); k < std::min(numBeats, j + searchWindowRadius); ++k)
                {
                    if (k + 1 >= numBeats) continue;
                    if (cost[i - 1][k] == std::numeric_limits<float>::infinity()) continue;

                    float dist = 1.0f - similarityMatrix[k + 1][j];
                    float transitionCost = similarityPenalty * dist;

                    // Penalize backward jumps and staying at the same beat (no forward progress)
                    if (j <= k)
                    {
                        transitionCost += backwardJumpPenalty;
                    }

                    float currentCost = cost[i - 1][k] + transitionCost + timeDeviationCost;

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
    DBG("Segment DP Fill: " << juce::String(juce::Time::getMillisecondCounterHiRes() - stepStartTime, 2) << " ms");

    // Backtrack from best beat (with or without tolerance at end)
    std::vector<int> resultPath(numSteps);
    
    float bestCost = std::numeric_limits<float>::infinity();
    int bestEnd = endBeat;
    int actualEndBeat = endBeat;
    
    if (endBeatTolerance > 0)
    {
        // Find the best endpoint within tolerance range
        int endBeatMin = std::max(0, endBeat - endBeatTolerance);
        int endBeatMax = std::min(numBeats - 1, endBeat + endBeatTolerance);
        
        for (int j = endBeatMin; j <= endBeatMax; ++j)
        {
            // Add small penalty for distance from ideal anchor
            float distancePenalty = std::abs(j - endBeat) * 0.01f;
            float totalCost = cost[numSteps - 1][j] + distancePenalty;
            
            if (totalCost < bestCost)
            {
                bestCost = totalCost;
                bestEnd = j;
            }
        }
        
        // If no valid path found in tolerance range, search entire space
        if (bestCost == std::numeric_limits<float>::infinity())
        {
            DBG("Warning: No valid path found within tolerance, searching entire space");
            for (int j = 0; j < numBeats; ++j)
            {
                if (cost[numSteps - 1][j] < bestCost)
                {
                    bestCost = cost[numSteps - 1][j];
                    bestEnd = j;
                }
            }
        }
    }
    else
    {
        // No tolerance - must end at exact beat
        bestCost = cost[numSteps - 1][endBeat];
        bestEnd = endBeat;
        
        // Fallback if exact beat has no valid path
        if (bestCost == std::numeric_limits<float>::infinity())
        {
            DBG("Warning: No valid path to exact endBeat, searching entire space");
            for (int j = 0; j < numBeats; ++j)
            {
                if (cost[numSteps - 1][j] < bestCost)
                {
                    bestCost = cost[numSteps - 1][j];
                    bestEnd = j;
                }
            }
        }
    }
    
    float finalCost = bestCost;
    actualEndBeat = bestEnd;
    resultPath[numSteps - 1] = bestEnd;

    for (int i = numSteps - 1; i > 0; --i)
    {
        resultPath[i - 1] = path[i][resultPath[i]];
    }
    
    // Update actual start beat from the path
    actualStartBeat = resultPath[0];

    DBG("Actual beats used: start=" << actualStartBeat << ", end=" << actualEndBeat);
    DBG("Total segment retarget time: " << juce::String(juce::Time::getMillisecondCounterHiRes() - functionStartTime, 2) << " ms");
    DBG("-----------------------------");

    return { resultPath, finalCost / (float)numSteps, actualStartBeat, actualEndBeat };
}
