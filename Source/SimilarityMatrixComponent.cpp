#include "SimilarityMatrixComponent.h"

SimilarityMatrixComponent::SimilarityMatrixComponent()
{
    // Set a default size or wait for resized()
}

void SimilarityMatrixComponent::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colours::black);

    if (matrix.empty())
    {
        g.setColour(juce::Colours::white);
        g.drawText("No data", getLocalBounds(), juce::Justification::centred, false);
        return;
    }

    g.drawImage(matrixImage, getLocalBounds().toFloat());
    g.drawImage(pathImage, getLocalBounds().toFloat());


    int numCells = matrix.size();
    if (numCells == 0) return;
    
    float cellWidth = (float)getWidth() / numCells;
    float cellHeight = (float)getHeight() / numCells;

    // Draw hover label
    if (isMouseOver)
    {
        int xBeat = static_cast<int>(mousePosition.x / cellWidth);
        int yBeat = static_cast<int>(mousePosition.y / cellHeight);

        if (xBeat >= 0 && xBeat < numCells && yBeat >= 0 && yBeat < numCells)
        {
            juce::String labelText = "Beat X: " + juce::String(xBeat) + ", Beat Y: " + juce::String(yBeat);
            
            float textWidth = g.getCurrentFont().getStringWidthFloat(labelText);

            auto x = mousePosition.x + 12;
            auto y = mousePosition.y;

            if (x + textWidth > getWidth())
                x = mousePosition.x - textWidth - 12;

            g.setColour(juce::Colours::black.withAlpha(0.7f));
            g.fillRoundedRectangle(x, y - 20, textWidth + 8, 18, 5.0f);

            g.setColour(juce::Colours::white);
            g.drawText(labelText, (int) x + 4, (int) y - 18, (int) textWidth, 14, juce::Justification::left, true);
        }
    }

    // Draw playhead
    if (beats.empty() || duration <= 0.0) return;

    if (inRetargetedMode && !path.empty())
    {
        float xProportion = (float)path.size() / (float)matrix.size();
        int pathWidth = (int)(getWidth() * xProportion);

        // In retargeted mode, X-axis is the new timeline (path index)
        int currentNewBeat = static_cast<int>(playheadPosition * (path.size() - 1));
        float xPosition = (currentNewBeat / (float)(path.size() - 1)) * pathWidth;

        g.setColour(juce::Colours::red.withAlpha(0.7f));
        g.drawVerticalLine(juce::roundToInt(xPosition), 0.0f, (float)getHeight());
    }
    else
    {
        double currentTime = playheadPosition * duration;

        int currentBeatIndex = -1;
        if (!beats.empty())
        {
            auto it = std::upper_bound(beats.begin(), beats.end(), currentTime);
            if (it != beats.begin())
            {
                currentBeatIndex = std::distance(beats.begin(), it) - 1;
            }
        }

        if (currentBeatIndex != -1)
        {
            double startTimeOfBeat = beats[currentBeatIndex];
            double endTimeOfBeat = (currentBeatIndex + 1 < (int)beats.size()) ? beats[currentBeatIndex + 1] : duration;
            double beatDuration = endTimeOfBeat - startTimeOfBeat;
            
            double fractionalPosition = 0.0;
            if (beatDuration > 0)
            {
                fractionalPosition = (currentTime - startTimeOfBeat) / beatDuration;
            }

            float xPosition = (currentBeatIndex + fractionalPosition) * cellWidth;

            g.setColour(juce::Colours::red.withAlpha(0.7f));
            g.drawVerticalLine(juce::roundToInt(xPosition), 0.0f, (float)getHeight());
        }
    }

    // Draw user constraints
    if (!constraints.empty() && duration > 0)
    {
        g.setColour(juce::Colours::green);
        for (const auto& c : constraints)
        {
            // c.first is Source Time (Y), c.second is Target Time (X)
            // Both axes mapped to Source Duration based on user request
            float x = (c.second / duration) * getWidth();
            float y = (c.first / duration) * getHeight();
            
            g.fillEllipse(x - 4.0f, y - 4.0f, 8.0f, 8.0f);
            g.drawEllipse(x - 6.0f, y - 6.0f, 12.0f, 12.0f, 2.0f);
        }
    }
}

void SimilarityMatrixComponent::mouseMove(const juce::MouseEvent& event)
{
    mousePosition = event.getPosition();
    isMouseOver = true;
    repaint();
}

void SimilarityMatrixComponent::mouseDown(const juce::MouseEvent& event)
{
    if (duration <= 0.0) return;

    float xProp = (float)event.x / (float)getWidth();
    float yProp = (float)event.y / (float)getHeight();

    // Map: X -> Target Time, Y -> Source Time
    // Both mapped relative to Source Duration (matrix axes)
    float clickedTargetTime = xProp * duration;
    float clickedSourceTime = yProp * duration;

    // Right click clears constraints
    if (event.mods.isRightButtonDown())
    {
        constraints.clear();
    }
    else
    {
        constraints.push_back({clickedSourceTime, clickedTargetTime});
    }

    if (onConstraintsChanged)
        onConstraintsChanged(constraints);
    
    repaint();
}

void SimilarityMatrixComponent::mouseExit(const juce::MouseEvent& event)
{
    isMouseOver = false;
    repaint();
}

void SimilarityMatrixComponent::updateMatrix(const std::vector<std::vector<float>>& newMatrix)
{
    matrix = newMatrix;
    updateMatrixImage();
    updatePathImage(); // Path may depend on matrix for coloring, so update it too
    repaint();
}

void SimilarityMatrixComponent::updateBeatInfo(const std::vector<double>& beatTimestamps, double totalDuration)
{
    beats = beatTimestamps;
    duration = totalDuration;
}

void SimilarityMatrixComponent::updatePath(const std::vector<int>& newPath)
{
    path = newPath;
    updatePathImage();
    repaint();
}

void SimilarityMatrixComponent::setPlayheadPosition(float newPosition)
{
    playheadPosition = juce::jlimit(0.0f, 1.0f, newPosition);
    repaint();
}

void SimilarityMatrixComponent::setRetargetedMode(bool isRetargeted)
{
    inRetargetedMode = isRetargeted;
    updatePathImage();
    repaint();
}

void SimilarityMatrixComponent::setTargetDuration(double targetDur)
{
    targetDuration = targetDur;
    // Don't necessarily clear constraints, maybe scale them? Or leave as is.
    // Ideally user sets constraints relative to current target duration.
    repaint();
}

void SimilarityMatrixComponent::clearConstraints()
{
    constraints.clear();
    if (onConstraintsChanged)
        onConstraintsChanged(constraints);
    repaint();
}

void SimilarityMatrixComponent::resized()
{
    updateMatrixImage();
    updatePathImage();
}

void SimilarityMatrixComponent::updateMatrixImage()
{
    if (matrix.empty() || getWidth() <= 0 || getHeight() <= 0)
    {
        matrixImage = juce::Image(); // Clear image
        return;
    }

    const int imageResolution = 256;
    matrixImage = juce::Image(juce::Image::ARGB, imageResolution, imageResolution, true);

    int numCells = matrix.size();
    if (numCells == 0) return;

    // --- Normalize the matrix for full contrast ---
    float minVal = 1.0f, maxVal = -1.0f;
    for (const auto& row : matrix)
    {
        for (float val : row)
        {
            minVal = std::min(minVal, val);
            maxVal = std::max(maxVal, val);
        }
    }

    juce::ColourGradient gradient;
    gradient.addColour(0.0, juce::Colours::darkblue);
    gradient.addColour(0.5, juce::Colours::grey);
    gradient.addColour(1.0, juce::Colours::yellow);

    juce::Image::BitmapData bitmapData(matrixImage, juce::Image::BitmapData::writeOnly);

    for (int y = 0; y < imageResolution; ++y)
    {
        for (int x = 0; x < imageResolution; ++x)
        {
            // Map pixel to matrix coordinates using nearest-neighbor
            int matrixX = static_cast<int>((float)x / (float)imageResolution * numCells);
            int matrixY = static_cast<int>((float)y / (float)imageResolution * numCells);

            // Clamp to be safe
            matrixX = juce::jmin(matrixX, numCells - 1);
            matrixY = juce::jmin(matrixY, numCells - 1);

            float value = matrix[matrixX][matrixY];
            double proportion = 0.0;
            if (maxVal > minVal)
            {
                proportion = (value - minVal) / (maxVal - minVal);
            }
            
            juce::Colour colour = gradient.getColourAtPosition(proportion);
            bitmapData.setPixelColour(x, y, colour);
        }
    }
}

void SimilarityMatrixComponent::updatePathImage()
{
    if (getWidth() <= 0 || getHeight() <= 0)
    {
        pathImage = juce::Image();
        return;
    }

    pathImage = juce::Image(juce::Image::ARGB, getWidth(), getHeight(), true);
    juce::Graphics g(pathImage);

    int numCells = matrix.size();
    if (path.empty() || numCells == 0)
        return;

    float cellWidth = (float)getWidth() / numCells;
    float cellHeight = (float)getHeight() / numCells;
    
    if (inRetargetedMode)
    {
        float xProportion = (float)path.size() / (float)matrix.size();
        int pathWidth = (int)(getWidth() * xProportion);

        float newCellWidth = (float)pathWidth / path.size();
        
        for (size_t i = 0; i < path.size() - 1; ++i)
        {
            float x1 = (i + 0.5f) * newCellWidth;
            float y1 = (path[i] + 0.5f) * cellHeight;
            float x2 = ((i + 1) + 0.5f) * newCellWidth;
            float y2 = (path[i+1] + 0.5f) * cellHeight;

            int u = path[i];
            int v = path[i+1];

            // Check if this is a natural progression (diagonal) or a jump
            if (v == u + 1)
            {
                    // Natural flow -> Cyan (standard path color)
                    g.setColour(juce::Colours::cyan.withAlpha(0.8f));
                    g.drawLine(x1, y1, x2, y2, 2.0f);
            }
            else
            {
                // It's a jump! Color code based on similarity of the transition
                float sim = 0.0f;

                if (u + 1 < numCells && v < numCells) 
                {
                    sim = matrix[u+1][v];
                } 
                else if (u + 1 >= numCells) 
                {
                    sim = 0.0f; 
                }
                
                float normalizedSim = juce::jlimit(0.0f, 1.0f, sim);
                
                normalizedSim = std::pow(normalizedSim, 4.0f);
                
                float hue = normalizedSim * 0.33f; // 0.0(Red) to 0.33(Green)
                
                g.setColour(juce::Colour::fromHSV(hue, 1.0f, 1.0f, 1.0f));
                g.drawLine(x1, y1, x2, y2, 3.0f); // Make jumps slightly thicker
            }
        }
    }
    else
    {
        g.setColour(juce::Colours::cyan.withAlpha(0.8f));
        juce::Path pathVis;
        
        bool first = true;

        for (size_t i = 0; i < path.size() - 1; ++i)
        {
            int u = path[i];
            int v = path[i+1];

            if (u >= 0 && u < numCells && v >= 0 && v < numCells)
            {
                float x = (u + 0.5f) * cellWidth;
                float y = (v + 0.5f) * cellHeight;

                if (first)
                {
                    pathVis.startNewSubPath(x, y);
                    first = false;
                }
                else
                {
                    pathVis.lineTo(x, y);
                }
                
                // Draw a small dot at each transition point
                g.fillEllipse(x - 2.0f, y - 2.0f, 4.0f, 4.0f);
            }
        }
        g.strokePath(pathVis, juce::PathStrokeType(1.5f));
    }
}
