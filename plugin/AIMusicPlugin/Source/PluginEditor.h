#pragma once
#include <juce_audio_processors/juce_audio_processors.h>
#include "PluginProcessor.h"

class AIMusicEditor : public juce::AudioProcessorEditor,
                      private juce::Timer
{
public:
    explicit AIMusicEditor (AIMusicProcessor&);
    ~AIMusicEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;

private:
    AIMusicProcessor& proc;

    // Checkpoint path
    juce::Label     lblCkpt;
    juce::TextButton btnBrowseCkpt { "Browse" };

    // Generation knobs
    juce::Slider sldTemperature, sldTopP, sldGridStep, sldMaxTokens, sldTempo;
    juce::Label  lblTemperature, lblTopP, lblGridStep, lblMaxTokens, lblTempo;
    juce::ToggleButton btnSyncTempo { "Sync" };

    // Audio folder + actions
    juce::Label     lblFolder;
    juce::TextButton btnProcess  { "Process Audio" };
    juce::TextButton btnTrain    { "Train" };
    juce::TextButton btnGenerate { "Generate" };

    // Status
    juce::Label lblStatus;
    juce::Label lblMessage;
    juce::TextButton btnShowMidi { "Show MIDI" };
    juce::String lastMidiPath;

    void timerCallback() override;
    void updateStatusLabel();
    void chooseFolder();
    void browseCheckpoint();
    void makeKnob (juce::Slider&, double min, double max, double def, double step = 0.0);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AIMusicEditor)
};
