#pragma once
#include <juce_audio_processors/juce_audio_processors.h>
#include "PluginProcessor.h"

class AIMusicEditor : public juce::AudioProcessorEditor,
                      private juce::Timer,
                      public juce::DragAndDropContainer
{
public:
    explicit AIMusicEditor (AIMusicProcessor&);
    ~AIMusicEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;

private:
    AIMusicProcessor& proc;

    // ── Tab bar ──────────────────────────────────────────────────────────────
    juce::TextButton tabProcess  { "Process & Train" };
    juce::TextButton tabGenerate { "Generate" };
    int currentTab { 0 };

    // ── Tab 1: Process & Train ────────────────────────────────────────────────
    juce::Label        lblFolder;
    juce::TextButton   btnBrowseFolder { "Select Audio Path" };
    juce::Label        lblInstruments;
    juce::ToggleButton chkLeadVox { "Lead Vox" };
    juce::ToggleButton chkHarmVox { "Harm Vox" };
    juce::ToggleButton chkGuitar  { "Guitar" };
    juce::ToggleButton chkBass    { "Bass" };
    juce::ToggleButton chkDrums   { "Drums" };
    juce::ToggleButton chkOther   { "Other" };
    juce::Slider       sldSeqLen;
    juce::Label        lblSeqLen;
    juce::TextButton   btnRunProcess { "Process Audio" };
    juce::TextButton   btnTrain      { "Train" };

    // ── Tab 2: Generate ───────────────────────────────────────────────────────
    juce::Label        lblCkpt;
    juce::TextButton   btnBrowseCkpt  { "Select Model" };
    juce::Slider       sldTemperature, sldTopP, sldMaxTokens, sldTempo;
    juce::Label        lblTemperature, lblTopP, lblMaxTokens, lblTempo;
    juce::ToggleButton btnSyncTempo   { "Sync" };
    juce::ComboBox     cmbSubdivision;
    juce::ToggleButton btnTriplets    { "Include Triplets" };
    juce::Label        lblSubdivision;
    juce::ToggleButton btnSeedFromData { "Seed from training data" };
    juce::TextButton   btnGenerate    { "Generate" };

    // ── Shared ────────────────────────────────────────────────────────────────
    juce::TextButton btnCancel { "Cancel" };
    juce::Label      lblStatus;
    juce::Label      lblMessage;
    juce::Label      lblTokenWarning;
    juce::TextButton btnShowMidi { "Show MIDI" };
    juce::String     lastMidiPath;

    std::unique_ptr<juce::Component> mirrorAnim;
    std::unique_ptr<juce::LookAndFeel> smallToggleLAF;
    juce::String prevStage;
    juce::String localErrorMessage;  // client-side errors that survive the server status poll

    void timerCallback() override;
    void mouseDrag (const juce::MouseEvent&) override;
    void updateStatusLabel();
    void updateTokenWarning();
    void updateTabVisibility();
    juce::String buildTracksString() const;
    void browseFolder (bool startAfterSelect = false);
    void browseCheckpoint();
    void makeKnob (juce::Slider&, double min, double max, double def, double step = 0.0);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AIMusicEditor)
};
