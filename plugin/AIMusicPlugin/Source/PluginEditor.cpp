#include "PluginEditor.h"

static const juce::Colour kBg  { 0xff1e1e2e };
static const juce::Colour kFg  { 0xffcdd6f4 };
static const juce::Colour kAcc { 0xff89b4fa };

AIMusicEditor::AIMusicEditor (AIMusicProcessor& p)
    : AudioProcessorEditor (&p), proc (p)
{
    setSize (480, 360);

    // ── checkpoint path ──────────────────────────────────────────────────
    lblCkpt.setText (proc.ckptPath.isNotEmpty() ? proc.ckptPath : "No checkpoint selected",
                     juce::dontSendNotification);
    lblCkpt.setColour (juce::Label::textColourId, kFg);
    lblCkpt.setJustificationType (juce::Justification::centredLeft);
    addAndMakeVisible (lblCkpt);

    btnBrowseCkpt.onClick = [this] { browseCheckpoint(); };
    addAndMakeVisible (btnBrowseCkpt);

    // ── knobs ────────────────────────────────────────────────────────────
    makeKnob (sldTemperature, 0.1, 2.0, proc.temperature, 0.01);
    makeKnob (sldTopP,        0.1, 1.0, proc.topP,        0.01);
    makeKnob (sldGridStep,    1,   24,  proc.forceGridStep, 1);
    makeKnob (sldMaxTokens,   64,  2048, proc.maxTokens,   64);
    makeKnob (sldTempo,       40,  240, proc.tempoBpm,     0.5);

    sldTemperature.onValueChange = [this] { proc.temperature   = (float) sldTemperature.getValue(); };
    sldTopP       .onValueChange = [this] { proc.topP          = (float) sldTopP.getValue(); };
    sldGridStep   .onValueChange = [this] { proc.forceGridStep = (int)   sldGridStep.getValue(); };
    sldMaxTokens  .onValueChange = [this] { proc.maxTokens     = (int)   sldMaxTokens.getValue(); };
    sldTempo      .onValueChange = [this] { proc.tempoBpm      = (float) sldTempo.getValue(); };

    auto makeLabel = [&] (juce::Label& l, const juce::String& text)
    {
        l.setText (text, juce::dontSendNotification);
        l.setJustificationType (juce::Justification::centred);
        l.setColour (juce::Label::textColourId, kFg);
        addAndMakeVisible (l);
    };
    makeLabel (lblTemperature, "Temp");
    makeLabel (lblTopP,        "Top-P");
    makeLabel (lblGridStep,    "Grid");
    makeLabel (lblMaxTokens,   "Tokens");
    makeLabel (lblTempo,       "Tempo");

    btnSyncTempo.setToggleState (proc.syncTempo, juce::dontSendNotification);
    btnSyncTempo.setColour (juce::ToggleButton::textColourId, kFg);
    btnSyncTempo.onStateChange = [this]
    {
        proc.syncTempo = btnSyncTempo.getToggleState();
        sldTempo.setEnabled (! proc.syncTempo);
    };
    sldTempo.setEnabled (! proc.syncTempo);
    addAndMakeVisible (btnSyncTempo);

    // ── audio folder + action buttons ────────────────────────────────────
    lblFolder.setText ("No folder selected", juce::dontSendNotification);
    lblFolder.setColour (juce::Label::textColourId, kFg);
    lblFolder.setJustificationType (juce::Justification::centredLeft);
    addAndMakeVisible (lblFolder);

    btnProcess.onClick  = [this] { chooseFolder(); };
    btnTrain.onClick    = [this] { proc.startTrain(); };
    btnGenerate.onClick = [this] { proc.startGenerate(); };
    addAndMakeVisible (btnProcess);
    addAndMakeVisible (btnTrain);
    addAndMakeVisible (btnGenerate);

    // ── status ───────────────────────────────────────────────────────────
    lblStatus.setColour (juce::Label::textColourId, kFg);
    lblStatus.setJustificationType (juce::Justification::centredLeft);
    addAndMakeVisible (lblStatus);

    lblMessage.setColour (juce::Label::textColourId, kAcc);
    lblMessage.setJustificationType (juce::Justification::centredLeft);
    addAndMakeVisible (lblMessage);

    btnShowMidi.setVisible (false);
    btnShowMidi.onClick = [this]
    {
        if (lastMidiPath.isNotEmpty())
            juce::File (lastMidiPath).revealToUser();
    };
    addAndMakeVisible (btnShowMidi);

    startTimer (1500);
}

AIMusicEditor::~AIMusicEditor() { stopTimer(); }

void AIMusicEditor::makeKnob (juce::Slider& s, double mn, double mx, double def, double step)
{
    s.setSliderStyle (juce::Slider::RotaryVerticalDrag);
    s.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 60, 16);
    s.setRange (mn, mx, step);
    s.setValue (def, juce::dontSendNotification);
    s.setColour (juce::Slider::rotarySliderFillColourId,  kAcc);
    s.setColour (juce::Slider::rotarySliderOutlineColourId, kFg.withAlpha (0.3f));
    s.setColour (juce::Slider::textBoxTextColourId, kFg);
    s.setColour (juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    addAndMakeVisible (s);
}

void AIMusicEditor::paint (juce::Graphics& g)
{
    g.fillAll (kBg);
    g.setColour (kFg);
    g.setFont (juce::Font (16.0f, juce::Font::bold));
    g.drawText ("AI Music Pipeline", getLocalBounds().removeFromTop (36),
                juce::Justification::centred);
}

void AIMusicEditor::resized()
{
    auto area = getLocalBounds().reduced (12);
    area.removeFromTop (36); // title

    // Checkpoint row
    auto ckptRow = area.removeFromTop (24);
    btnBrowseCkpt.setBounds (ckptRow.removeFromRight (70));
    ckptRow.removeFromRight (4);
    lblCkpt.setBounds (ckptRow);
    area.removeFromTop (8);

    // Knobs row
    auto knobArea = area.removeFromTop (90);
    int knobW = knobArea.getWidth() / 5;
    using KnobPair = std::pair<juce::Slider*, juce::Label*>;
    for (auto pair : { KnobPair {&sldTemperature, &lblTemperature},
                       KnobPair {&sldTopP,         &lblTopP},
                       KnobPair {&sldGridStep,     &lblGridStep},
                       KnobPair {&sldMaxTokens,    &lblMaxTokens},
                       KnobPair {&sldTempo,        &lblTempo} })
    {
        auto col = knobArea.removeFromLeft (knobW);
        pair.second->setBounds (col.removeFromBottom (18));
        pair.first ->setBounds (col);
    }

    // Sync tempo toggle (sits below tempo label)
    area.removeFromTop (2);
    auto syncRow = area.removeFromTop (20);
    syncRow.removeFromLeft (knobW * 4); // align under Tempo knob
    btnSyncTempo.setBounds (syncRow.removeFromLeft (knobW));
    area.removeFromTop (8);

    // Audio folder label
    lblFolder.setBounds (area.removeFromTop (22));
    area.removeFromTop (6);

    // Action buttons
    auto btnRow = area.removeFromTop (34);
    btnProcess .setBounds (btnRow.removeFromLeft (140));
    btnRow.removeFromLeft (6);
    btnTrain   .setBounds (btnRow.removeFromLeft (90));
    btnRow.removeFromLeft (6);
    btnGenerate.setBounds (btnRow.removeFromLeft (110));
    area.removeFromTop (10);

    // Status
    lblStatus .setBounds (area.removeFromTop (22));
    area.removeFromTop (4);
    auto msgRow = area.removeFromTop (22);
    btnShowMidi.setBounds (msgRow.removeFromRight (90));
    msgRow.removeFromRight (6);
    lblMessage.setBounds (msgRow);
}

void AIMusicEditor::timerCallback() { updateStatusLabel(); }

void AIMusicEditor::updateStatusLabel()
{
    auto& s = proc.lastStatus;
    lblStatus.setText ("Status: " + s.stage + (s.error.isNotEmpty() ? "  \xe2\x80\x94 " + s.error : ""),
                       juce::dontSendNotification);

    if (s.stage == "training" && s.epoch >= 0)
    {
        juce::String ep = "Epoch " + juce::String (s.epoch);
        if (s.totalEpochs > 0) ep += " / " + juce::String (s.totalEpochs);
        if (s.valLoss >= 0)    ep += "   val loss: " + juce::String (s.valLoss, 4);
        lblMessage.setText (ep, juce::dontSendNotification);
    }
    else
    {
        lblMessage.setText (s.message, juce::dontSendNotification);
    }

    // Parse midi_id=JOBID from status message to show reveal button
    if (s.stage == "done" && s.message.startsWith ("midi_id="))
    {
        auto jobId = s.message.fromFirstOccurrenceOf ("midi_id=", false, false);
        if (jobId.isNotEmpty())
        {
            // Reconstruct path matching server's out_dir layout
            auto repoRoot = juce::String (
#ifdef AI_REPO_ROOT
                AI_REPO_ROOT
#else
                ""
#endif
            );
            if (repoRoot.isNotEmpty())
            {
                lastMidiPath = repoRoot + "/runs/generated/plugin/" + jobId + "/generated.mid";
                btnShowMidi.setVisible (juce::File (lastMidiPath).existsAsFile());
            }
        }
    }
    else if (s.stage != "done")
    {
        btnShowMidi.setVisible (false);
    }

    bool serverReady = (s.stage == "idle" || s.stage == "done" || s.stage == "error");
    btnProcess .setEnabled (true);
    btnTrain   .setEnabled (serverReady);
    btnGenerate.setEnabled (serverReady && proc.ckptPath.isNotEmpty());
}

void AIMusicEditor::chooseFolder()
{
    auto chooser = std::make_shared<juce::FileChooser> (
        "Select audio folder", juce::File::getSpecialLocation (juce::File::userMusicDirectory));

    chooser->launchAsync (juce::FileBrowserComponent::openMode |
                          juce::FileBrowserComponent::canSelectDirectories,
        [this, chooser] (const juce::FileChooser& fc)
        {
            auto folder = fc.getResult();
            if (folder.isDirectory())
            {
                lblFolder.setText (folder.getFullPathName(), juce::dontSendNotification);
                proc.startProcess (folder.getFullPathName());
            }
        });
}

void AIMusicEditor::browseCheckpoint()
{
    auto chooser = std::make_shared<juce::FileChooser> (
        "Select checkpoint (.pt)", juce::File::getSpecialLocation (juce::File::userHomeDirectory),
        "*.pt");

    chooser->launchAsync (juce::FileBrowserComponent::openMode |
                          juce::FileBrowserComponent::canSelectFiles,
        [this, chooser] (const juce::FileChooser& fc)
        {
            auto f = fc.getResult();
            if (f.existsAsFile())
            {
                proc.ckptPath = f.getFullPathName();
                lblCkpt.setText (f.getFullPathName(), juce::dontSendNotification);
            }
        });
}
