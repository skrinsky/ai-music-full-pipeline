#pragma once
#include <juce_audio_processors/juce_audio_processors.h>
#include "PipelineClient.h"

class AIMusicProcessor : public juce::AudioProcessor,
                         private juce::Timer
{
public:
    AIMusicProcessor();
    ~AIMusicProcessor() override;

    // AudioProcessor boilerplate
    void prepareToPlay (double, int) override {}
    void releaseResources() override {}
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }
    const juce::String getName() const override { return "AI Music"; }
    bool acceptsMidi() const override  { return false; }
    bool producesMidi() const override { return true; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}
    void getStateInformation (juce::MemoryBlock&) override {}
    void setStateInformation (const void*, int) override {}
    bool isBusesLayoutSupported (const BusesLayout&) const override { return true; }

    // Pipeline control — called from editor
    void startProcess (const juce::String& audioFolder);
    void startTrain();
    void startGenerate();

    // State the editor reads
    PipelineStatus lastStatus;
    juce::String   pendingJobId;
    juce::String   ckptPath;
    juce::String   audioFolder;

    // Generation parameters (owned by editor, read by processor on Generate)
    float  temperature    { 0.75f };
    float  topP           { 0.95f };
    float  tempoBpm       { 75.0f };
    int    forceGridStep  { 6 };
    int    maxTokens      { 512 };
    bool   syncTempo      { false };

    double getHostBpm() const;

    // MIDI to send out on next processBlock call (filled from background thread)
    juce::MidiBuffer pendingMidi;
    juce::CriticalSection midiLock;

    std::function<void()> onStatusChanged;

private:
    PipelineClient client;
    std::unique_ptr<juce::ChildProcess> serverProcess;
    juce::ApplicationProperties appProperties;

    juce::PropertiesFile* getPrefs();
    void launchServer();
    void tryLaunchServerFromRepoRoot (const juce::File& repoRoot);
    juce::File findRepoRoot (const juce::File& startDir);
    void timerCallback() override;
    void pollForMidi();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AIMusicProcessor)
};
