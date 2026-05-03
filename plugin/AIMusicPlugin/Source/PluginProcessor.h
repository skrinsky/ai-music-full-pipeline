#pragma once
#include <juce_audio_processors/juce_audio_processors.h>
#include "PipelineClient.h"
#include <vector>
#include <utility>

class AIMusicProcessor : public juce::AudioProcessor,
                         private juce::Timer
{
public:
    AIMusicProcessor();
    ~AIMusicProcessor() override;

    // AudioProcessor boilerplate
    void prepareToPlay (double sampleRate, int blockSize) override;
    void releaseResources() override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }
    const juce::String getName() const override { return "Mirror Mirror"; }
    bool acceptsMidi() const override  { return false; }
    bool producesMidi() const override { return true; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}
    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;
    bool isBusesLayoutSupported (const BusesLayout&) const override { return true; }

    // Pipeline control — called from editor
    void startProcess (const juce::String& audioFolder);
    void startTrain (const juce::String& eventsDir = {});
    void startGenerate();
    void cancelJob();

    // State the editor reads
    PipelineStatus lastStatus;
    juce::String   pendingJobId;
    juce::String   ckptPath;
    juce::String   audioFolder;
    juce::String   selectedTracks;  // comma-separated demucs stems, empty = all

    // Generation parameters (owned by editor, read by processor on Generate)
    float  temperature    { 0.75f };
    float  topP           { 0.95f };
    float  tempoBpm       { 75.0f };
    int    gridSubdivision { 6 };   // straight step in ticks: 24=1/4, 12=1/8, 6=1/16, 3=1/32
    bool   allowTriplets  { true };
    int    maxTokens      { 512 };
    bool   syncTempo      { true };
    bool   seedFromData   { true };
    bool   quantize       { true };

    double getHostBpm() const { return cachedBpm.load(); }
    int    loadCheckpointInfo();
    bool   isTrainingDataReady();
    juce::String getPref (const juce::String& key, const juce::String& fallback = {});
    void         setPref (const juce::String& key, const juce::String& value);
    juce::String fetchLatestEvents() { return client.fetchLatestEvents(); }

    // MIDI to send out on next processBlock call (filled from background thread)
    juce::MidiBuffer pendingMidi;
    juce::CriticalSection midiLock;

    std::atomic<double> cachedBpm { 120.0 };
    int trainingCtxLen { 0 };

    // Audio preview of generated MIDI
    void startPreview (const juce::String& midiFilePath);
    void stopPreview();
    bool isPreviewPlaying() const { return previewActive.load(); }

    std::function<void()> onStatusChanged;
    std::function<void()> onStateLoaded;   // editor refreshes UI after DAW session restore
    std::function<void(bool)> onPreviewStateChanged; // called on message thread: true=started, false=stopped

private:
    PipelineClient client;
    juce::int64 lastServerLaunchMs { 0 };   // cooldown — don't re-launch within 15 s
    juce::ApplicationProperties appProperties;

#if JUCE_MAC
    // macOS: Apple DLS MusicDevice (built-in GM synth — no dependencies)
    void*                        previewDLSSynth     { nullptr }; // AudioUnit opaque ptr
    std::atomic<bool>            previewResetPending { false };
    juce::CriticalSection        previewLock;
    std::vector<std::pair<double, juce::MidiMessage>> previewEvents;
    juce::int64                  previewSamplePos    { 0 };
    double                       previewDuration     { 0.0 };
    int                          previewNextEvent    { 0 };
#else
    // Non-macOS: server renders WAV, plugin plays it back via AudioTransportSource
    juce::AudioFormatManager     previewFormatManager;
    juce::AudioTransportSource   previewTransport;
    std::unique_ptr<juce::AudioFormatReaderSource> previewReaderSource;
    juce::File                   previewTempFile;
    std::shared_ptr<std::atomic<bool>> previewAlive { std::make_shared<std::atomic<bool>>(true) };
    std::atomic<int>             previewDownloadGen  { 0 };
    struct PreviewDownloadThread;
    std::unique_ptr<PreviewDownloadThread> previewDownloadThread;
    void loadPreviewWav (juce::MemoryBlock&& wavData, int gen);
#endif
    std::atomic<bool>            previewActive      { false };
    std::atomic<bool>            previewStopRequest { false };

    juce::PropertiesFile* getPrefs();
    void launchServer();
    void tryLaunchServerFromRepoRoot (const juce::File& repoRoot);
    juce::File findRepoRoot (const juce::File& startDir);
    void timerCallback() override;
    void pollForMidi();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AIMusicProcessor)
};
