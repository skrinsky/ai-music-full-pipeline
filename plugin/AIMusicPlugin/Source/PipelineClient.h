#pragma once
#include <juce_core/juce_core.h>

struct PipelineStatus
{
    juce::String stage;      // idle | processing | training | generating | done | error
    juce::String message;
    int    epoch      = -1;
    int    totalEpochs= -1;
    double valLoss    = -1.0;
    float  progress   = -1.f; // 0–1 during preprocessing, -1 otherwise
    juce::String error;
};

// Thin HTTP client that talks to plugin/server.py on localhost:7437
class PipelineClient
{
public:
    explicit PipelineClient (int port = 7437);

    bool        isServerReachable();
    PipelineStatus getStatus();

    // Fire-and-forget POST calls — server runs jobs in background threads
    bool postProcess (const juce::String& audioFolder,
                      const juce::String& tracks = {},
                      bool normalizeKey = true);

    bool postTrain   (const juce::String& eventsDir  = "runs/events",
                      const juce::String& ckptPath   = "runs/checkpoints/es_model.pt",
                      const juce::String& device     = "auto",
                      int epochs = 200,
                      int seqLen = 512);

    // Returns job_id on success, empty string on failure
    juce::String postGenerate (const juce::String& ckpt,
                               const juce::String& vocabJson,
                               const juce::String& seedPkl      = {},
                               float temperature                 = 0.75f,
                               float topP                        = 0.95f,
                               float tempoBpm                    = 75.0f,
                               int   gridStraightStep            = 6,
                               int   gridTripletStep             = 0,
                               int   maxTokens                   = 512,
                               bool  useSeed                     = false);

    bool postCancel();

    // Download MIDI for a completed job; returns true + fills midiData on success
    bool fetchMidi (const juce::String& jobId, juce::MemoryBlock& midiData);

    // Download WAV preview (server synthesises on demand); sampleRate and bpm forwarded
    bool fetchPreviewWav (const juce::String& jobId, juce::MemoryBlock& wavData,
                          int sampleRate = 44100, double bpm = 0.0);

    // Returns the SEQ_LEN the checkpoint was trained with, or 0 on failure
    int fetchCheckpointInfo (const juce::String& ckptPath);

    // Returns path to most recently created events folder, or empty string
    juce::String fetchLatestEvents();

private:
    juce::String baseUrl;

    juce::String get  (const juce::String& path);
    juce::String post (const juce::String& path, const juce::String& jsonBody);
};
