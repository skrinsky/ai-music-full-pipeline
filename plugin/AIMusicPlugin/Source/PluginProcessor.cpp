#include "PluginProcessor.h"
#include "PluginEditor.h"

AIMusicProcessor::AIMusicProcessor()
    : AudioProcessor (BusesProperties()
                          .withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
      client (7437)
{
}

AIMusicProcessor::~AIMusicProcessor()
{
    stopTimer();
    if (serverProcess != nullptr)
        serverProcess->kill();
}

juce::File AIMusicProcessor::findRepoRoot (const juce::File& startDir)
{
    auto search = startDir;
    for (int i = 0; i < 10; ++i)
    {
        if (search.getChildFile ("plugin/server.py").existsAsFile())
            return search;
        auto parent = search.getParentDirectory();
        if (parent == search) break; // reached filesystem root
        search = parent;
    }
    return {};
}

void AIMusicProcessor::tryLaunchServerFromRepoRoot (const juce::File& repoRoot)
{
    auto serverScript = repoRoot.getChildFile ("plugin/server.py");
    if (! serverScript.existsAsFile()) return;

    serverProcess = std::make_unique<juce::ChildProcess>();

    auto activate = repoRoot.getChildFile (".venv-ai-music/bin/activate");
    if (activate.existsAsFile())
    {
        // Launch via bash so the venv activates properly — on macOS the Python.app
        // framework doesn't pick up venv site-packages unless invoked through a shell.
        auto q = [] (const juce::String& s) { return "\"" + s + "\""; };
        auto shellCmd = ". " + q (activate.getFullPathName())
                      + " && python " + q (serverScript.getFullPathName())
                      + " --root "   + q (repoRoot.getFullPathName());
        serverProcess->start ({ "/bin/bash", "-c", shellCmd });
    }
    else
    {
        serverProcess->start ({ "python3", serverScript.getFullPathName(),
                                "--root", repoRoot.getFullPathName() });
    }
}

juce::PropertiesFile* AIMusicProcessor::getPrefs()
{
    if (appProperties.getUserSettings() == nullptr)
    {
        juce::PropertiesFile::Options opts;
        opts.applicationName = "AIMusicPlugin";
        opts.filenameSuffix  = ".xml";
        opts.folderName      = "AIMusicPlugin";
        appProperties.setStorageParameters (opts);
    }
    return appProperties.getUserSettings();
}

void AIMusicProcessor::launchServer()
{
    if (client.isServerReachable()) return;

    // 0. Compile-time path — always correct for dev builds
#ifdef AI_REPO_ROOT
    {
        juce::File compiledRoot { juce::String (AI_REPO_ROOT) };
        if (compiledRoot.getChildFile ("plugin/server.py").existsAsFile())
        {
            tryLaunchServerFromRepoRoot (compiledRoot);
            return;
        }
    }
#endif

    // 1. Try repo root saved from a previous session
    if (auto* prefs = getPrefs())
    {
        auto saved = prefs->getValue ("repoRoot");
        if (saved.isNotEmpty())
        {
            auto root = juce::File (saved);
            if (root.getChildFile ("plugin/server.py").existsAsFile())
            {
                tryLaunchServerFromRepoRoot (root);
                return;
            }
        }
    }

    // 2. Fall back to walking up from plugin binary (works for dev/build installs)
    auto pluginDir = juce::File::getSpecialLocation (juce::File::currentExecutableFile)
                         .getParentDirectory();
    auto repoRoot = findRepoRoot (pluginDir);
    if (repoRoot.exists())
        tryLaunchServerFromRepoRoot (repoRoot);
}

void AIMusicProcessor::processBlock (juce::AudioBuffer<float>& audio, juce::MidiBuffer& midi)
{
    audio.clear();
    juce::ScopedLock sl (midiLock);
    if (! pendingMidi.isEmpty())
    {
        midi.swapWith (pendingMidi);
        pendingMidi.clear();
    }
}

juce::AudioProcessorEditor* AIMusicProcessor::createEditor()
{
    if (! isTimerRunning())
        startTimer (2000);
    return new AIMusicEditor (*this);
}

// ── pipeline actions ──────────────────────────────────────────────────────────

void AIMusicProcessor::startProcess (const juce::String& folder)
{
    audioFolder = folder;

    // Discover repo root from the chosen folder and remember it for next launch
    auto repoRoot = findRepoRoot (juce::File (folder));
    if (repoRoot.exists())
    {
        if (auto* prefs = getPrefs())
        {
            prefs->setValue ("repoRoot", repoRoot.getFullPathName());
            prefs->saveIfNeeded();
        }
        if (! client.isServerReachable())
            tryLaunchServerFromRepoRoot (repoRoot);
    }

    client.postProcess (folder);
}

void AIMusicProcessor::startTrain()
{
    auto startDir2  = juce::File (ckptPath.isNotEmpty() ? ckptPath : audioFolder);
    auto repoRoot2  = findRepoRoot (startDir2);
    auto eventsDir  = repoRoot2.exists()
                          ? repoRoot2.getChildFile ("runs/events").getFullPathName()
                          : juce::String ("runs/events");
    client.postTrain (eventsDir,
                      ckptPath, "auto", 200);
}

double AIMusicProcessor::getHostBpm() const
{
    if (auto* ph = getPlayHead())
    {
        if (auto pos = ph->getPosition())
            if (auto bpm = pos->getBpm())
                return *bpm;
    }
    return 120.0;
}

void AIMusicProcessor::startGenerate()
{
    float bpm = syncTempo ? (float) getHostBpm() : tempoBpm;
    // Pass empty vocab_json — the server searches all known event directories.
    pendingJobId = client.postGenerate (ckptPath, {}, {},
                                        temperature, topP, bpm,
                                        forceGridStep, maxTokens);
}

// ── timer: poll status + MIDI ─────────────────────────────────────────────────

void AIMusicProcessor::timerCallback()
{
    if (! client.isServerReachable())
    {
        // Clear a dead process handle so launchServer() can run again
        if (serverProcess != nullptr && ! serverProcess->isRunning())
            serverProcess = nullptr;
        if (serverProcess == nullptr)
            launchServer();
    }

    lastStatus = client.getStatus();

    if (lastStatus.stage == "done" && pendingJobId.isNotEmpty())
        pollForMidi();

    if (onStatusChanged)
        juce::MessageManager::callAsync (onStatusChanged);
}

void AIMusicProcessor::pollForMidi()
{
    juce::MemoryBlock data;
    if (! client.fetchMidi (pendingJobId, data))
        return;

    pendingJobId.clear();

    // Parse the MIDI file and convert to a MidiBuffer
    juce::MidiFile mf;
    juce::MemoryInputStream mis (data, false);
    if (! mf.readFrom (mis)) return;

    mf.convertTimestampTicksToSeconds();
    juce::MidiBuffer buf;
    for (int t = 0; t < mf.getNumTracks(); ++t)
    {
        const juce::MidiMessageSequence* track = mf.getTrack (t);
        for (int i = 0; i < track->getNumEvents(); ++i)
        {
            auto& ev = track->getEventPointer (i)->message;
            if (ev.isNoteOnOrOff())
            {
                int samplePos = (int) (ev.getTimeStamp() * getSampleRate());
                buf.addEvent (ev, samplePos);
            }
        }
    }

    juce::ScopedLock sl (midiLock);
    pendingMidi.swapWith (buf);
}

// required by JUCE plugin factory
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new AIMusicProcessor();
}
