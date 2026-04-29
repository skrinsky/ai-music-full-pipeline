#include "PluginProcessor.h"
#include "PluginEditor.h"

AIMusicProcessor::AIMusicProcessor()
    : AudioProcessor (BusesProperties()
                          .withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
      client (7437)
{
    // Restore last-used audio folder so isTrainingDataReady() works without re-selecting
    auto saved = getPref ("lastAudioDir");
    if (saved.isNotEmpty()) audioFolder = saved;
}

AIMusicProcessor::~AIMusicProcessor()
{
    stopTimer();
    // Server is intentionally left running so jobs survive DAW close.
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

    auto q = [] (const juce::String& s) { return "\"" + s + "\""; };

    auto activate = repoRoot.getChildFile (".venv-ai-music/bin/activate");
    juce::String shellCmd;
    if (activate.existsAsFile())
        shellCmd = ". " + q (activate.getFullPathName())
                 + " && nohup python " + q (serverScript.getFullPathName())
                 + " --root " + q (repoRoot.getFullPathName())
                 + " > /dev/null 2>&1 &";
    else
        shellCmd = "nohup python3 " + q (serverScript.getFullPathName())
                 + " --root " + q (repoRoot.getFullPathName())
                 + " > /dev/null 2>&1 &";

    // Shell exits immediately after forking the server; server survives DAW close.
    juce::ChildProcess shell;
    shell.start ({ "/bin/bash", "-c", shellCmd });
    shell.waitForProcessToFinish (3000);
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
    lastServerLaunchMs = juce::Time::currentTimeMillis();

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

    if (auto* ph = getPlayHead())
        if (auto pos = ph->getPosition())
            if (auto bpm = pos->getBpm())
                cachedBpm.store (*bpm);

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

    client.postProcess (folder, selectedTracks);
}

void AIMusicProcessor::startTrain()
{
    auto startDir2  = juce::File (ckptPath.isNotEmpty() ? ckptPath : audioFolder);
    auto repoRoot2  = findRepoRoot (startDir2);
    auto eventsDir  = repoRoot2.exists()
                          ? repoRoot2.getChildFile ("runs/events").getFullPathName()
                          : juce::String ("runs/events");
    client.postTrain (eventsDir, ckptPath, "auto", 200, seqLen);
}

void AIMusicProcessor::startGenerate()
{
    float bpm = syncTempo ? (float) getHostBpm() : tempoBpm;
    int tripletStep = allowTriplets ? (gridSubdivision * 2 / 3) : 0;
    pendingJobId = client.postGenerate (ckptPath, {}, {},
                                        temperature, topP, bpm,
                                        gridSubdivision, tripletStep, maxTokens,
                                        seedFromData);
}

juce::String AIMusicProcessor::getPref (const juce::String& key, const juce::String& fallback)
{
    if (auto* p = getPrefs()) return p->getValue (key, fallback);
    return fallback;
}

void AIMusicProcessor::setPref (const juce::String& key, const juce::String& value)
{
    if (auto* p = getPrefs()) { p->setValue (key, value); p->saveIfNeeded(); }
}

bool AIMusicProcessor::isTrainingDataReady()
{
    // Only use paths explicitly set this session — no pref fallback here.
    // audioFolder is restored from prefs on startup so returning users still work.
    juce::File startDir;
    if      (audioFolder.isNotEmpty()) startDir = juce::File (audioFolder);
    else if (ckptPath.isNotEmpty())    startDir = juce::File (ckptPath).getParentDirectory();
    else return false;

    auto repoRoot = findRepoRoot (startDir);
    if (! repoRoot.exists()) return false;

    auto eventsDir = repoRoot.getChildFile ("runs/events");
    return eventsDir.getChildFile ("events_train.pkl").existsAsFile()
        && eventsDir.getChildFile ("events_val.pkl").existsAsFile();
}

int AIMusicProcessor::loadCheckpointInfo()
{
    trainingCtxLen = client.fetchCheckpointInfo (ckptPath);
    return trainingCtxLen;
}

void AIMusicProcessor::cancelJob()
{
    client.postCancel();
    pendingJobId.clear();
}

// ── timer: poll status + MIDI ─────────────────────────────────────────────────

void AIMusicProcessor::timerCallback()
{
    if (! client.isServerReachable())
    {
        // Relaunch at most once every 15 s — gives detached server time to start up
        auto nowMs = juce::Time::currentTimeMillis();
        if (nowMs - lastServerLaunchMs > 15000)
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
