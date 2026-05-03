#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <algorithm>

#if JUCE_MAC
# include <AudioUnit/AudioUnit.h>
# include <AudioToolbox/AudioToolbox.h>
static inline AudioUnit toAU (void* p) { return static_cast<AudioUnit> (p); }
#else
// ── Non-macOS preview: background download thread ─────────────────────────────
struct AIMusicProcessor::PreviewDownloadThread : public juce::Thread
{
    AIMusicProcessor&                    proc;
    juce::String                         jobId;
    int                                  sampleRate;
    double                               bpm;
    int                                  gen;
    std::shared_ptr<std::atomic<bool>>   alive;

    PreviewDownloadThread (AIMusicProcessor& p, const juce::String& id,
                           int fs, double bpm_, int g,
                           std::shared_ptr<std::atomic<bool>> a)
        : juce::Thread ("MirrorPreview"), proc (p),
          jobId (id), sampleRate (fs), bpm (bpm_), gen (g), alive (std::move (a))
    {}

    void run() override
    {
        juce::MemoryBlock wav;
        bool ok = proc.client.fetchPreviewWav (jobId, wav, sampleRate, bpm);
        if (threadShouldExit()) return; // destructor is waiting — don't touch proc

        auto aliveCopy = alive;
        int  genCopy   = gen;

        if (ok)
        {
            juce::MessageManager::callAsync (
                [&proc = proc, wav = std::move (wav), aliveCopy, genCopy]() mutable
                {
                    if (aliveCopy->load())
                        proc.loadPreviewWav (std::move (wav), genCopy);
                });
        }
        else
        {
            juce::MessageManager::callAsync (
                [&proc = proc, aliveCopy]()
                {
                    if (aliveCopy->load() && proc.onPreviewStateChanged)
                        proc.onPreviewStateChanged (false);
                });
        }
    }
};
#endif

AIMusicProcessor::AIMusicProcessor()
    : AudioProcessor (BusesProperties()
                          .withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
      client (7437)
{
    // Restore last-used audio folder so isTrainingDataReady() works without re-selecting
    auto saved = getPref ("lastAudioDir");
    if (saved.isNotEmpty()) audioFolder = saved;

#if JUCE_MAC
    // Apple DLS MusicDevice (GM sounds, built into every Mac)
    AudioComponentDescription desc = {};
    desc.componentType         = kAudioUnitType_MusicDevice;
    desc.componentSubType      = kAudioUnitSubType_DLSSynth;
    desc.componentManufacturer = kAudioUnitManufacturer_Apple;
    if (auto* comp = AudioComponentFindNext (nullptr, &desc))
    {
        AudioUnit au = nullptr;
        if (AudioComponentInstanceNew (comp, &au) == noErr && au != nullptr)
        {
            if (AudioUnitInitialize (au) == noErr)
                previewDLSSynth = au;
            else
                AudioComponentInstanceDispose (au);
        }
    }
#else
    previewFormatManager.registerBasicFormats();
#endif
}

void AIMusicProcessor::prepareToPlay (double sampleRate, int blockSize)
{
#if JUCE_MAC
    if (auto au = toAU (previewDLSSynth))
    {
        auto sr = sampleRate;
        AudioUnitSetProperty (au, kAudioUnitProperty_SampleRate,
                              kAudioUnitScope_Output, 0, &sr, sizeof (sr));
        auto mf = (UInt32) blockSize;
        AudioUnitSetProperty (au, kAudioUnitProperty_MaximumFramesPerSlice,
                              kAudioUnitScope_Global, 0, &mf, sizeof (mf));
        AudioUnitReset (au, kAudioUnitScope_Global, 0);
    }
#else
    previewTransport.prepareToPlay (blockSize, sampleRate);
#endif
}

void AIMusicProcessor::releaseResources()
{
#if ! JUCE_MAC
    previewTransport.releaseResources();
#endif
}

AIMusicProcessor::~AIMusicProcessor()
{
    stopTimer();
#if JUCE_MAC
    if (auto au = toAU (previewDLSSynth))
    {
        AudioUnitUninitialize (au);
        AudioComponentInstanceDispose (au);
    }
#else
    // Signal alive=false BEFORE stopping the thread so any queued callAsync returns early
    previewAlive->store (false);
    if (previewDownloadThread != nullptr)
        previewDownloadThread->stopThread (5000);
    previewTransport.stop();
    previewTransport.setSource (nullptr);
#endif
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

    {
        juce::ScopedLock sl (midiLock);
        if (! pendingMidi.isEmpty())
        {
            midi.swapWith (pendingMidi);
            pendingMidi.clear();
        }
    }

#if JUCE_MAC
    {
        auto au = toAU (previewDLSSynth);

        if (previewResetPending.exchange (false) && au)
            AudioUnitReset (au, kAudioUnitScope_Global, 0);

        if (previewStopRequest.exchange (false) && au)
        {
            for (int ch = 0; ch < 16; ++ch)
                MusicDeviceMIDIEvent (au, (UInt32)(0xB0 | ch), 123, 0, 0);
            previewActive.store (false);
        }

        if (previewActive.load() && au)
        {
            juce::ScopedTryLock sl (previewLock);
            if (sl.isLocked())
            {
                const double sr         = getSampleRate();
                const int    numSamples = audio.getNumSamples();

                while (previewNextEvent < (int) previewEvents.size())
                {
                    auto& [tSec, msg]    = previewEvents[(size_t) previewNextEvent];
                    juce::int64 evSample = (juce::int64) (tSec * sr);
                    if (evSample >= previewSamplePos + numSamples) break;
                    auto offset = (UInt32) juce::jlimit<juce::int64> (0, numSamples - 1,
                                                                       evSample - previewSamplePos);
                    const auto* raw = msg.getRawData();
                    auto        sz  = msg.getRawDataSize();
                    MusicDeviceMIDIEvent (au, raw[0],
                                          sz > 1 ? (UInt32) raw[1] : 0,
                                          sz > 2 ? (UInt32) raw[2] : 0,
                                          offset);
                    ++previewNextEvent;
                }

                const int nCh = audio.getNumChannels();
                struct { AudioBufferList head; AudioBuffer extra; } abl;
                abl.head.mNumberBuffers              = 2;
                abl.head.mBuffers[0].mNumberChannels = 1;
                abl.head.mBuffers[0].mDataByteSize   = (UInt32)(numSamples * sizeof (float));
                abl.head.mBuffers[0].mData           = audio.getWritePointer (0);
                abl.extra.mNumberChannels            = 1;
                abl.extra.mDataByteSize              = (UInt32)(numSamples * sizeof (float));
                abl.extra.mData                      = audio.getWritePointer (nCh > 1 ? 1 : 0);

                AudioTimeStamp ts = {};
                ts.mFlags      = kAudioTimeStampSampleTimeValid;
                ts.mSampleTime = (Float64) previewSamplePos;
                AudioUnitRenderActionFlags renderFlags = 0;
                AudioUnitRender (au, &renderFlags, &ts, 0, (UInt32) numSamples, &abl.head);

                previewSamplePos += numSamples;

                if (previewNextEvent >= (int) previewEvents.size()
                    && previewSamplePos > (juce::int64) ((previewDuration + 2.0) * sr))
                {
                    for (int ch = 0; ch < 16; ++ch)
                        MusicDeviceMIDIEvent (au, (UInt32)(0xB0 | ch), 123, 0, 0);
                    previewActive.store (false);
                    if (onPreviewStateChanged)
                        juce::MessageManager::callAsync ([this] { onPreviewStateChanged (false); });
                }
            }
        }
    }
#else
    // Non-macOS: play back the WAV downloaded from the server
    if (previewStopRequest.exchange (false))
        previewActive.store (false);

    if (previewActive.load())
    {
        juce::AudioSourceChannelInfo info (&audio, 0, audio.getNumSamples());
        previewTransport.getNextAudioBlock (info);

        if (! previewTransport.isPlaying())
        {
            previewActive.store (false);
            if (onPreviewStateChanged)
                juce::MessageManager::callAsync ([this] { onPreviewStateChanged (false); });
        }
    }
#endif
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

void AIMusicProcessor::startTrain (const juce::String& eventsDir)
{
    if (eventsDir.isNotEmpty())
    {
        client.postTrain (eventsDir, ckptPath, "auto", 200, 512);
        return;
    }
    auto startDir2 = juce::File (ckptPath.isNotEmpty() ? ckptPath : audioFolder);
    auto repoRoot2 = findRepoRoot (startDir2);
    auto dir       = repoRoot2.exists()
                         ? repoRoot2.getChildFile ("runs/events").getFullPathName()
                         : juce::String ("runs/events");
    client.postTrain (dir, ckptPath, "auto", 200, 512);
}

void AIMusicProcessor::startGenerate()
{
    float bpm          = syncTempo ? (float) getHostBpm() : tempoBpm;
    int straightStep   = quantize ? gridSubdivision : 0;
    int tripletStep    = (quantize && allowTriplets) ? (gridSubdivision * 2 / 3) : 0;
    pendingJobId = client.postGenerate (ckptPath, {}, {},
                                        temperature, topP, bpm,
                                        straightStep, tripletStep, maxTokens,
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
    // audioFolder is the only valid anchor — restored from prefs on startup so
    // returning users still work. ckptPath is intentionally NOT used here: a
    // loaded model does not mean preprocessing has been run.
    if (audioFolder.isEmpty()) return false;
    juce::File startDir = juce::File (audioFolder);

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

void AIMusicProcessor::startPreview (const juce::String& midiFilePath)
{
#if JUCE_MAC
    if (toAU (previewDLSSynth) == nullptr) return;

    juce::File f (midiFilePath);
    if (! f.existsAsFile()) return;

    juce::FileInputStream stream (f);
    if (! stream.openedOk()) return;

    juce::MidiFile mf;
    if (! mf.readFrom (stream)) return;

    // Convert ticks → seconds using the current DAW BPM so preview tempo stays
    // locked to the session, regardless of what's embedded in the MIDI file.
    // Fall back to the MIDI-embedded tempo for SMPTE-format files (getTimeFormat < 0).
    const int timeFormat = mf.getTimeFormat();
    const double dawBpm  = cachedBpm.load();
    const bool   useDawBpm = (timeFormat > 0 && dawBpm > 0.0);
    if (! useDawBpm)
        mf.convertTimestampTicksToSeconds();

    const double ticksPerBeat    = useDawBpm ? (double) timeFormat : 1.0;
    const double secondsPerTick  = useDawBpm ? 60.0 / (dawBpm * ticksPerBeat) : 1.0;

    std::vector<std::pair<double, juce::MidiMessage>> events;
    double maxT = 0.0;

    for (int t = 0; t < mf.getNumTracks(); ++t)
    {
        const auto* track = mf.getTrack (t);
        for (int i = 0; i < track->getNumEvents(); ++i)
        {
            const auto& msg = track->getEventPointer (i)->message;
            if (msg.isNoteOnOrOff() || msg.isProgramChange() || msg.isController())
            {
                double ts = msg.getTimeStamp() * (useDawBpm ? secondsPerTick : 1.0);
                events.push_back ({ ts, msg });
                if (msg.isNoteOnOrOff()) maxT = std::max (maxT, ts);
            }
        }
    }

    std::sort (events.begin(), events.end(),
               [] (const auto& a, const auto& b) { return a.first < b.first; });

    {
        juce::ScopedLock sl (previewLock);
        previewEvents    = std::move (events);
        previewDuration  = maxT;
        previewNextEvent = 0;
        previewSamplePos = 0;
    }

    previewResetPending.store (true);
    previewStopRequest .store (false);
    previewActive      .store (true);

    if (onPreviewStateChanged)
        onPreviewStateChanged (true);
#else
    // Job ID is the parent directory name of the MIDI file (e.g. ".../plugin/<jobId>/generated.mid")
    auto jobId = juce::File (midiFilePath).getParentDirectory().getFileName();
    if (jobId.isEmpty()) return;

    int gen = previewDownloadGen.fetch_add (1) + 1;

    // Stop any previous download; it will see threadShouldExit() and bail
    if (previewDownloadThread != nullptr)
        previewDownloadThread->stopThread (3000);

    previewStopRequest.store (false);
    previewActive     .store (false); // will flip to true once WAV is loaded

    if (onPreviewStateChanged)
        onPreviewStateChanged (true); // show "Stop" while downloading

    int fs = (int) getSampleRate();
    if (fs <= 0) fs = 44100;
    previewDownloadThread = std::make_unique<PreviewDownloadThread> (
        *this, jobId, fs, cachedBpm.load(), gen, previewAlive);
    previewDownloadThread->startThread();
#endif
}

void AIMusicProcessor::stopPreview()
{
#if JUCE_MAC
    previewStopRequest.store (true); // audio thread sends all-notes-off CC safely
#else
    previewActive.store (false);
    if (previewDownloadThread != nullptr)
        previewDownloadThread->stopThread (3000);
    juce::MessageManager::callAsync ([this] {
        previewTransport.stop();
        previewTransport.setSource (nullptr);
        previewReaderSource.reset();
    });
#endif
    if (onPreviewStateChanged)
        onPreviewStateChanged (false);
}

#if ! JUCE_MAC
void AIMusicProcessor::loadPreviewWav (juce::MemoryBlock&& wavData, int gen)
{
    // Called on the message thread. Discard if a newer download has started.
    if (gen != previewDownloadGen.load()) return;

    // Write to a temp file so AudioFormatReaderSource can seek
    previewTempFile = juce::File::createTempFile ("wav");
    {
        juce::FileOutputStream fos (previewTempFile);
        if (! fos.openedOk()) return;
        fos.write (wavData.getData(), wavData.getSize());
    }

    auto* reader = previewFormatManager.createReaderFor (previewTempFile);
    if (reader == nullptr) return;

    previewReaderSource = std::make_unique<juce::AudioFormatReaderSource> (reader, true);
    previewTransport.setSource (previewReaderSource.get(), 0, nullptr, reader->sampleRate);
    previewTransport.setPosition (0.0);
    previewTransport.start();
    previewActive.store (true);
}
#endif

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

// ── preset / session state ────────────────────────────────────────────────────

void AIMusicProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    juce::XmlElement xml ("MirrorMirrorPreset");
    xml.setAttribute ("version",        1);
    xml.setAttribute ("temperature",    temperature);
    xml.setAttribute ("topP",           topP);
    xml.setAttribute ("tempoBpm",       tempoBpm);
    xml.setAttribute ("gridSubdivision", gridSubdivision);
    xml.setAttribute ("allowTriplets",  allowTriplets ? 1 : 0);
    xml.setAttribute ("maxTokens",      maxTokens);
    xml.setAttribute ("syncTempo",      syncTempo     ? 1 : 0);
    xml.setAttribute ("seedFromData",   seedFromData  ? 1 : 0);
    xml.setAttribute ("quantize",       quantize      ? 1 : 0);
    xml.setAttribute ("ckptPath",       ckptPath);
    xml.setAttribute ("audioFolder",    audioFolder);
    xml.setAttribute ("selectedTracks", selectedTracks);
    copyXmlToBinary (xml, destData);
}

void AIMusicProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    auto xml = getXmlFromBinary (data, sizeInBytes);
    if (xml == nullptr || xml->getTagName() != "MirrorMirrorPreset") return;
    temperature     = (float) xml->getDoubleAttribute ("temperature",    temperature);
    topP            = (float) xml->getDoubleAttribute ("topP",           topP);
    tempoBpm        = (float) xml->getDoubleAttribute ("tempoBpm",       tempoBpm);
    gridSubdivision =         xml->getIntAttribute    ("gridSubdivision", gridSubdivision);
    allowTriplets   =         xml->getIntAttribute    ("allowTriplets",  allowTriplets ? 1 : 0) != 0;
    maxTokens       =         xml->getIntAttribute    ("maxTokens",      maxTokens);
    syncTempo       =         xml->getIntAttribute    ("syncTempo",      syncTempo    ? 1 : 0) != 0;
    seedFromData    =         xml->getIntAttribute    ("seedFromData",   seedFromData ? 1 : 0) != 0;
    quantize        =         xml->getIntAttribute    ("quantize",       quantize     ? 1 : 0) != 0;
    ckptPath        =         xml->getStringAttribute ("ckptPath",       ckptPath);
    audioFolder     =         xml->getStringAttribute ("audioFolder",    audioFolder);
    selectedTracks  =         xml->getStringAttribute ("selectedTracks", selectedTracks);
    if (onStateLoaded)
        juce::MessageManager::callAsync (onStateLoaded);
}

// required by JUCE plugin factory
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new AIMusicProcessor();
}
