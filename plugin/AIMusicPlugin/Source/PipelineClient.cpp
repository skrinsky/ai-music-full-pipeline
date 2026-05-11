#include "PipelineClient.h"

PipelineClient::PipelineClient (int port)
    : baseUrl ("http://127.0.0.1:" + juce::String (port))
{}

// ── private helpers ───────────────────────────────────────────────────────────

juce::String PipelineClient::get (const juce::String& path)
{
    juce::URL url (baseUrl + path);
    int statusCode = 0;
    auto stream = url.createInputStream (juce::URL::InputStreamOptions (juce::URL::ParameterHandling::inAddress)
                                             .withConnectionTimeoutMs (1500)
                                             .withStatusCode (&statusCode));
    if (stream == nullptr) return {};
    return stream->readEntireStreamAsString();
}

juce::String PipelineClient::post (const juce::String& path, const juce::String& jsonBody)
{
    juce::URL url (baseUrl + path);
    url = url.withPOSTData (jsonBody);
    int statusCode = 0;
    juce::StringPairArray headers;
    headers.set ("Content-Type", "application/json");
    auto stream = url.createInputStream (juce::URL::InputStreamOptions (juce::URL::ParameterHandling::inPostData)
                                             .withExtraHeaders ("Content-Type: application/json")
                                             .withConnectionTimeoutMs (5000)
                                             .withStatusCode (&statusCode));
    if (stream == nullptr) return {};
    return stream->readEntireStreamAsString();
}

// ── public API ────────────────────────────────────────────────────────────────

bool PipelineClient::isServerReachable()
{
    auto resp = get ("/health");
    return resp.contains ("ok");
}

PipelineStatus PipelineClient::getStatus()
{
    auto resp = get ("/status");
    PipelineStatus s;
    if (resp.isEmpty()) { s.stage = "unreachable"; return s; }

    auto json = juce::JSON::parse (resp);
    if (auto* obj = json.getDynamicObject())
    {
        s.stage    = obj->getProperty ("stage").toString();
        s.message  = obj->getProperty ("message").toString();
        s.error    = obj->getProperty ("error").toString();
        s.ckptPath = obj->getProperty ("ckpt_path").toString();
        s.midiPath = obj->getProperty ("midi_path").toString();
        auto ep   = obj->getProperty ("epoch");
        auto vl   = obj->getProperty ("val_loss");
        auto te   = obj->getProperty ("total_epochs");
        auto pr   = obj->getProperty ("progress");
        if (! ep.isVoid())  s.epoch       = (int) ep;
        if (! vl.isVoid())  s.valLoss     = (double) vl;
        if (! te.isVoid())  s.totalEpochs = (int) te;
        if (! pr.isVoid())  s.progress      = (float) (double) pr;
        auto bp = obj->getProperty ("batch_progress");
        if (! bp.isVoid())  s.batchProgress = (float) (double) bp;
    }
    return s;
}

juce::StringArray PipelineClient::fetchExistingProcessed (const juce::String& audioFolder)
{
    auto encoded = juce::URL::addEscapeChars (audioFolder, true);
    auto resp = get ("/check_existing?audio_folder=" + encoded);
    juce::StringArray result;
    if (resp.isEmpty()) return result;
    auto json = juce::JSON::parse (resp);
    if (auto* obj = json.getDynamicObject())
        if (auto* arr = obj->getProperty ("existing").getArray())
            for (auto& v : *arr)
                result.add (v.toString());
    return result;
}

bool PipelineClient::postProcess (const juce::String&      audioFolder,
                                  const juce::String&      tracks,
                                  bool                     normalizeKey,
                                  float                    discIntensity,
                                  const juce::String&      projectName,
                                  const juce::StringArray& filesToSkip)
{
    juce::Array<juce::var> skipArr;
    for (auto& f : filesToSkip)
        skipArr.add (juce::var (f));

    auto* obj = new juce::DynamicObject();
    obj->setProperty ("audio_folder",   audioFolder);
    obj->setProperty ("tracks",         tracks);
    obj->setProperty ("normalize_key",  normalizeKey);
    obj->setProperty ("disc_intensity", (double) discIntensity);
    obj->setProperty ("project_name",   projectName);
    obj->setProperty ("files_to_skip",  juce::var (skipArr));
    auto resp = post ("/process", juce::JSON::toString (juce::var (obj)));
    return resp.contains ("started");
}

bool PipelineClient::postTrain (const juce::String& eventsDir,
                                const juce::String& ckptPath,
                                const juce::String& device,
                                int epochs,
                                int seqLen,
                                const juce::String& projectName,
                                const juce::String& pretrainCkpt,
                                bool forceRestart)
{
    auto* obj = new juce::DynamicObject();
    obj->setProperty ("events_dir",     eventsDir);
    obj->setProperty ("ckpt_path",      ckptPath);
    obj->setProperty ("device",         device);
    obj->setProperty ("epochs",         epochs);
    obj->setProperty ("seq_len",        seqLen);
    obj->setProperty ("project_name",   projectName);
    obj->setProperty ("pretrain_ckpt",  pretrainCkpt);
    obj->setProperty ("force_restart",  forceRestart);
    auto resp = post ("/train", juce::JSON::toString (juce::var (obj)));
    return resp.contains ("started");
}

std::pair<bool, int> PipelineClient::fetchCheckpointStatus (const juce::String& projectName)
{
    auto encoded = juce::URL::addEscapeChars (projectName, true);
    auto resp = get ("/checkpoint_status?project_name=" + encoded);
    if (resp.isEmpty()) return {false, -1};
    auto parsed = juce::JSON::parse (resp);
    bool exists = parsed.getProperty ("exists", false);
    int  epoch  = (int) parsed.getProperty ("epoch", -1);
    return {exists, epoch};
}

juce::String PipelineClient::postGenerate (const juce::String& ckpt,
                                           const juce::String& vocabJson,
                                           const juce::String& seedPkl,
                                           float temperature,
                                           float topP,
                                           float tempoBpm,
                                           int   gridStraightStep,
                                           int   gridTripletStep,
                                           int   maxTokens,
                                           bool  useSeed,
                                           const juce::String& projectName)
{
    auto* obj = new juce::DynamicObject();
    obj->setProperty ("ckpt",               ckpt);
    obj->setProperty ("vocab_json",         vocabJson);
    obj->setProperty ("seed_pkl",           seedPkl);
    obj->setProperty ("temperature",        temperature);
    obj->setProperty ("top_p",              topP);
    obj->setProperty ("tempo_bpm",          tempoBpm);
    obj->setProperty ("grid_straight_step", gridStraightStep);
    obj->setProperty ("grid_triplet_step",  gridTripletStep);
    obj->setProperty ("max_tokens",         maxTokens);
    obj->setProperty ("use_seed",           useSeed);
    obj->setProperty ("project_name",       projectName);
    auto resp = post ("/generate", juce::JSON::toString (juce::var (obj)));
    if (resp.isEmpty()) return {};
    auto json = juce::JSON::parse (resp);
    if (auto* parsed = json.getDynamicObject())
        return parsed->getProperty ("job_id").toString();
    return {};
}

bool PipelineClient::postCancel()
{
    auto* obj = new juce::DynamicObject();
    auto resp = post ("/cancel", juce::JSON::toString (juce::var (obj)));
    return resp.contains ("cancelled");
}

int PipelineClient::fetchCheckpointInfo (const juce::String& ckptPath)
{
    auto encoded = juce::URL::addEscapeChars (ckptPath, true);
    auto resp = get ("/checkpoint_info?ckpt=" + encoded);
    if (resp.isEmpty()) return 0;
    auto json = juce::JSON::parse (resp);
    if (auto* obj = json.getDynamicObject())
    {
        auto val = obj->getProperty ("seq_len");
        if (! val.isVoid()) return (int) val;
    }
    return 0;
}

bool PipelineClient::fetchPreviewWav (const juce::String& jobId,
                                       juce::MemoryBlock& wavData,
                                       int sampleRate,
                                       double bpm)
{
    juce::String path = "/preview/" + jobId + "?fs=" + juce::String (sampleRate);
    if (bpm > 0.0)
        path += "&bpm=" + juce::String (bpm, 2);
    juce::URL url (baseUrl + path);
    int statusCode = 0;
    auto stream = url.createInputStream (
        juce::URL::InputStreamOptions (juce::URL::ParameterHandling::inAddress)
            .withConnectionTimeoutMs (60000)   // server may need time to synthesise
            .withStatusCode (&statusCode));
    if (stream == nullptr || statusCode != 200) return false;
    wavData.reset();
    stream->readIntoMemoryBlock (wavData);
    return wavData.getSize() > 0;
}

juce::String PipelineClient::fetchLatestEvents()
{
    auto resp = get ("/latest_events");
    if (resp.isEmpty()) return {};
    auto json = juce::JSON::parse (resp);
    if (auto* obj = json.getDynamicObject())
    {
        auto val = obj->getProperty ("events_dir");
        if (! val.isVoid()) return val.toString();
    }
    return {};
}

juce::String PipelineClient::fetchDiscPreview (const juce::String& eventsDir)
{
    juce::String path = "/disc_preview";
    if (eventsDir.isNotEmpty())
        path += "?events_dir=" + juce::URL::addEscapeChars (eventsDir, true);
    return get (path);
}

bool PipelineClient::fetchMidi (const juce::String& jobId, juce::MemoryBlock& midiData)
{
    juce::URL url (baseUrl + "/midi/" + jobId);
    int statusCode = 0;
    auto stream = url.createInputStream (juce::URL::InputStreamOptions (juce::URL::ParameterHandling::inAddress)
                                             .withConnectionTimeoutMs (5000)
                                             .withStatusCode (&statusCode));
    if (stream == nullptr || statusCode != 200) return false;
    midiData.reset();
    stream->readIntoMemoryBlock (midiData);
    return midiData.getSize() > 0;
}
