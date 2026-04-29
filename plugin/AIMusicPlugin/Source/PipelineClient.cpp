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
        s.stage   = obj->getProperty ("stage").toString();
        s.message = obj->getProperty ("message").toString();
        s.error   = obj->getProperty ("error").toString();
        auto ep   = obj->getProperty ("epoch");
        auto vl   = obj->getProperty ("val_loss");
        auto te   = obj->getProperty ("total_epochs");
        if (! ep.isVoid())  s.epoch       = (int) ep;
        if (! vl.isVoid())  s.valLoss     = (double) vl;
        if (! te.isVoid())  s.totalEpochs = (int) te;
    }
    return s;
}

bool PipelineClient::postProcess (const juce::String& audioFolder,
                                  const juce::String& tracks,
                                  bool normalizeKey)
{
    auto* obj = new juce::DynamicObject();
    obj->setProperty ("audio_folder",  audioFolder);
    obj->setProperty ("tracks",        tracks);
    obj->setProperty ("normalize_key", normalizeKey);
    auto resp = post ("/process", juce::JSON::toString (juce::var (obj)));
    return resp.contains ("started");
}

bool PipelineClient::postTrain (const juce::String& eventsDir,
                                const juce::String& ckptPath,
                                const juce::String& device,
                                int epochs,
                                int seqLen)
{
    auto* obj = new juce::DynamicObject();
    obj->setProperty ("events_dir", eventsDir);
    obj->setProperty ("ckpt_path",  ckptPath);
    obj->setProperty ("device",     device);
    obj->setProperty ("epochs",     epochs);
    obj->setProperty ("seq_len",    seqLen);
    auto resp = post ("/train", juce::JSON::toString (juce::var (obj)));
    return resp.contains ("started");
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
                                           bool  useSeed)
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
