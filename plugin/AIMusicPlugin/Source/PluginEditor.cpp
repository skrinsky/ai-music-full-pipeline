#include "PluginEditor.h"

static const juce::Colour kBg  { 0xff1e1e2e };
static const juce::Colour kFg  { 0xffcdd6f4 };
static const juce::Colour kAcc { 0xff89b4fa };

// ── Mirror Mirror animation ───────────────────────────────────────────────────
struct MirrorMirror : public juce::Component, private juce::Timer
{
    float phase          = 0.0f;
    bool  isError        = false;
    int   errorHoldFrames { 0 };   // counts down after error clears, giving a delay
    float nodPhase       { -1.f }; // -1 = inactive; ≥0 = nod in progress
    float celebPhase     { -1.f }; // -1 = inactive; ≥0 = celebration burst + wink

    void triggerNod()         { nodPhase   = 0.f; }
    void triggerCelebration() { celebPhase = 0.f; }

    MirrorMirror()  { startTimerHz (24); }
    ~MirrorMirror() override { stopTimer(); }

    void paint (juce::Graphics& g) override
    {
        float w = (float) getWidth(), h = (float) getHeight();
        float cx = w * 0.5f;

        // Mirror oval — right portion of component; left ~36px reserved for person
        float ow = 62.f, oh = h * 0.76f;
        float oy = 12.f;
        float ox = (w - ow) * 0.5f;  // centered — no person zone needed
        auto oval = juce::Rectangle<float> (ox, oy, ow, oh);
        float rx = ow * 0.5f, ry = oh * 0.5f;
        float mcx = oval.getCentreX(), mcy = oval.getCentreY();

        // ── magical glow behind frame ─────────────────────────────────────────
        float gl = 0.28f + 0.18f * std::sin (phase * 0.7f);
        g.setColour (juce::Colour (0xff6633cc).withAlpha (gl * 0.55f));
        g.fillEllipse (oval.expanded (5.f));

        // ── thick gold frame ──────────────────────────────────────────────────
        juce::ColourGradient frameGrad (
            juce::Colour (0xffFFE566), cx, oval.getY(),
            juce::Colour (0xff7A4E00), cx, oval.getBottom(), false);
        frameGrad.addColour (0.25, juce::Colour (0xffFFD700));
        frameGrad.addColour (0.55, juce::Colour (0xffCE9E22));
        frameGrad.addColour (0.80, juce::Colour (0xff9A6B00));
        g.setGradientFill (frameGrad);
        g.fillEllipse (oval);

        // inner bevel shadow
        g.setColour (juce::Colour (0xff2a1800).withAlpha (0.55f));
        g.drawEllipse (oval.reduced (5.f), 2.f);

        // ── dark magical surface ──────────────────────────────────────────────
        auto surf_rect = oval.reduced (8.f);
        float srx = surf_rect.getWidth() * 0.5f, sry = surf_rect.getHeight() * 0.5f;
        float scx = surf_rect.getCentreX(), scy = surf_rect.getCentreY();
        juce::ColourGradient surf (
            juce::Colour (0xff160830), scx + std::sin (phase * 0.4f) * srx * 0.4f, surf_rect.getY(),
            juce::Colour (0xff091525), scx, surf_rect.getBottom(), false);
        surf.addColour (0.45, juce::Colour (0xff23104a));
        g.setGradientFill (surf);
        g.fillEllipse (surf_rect);

        // shimmer sweep
        float sw = std::sin (phase * 1.2f) * 0.5f + 0.5f;
        juce::Path shim;
        shim.addEllipse (surf_rect.getX() + surf_rect.getWidth() * (0.04f + sw * 0.55f),
                         surf_rect.getY(), surf_rect.getWidth() * 0.18f, surf_rect.getHeight());
        g.setColour (juce::Colours::white.withAlpha (0.065f));
        g.fillPath (shim);

        // ── top crown ornament ────────────────────────────────────────────────
        g.setColour (juce::Colour (0xffFFD700));
        g.fillEllipse (mcx - 5.f, oy - 9.f, 10.f, 10.f);
        g.setColour (juce::Colour (0xffFFF088));
        g.fillEllipse (mcx - 3.f, oy - 7.f, 6.f, 6.f);
        g.setColour (juce::Colour (0xffCE9E22));
        g.fillEllipse (mcx - 15.f, oy - 5.f, 7.f, 7.f);
        g.fillEllipse (mcx + 8.f,  oy - 5.f, 7.f, 7.f);
        juce::Path arch;
        arch.startNewSubPath (mcx - 11.f, oy + 2.f);
        arch.quadraticTo (mcx, oy - 5.f, mcx + 11.f, oy + 2.f);
        g.setColour (juce::Colour (0xffFFD700));
        g.strokePath (arch, juce::PathStrokeType (1.5f));

        // ── side scroll ornaments (left & right of oval) ──────────────────────
        auto drawScroll = [&] (float sx, float sy, float dir)
        {
            juce::Path sc;
            sc.startNewSubPath (sx, sy - 8.f);
            sc.cubicTo (sx + dir * 8.f, sy - 8.f, sx + dir * 10.f, sy + 1.f, sx + dir * 6.f, sy + 7.f);
            sc.cubicTo (sx + dir * 2.f, sy + 11.f, sx - dir * 1.f, sy + 7.f, sx, sy + 8.f);
            g.setColour (juce::Colour (0xffFFD700));
            g.strokePath (sc, juce::PathStrokeType (1.3f));
            g.fillEllipse (sx + dir * 4.5f - 2.f, sy - 2.f, 4.f, 4.f);
        };
        drawScroll (oval.getX() - 1.f, mcy - 6.f, -1.f);
        drawScroll (oval.getRight() + 1.f, mcy - 6.f,  1.f);

        // ── bottom scroll ─────────────────────────────────────────────────────
        float botY = oval.getBottom();
        juce::Path botScroll;
        botScroll.startNewSubPath (mcx - 11.f, botY);
        botScroll.cubicTo (mcx - 7.f, botY + 7.f, mcx - 2.f, botY + 8.f, mcx,      botY + 5.f);
        botScroll.cubicTo (mcx + 2.f, botY + 8.f, mcx + 7.f, botY + 7.f, mcx + 11.f, botY);
        g.setColour (juce::Colour (0xffCE9E22));
        g.strokePath (botScroll, juce::PathStrokeType (1.2f));
        g.setColour (juce::Colour (0xffFFD700));
        g.fillEllipse (mcx - 3.f, botY + 3.f, 6.f, 6.f);

        // ── gem dots around frame ─────────────────────────────────────────────
        g.setColour (juce::Colour (0xff8A5E00));
        g.drawEllipse (oval.reduced (1.5f), 1.f);
        for (int i = 0; i < 12; ++i)
        {
            float a  = i * juce::MathConstants<float>::twoPi / 12.f
                       - juce::MathConstants<float>::halfPi;
            float px = mcx + std::cos (a) * (rx - 3.5f);
            float py = mcy + std::sin (a) * (ry - 3.5f);
            bool big = (i % 3 == 0);
            g.setColour (big ? juce::Colour (0xffFFEE44) : juce::Colour (0xffAA8800));
            float dr = big ? 2.8f : 1.8f;
            g.fillEllipse (px - dr, py - dr, dr * 2.f, dr * 2.f);
        }

        // ── ghostly face (breathes, cursor-tracking eyes) ─────────────────────
        float vis = 0.55f + 0.40f * std::sin (phase * 0.28f);
        float es  = srx * 0.17f;

        // Nod offset: damped sine — multiple back-and-forth nods, ~5s decay
        float nodY = 0.f;
        if (nodPhase >= 0.f)
            nodY = sry * 0.28f * std::sin (nodPhase * 3.5f)
                               * std::exp  (-nodPhase * 0.28f);

        float ey  = scy - sry * 0.14f + nodY;
        float ex  = srx * 0.30f;

        // Wink: right eye closes on celebration (sin peaks ~0.4s, opens ~0.8s)
        float winkClose = (celebPhase >= 0.f)
            ? juce::jlimit (0.f, 1.f, std::sin (celebPhase * 2.5f))
            : 0.f;

        auto mouseScreen = juce::Desktop::getInstance().getMainMouseSource().getScreenPosition();
        auto mouse = getLocalPoint (nullptr, mouseScreen.toInt()).toFloat();
        auto pupilOff = [&] (float ecx2, float ecy2) -> juce::Point<float>
        {
            float dx = mouse.x - ecx2, dy = mouse.y - ecy2;
            float d  = std::sqrt (dx * dx + dy * dy);
            if (d < 0.01f) return {};
            float s = std::min (es * 0.42f, d * 0.12f);
            return { dx / d * s, dy / d * s };
        };

        float ps = es * 0.52f;

        // ── left eye (never winks) ────────────────────────────────────────────
        g.setColour (juce::Colour (0xff4488ff).withAlpha (vis * 0.35f));
        g.fillEllipse (scx - ex - es * 2.2f, ey - es * 2.2f, es * 4.4f, es * 4.4f);
        g.setColour (juce::Colour (0xffaaddff).withAlpha (vis));
        g.fillEllipse (scx - ex - es, ey - es, es * 2.f, es * 2.f);
        g.setColour (juce::Colour (0xff0a1a2e).withAlpha (vis * 0.95f));
        {
            auto lOff = pupilOff (scx - ex, ey);
            g.fillEllipse (scx - ex + lOff.x - ps, ey + lOff.y - ps, ps * 2.f, ps * 2.f);
        }

        // ── right eye (squishes closed on wink) ───────────────────────────────
        float rEyeHScale = 1.f - winkClose * 0.96f;
        g.setColour (juce::Colour (0xff4488ff).withAlpha (vis * 0.35f));
        g.fillEllipse (scx + ex - es * 2.2f, ey - es * 2.2f * rEyeHScale,
                       es * 4.4f,             es * 4.4f * rEyeHScale);
        g.setColour (juce::Colour (0xffaaddff).withAlpha (vis));
        g.fillEllipse (scx + ex - es, ey - es * rEyeHScale,
                       es * 2.f,      es * 2.f * rEyeHScale);
        if (rEyeHScale > 0.12f)   // pupil disappears as eye closes
        {
            g.setColour (juce::Colour (0xff0a1a2e).withAlpha (vis * 0.95f));
            auto rOff = pupilOff (scx + ex, ey);
            g.fillEllipse (scx + ex + rOff.x - ps,
                           ey + rOff.y - ps * rEyeHScale,
                           ps * 2.f, ps * 2.f * rEyeHScale);
        }
        // Eyelid crease appears as the eye closes
        if (winkClose > 0.45f)
        {
            float lidVis = (winkClose - 0.45f) / 0.55f;
            juce::Path lid;
            lid.startNewSubPath (scx + ex - es, ey);
            lid.quadraticTo     (scx + ex,      ey - es * 0.55f, scx + ex + es, ey);
            g.setColour (juce::Colour (0xffcceeFF).withAlpha (vis * lidVis * 0.85f));
            g.strokePath (lid, juce::PathStrokeType (1.3f));
        }

        float fmy = scy + sry * 0.24f + nodY;
        if (isError || errorHoldFrames > 0)
        {
            // errScale fades the O out after error clears (1.0 while error active)
            float errScale  = isError ? 1.0f : errorHoldFrames / 48.f;
            float mouthOpen = 0.5f + 0.5f * std::sin (phase * 2.2f);
            float omw = srx * 0.20f;                         // smaller than before
            float omh = sry * 0.18f * mouthOpen;

            if (omh > 0.5f)
            {
                // ── deep void interior ─────────────────────────────────────
                g.setColour (juce::Colour (0xff010408).withAlpha (vis * errScale));
                g.fillEllipse (scx - omw, fmy - omh * 0.5f, omw * 2.f, omh * 2.f);

                // ── violet swirl — pulses at a different rate ──────────────
                float swirl = 0.45f + 0.40f * std::sin (phase * 3.1f);
                g.setColour (juce::Colour (0xff9922ee).withAlpha (vis * swirl * 0.55f * mouthOpen * errScale));
                g.fillEllipse (scx - omw * 0.58f, fmy - omh * 0.40f,
                               omw * 1.16f, omh * 0.80f);

                // ── bright inner rim ───────────────────────────────────────
                g.setColour (juce::Colour (0xff99ddff).withAlpha (vis * mouthOpen * errScale));
                g.drawEllipse (scx - omw, fmy - omh * 0.5f,
                               omw * 2.f, omh * 2.f, 1.6f);

                // ── outer aura ring ────────────────────────────────────────
                g.setColour (juce::Colour (0xff5533cc).withAlpha (vis * mouthOpen * 0.45f * errScale));
                g.drawEllipse (scx - omw - 2.f, fmy - omh * 0.5f - 1.2f,
                               omw * 2.f + 4.f, omh * 2.f + 2.4f, 2.8f);
            }
            else
            {
                // Almost-closed — thin line so it doesn't snap to smile
                g.setColour (juce::Colour (0xff88bbff).withAlpha (vis * 0.30f * errScale));
                g.drawLine (scx - omw * 0.5f, fmy, scx + omw * 0.5f, fmy, 1.0f);
            }
        }
        else
        {
            // Normal smile arc
            juce::Path mouth;
            float mw = srx * 0.40f;
            mouth.startNewSubPath (scx - mw, fmy);
            mouth.quadraticTo (scx, fmy + sry * 0.16f, scx + mw, fmy);
            g.setColour (juce::Colour (0xff88bbff).withAlpha (vis * 0.65f));
            g.strokePath (mouth, juce::PathStrokeType (1.2f));
        }

        // ── sparkles outside frame ────────────────────────────────────────────
        for (int i = 0; i < 5; ++i)
        {
            float sp  = phase * 2.0f + i * juce::MathConstants<float>::twoPi / 5.f;
            float alp = std::max (0.f, std::sin (sp));
            if (alp < 0.05f) continue;
            float sa  = i * juce::MathConstants<float>::twoPi / 5.f + phase * 0.25f;
            float spx = mcx + std::cos (sa) * (rx + 6.f);
            float spy = mcy + std::sin (sa) * (ry + 6.f);
            float sz  = 2.2f * alp;
            g.setColour (juce::Colour (0xffFFEE88).withAlpha (alp * 0.9f));
            g.fillEllipse (spx - sz, spy - sz, sz * 2.f, sz * 2.f);
            g.setColour (juce::Colour (0xffFFFFCC).withAlpha (alp * 0.6f));
            g.drawLine (spx - sz * 2.f, spy, spx + sz * 2.f, spy, 0.8f);
            g.drawLine (spx, spy - sz * 2.f, spx, spy + sz * 2.f, 0.8f);
        }

        // ── celebration burst: particles fly from mirror center in all directions ──
        if (celebPhase >= 0.f)
        {
            constexpr int kNP = 28;
            float fade = std::max (0.f, 1.f - celebPhase / 3.8f);

            // Extra mirror glow during burst
            float burstGlow = fade * 0.7f;
            g.setColour (juce::Colour (0xffcc88ff).withAlpha (burstGlow * 0.5f));
            g.fillEllipse (oval.expanded (8.f + burstGlow * 12.f));

            static const juce::Colour kPC[] = {
                juce::Colour (0xffFFE566),  // gold
                juce::Colour (0xffFFFFFF),  // white
                juce::Colour (0xff99EEFF),  // cyan
                juce::Colour (0xffDD55FF),  // violet
                juce::Colour (0xffFFAA44),  // amber
                juce::Colour (0xff88FFCC),  // mint
                juce::Colour (0xffFF88BB),  // rose
            };

            for (int i = 0; i < kNP; ++i)
            {
                float angle = i * juce::MathConstants<float>::twoPi / kNP
                              + (i % 5) * 0.18f;
                float speed = 30.f + (i % 5) * 12.f;
                float r     = celebPhase * speed;
                float px    = mcx + std::cos (angle) * r;
                float py    = mcy + std::sin (angle) * r;
                float sz    = std::max (0.f, 4.2f - celebPhase * 0.85f)
                              * (1.f + 0.5f * (i % 2));
                float alpha = fade * (0.65f + 0.35f * std::sin (angle * 3.f + phase * 2.f));
                if (sz < 0.1f || alpha < 0.02f) continue;

                auto col = kPC[i % 7];
                g.setColour (col.withAlpha (alpha));
                g.fillEllipse (px - sz, py - sz, sz * 2.f, sz * 2.f);

                // 4-pointed star cross on every 3rd particle
                if (i % 3 == 0)
                {
                    float arm = sz * 2.4f;
                    g.setColour (col.withAlpha (alpha * 0.7f));
                    g.drawLine (px - arm, py, px + arm, py, 1.0f);
                    g.drawLine (px, py - arm, px, py + arm, 1.0f);
                    // diagonal arms for extra sparkle
                    g.setColour (col.withAlpha (alpha * 0.4f));
                    g.drawLine (px - arm * 0.7f, py - arm * 0.7f,
                                px + arm * 0.7f, py + arm * 0.7f, 0.8f);
                    g.drawLine (px + arm * 0.7f, py - arm * 0.7f,
                                px - arm * 0.7f, py + arm * 0.7f, 0.8f);
                }
            }
        }
    }

    void timerCallback() override
    {
        phase += 0.05f;
        if (isError)       errorHoldFrames = 48;   // keep O visible ~2s after error clears
        else if (errorHoldFrames > 0) --errorHoldFrames;
        if (nodPhase >= 0.f)
        {
            nodPhase += 0.065f;
            if (nodPhase > 12.0f) nodPhase = -1.f;
        }
        if (celebPhase >= 0.f)
        {
            celebPhase += 0.065f;
            if (celebPhase > 5.0f) celebPhase = -1.f;
        }
        // Drive title-sparkle animation in the parent editor
        if (auto* parent = getParentComponent())
            parent->repaint (0, 0, parent->getWidth(), 38);
        repaint();
    }
};

// Small-font LookAndFeel for cramped toggle buttons — draws an X instead of a tick
struct SmallToggleLAF : public juce::LookAndFeel_V4
{
    void drawTickBox (juce::Graphics& g, juce::Component& component,
                      float x, float y, float w, float h,
                      bool ticked, bool isEnabled,
                      bool /*highlighted*/, bool /*down*/) override
    {
        juce::ignoreUnused (isEnabled);
        auto box = juce::Rectangle<float> (x, y, w, h);
        g.setColour (component.findColour (juce::ToggleButton::tickDisabledColourId));
        g.drawRoundedRectangle (box, 2.0f, 1.0f);
        if (ticked)
        {
            g.setColour (component.findColour (juce::ToggleButton::tickColourId));
            auto inset = box.reduced (3.0f);
            g.drawLine (inset.getX(), inset.getY(), inset.getRight(), inset.getBottom(), 1.5f);
            g.drawLine (inset.getRight(), inset.getY(), inset.getX(), inset.getBottom(), 1.5f);
        }
    }

    void drawToggleButton (juce::Graphics& g, juce::ToggleButton& btn,
                           bool highlighted, bool down) override
    {
        constexpr float fontSize  = 11.5f;
        constexpr float tickWidth = fontSize * 1.1f;
        drawTickBox (g, btn, 4.0f, ((float) btn.getHeight() - tickWidth) * 0.5f,
                     tickWidth, tickWidth,
                     btn.getToggleState(), btn.isEnabled(), highlighted, down);
        g.setColour (btn.findColour (juce::ToggleButton::textColourId));
        g.setFont (fontSize);
        if (! btn.isEnabled()) g.setOpacity (0.5f);
        g.drawFittedText (btn.getButtonText(),
                          btn.getLocalBounds()
                             .withTrimmedLeft (juce::roundToInt (tickWidth) + 10)
                             .withTrimmedRight (2),
                          juce::Justification::centredLeft, 10);
    }
};

AIMusicEditor::AIMusicEditor (AIMusicProcessor& p)
    : AudioProcessorEditor (&p), proc (p),
      mirrorAnim    (std::make_unique<MirrorMirror>()),
      smallToggleLAF (std::make_unique<SmallToggleLAF>())
{
    setSize (480, 440);
    addAndMakeVisible (*mirrorAnim);

    // ── Tab bar ───────────────────────────────────────────────────────────────
    auto styleTab = [&] (juce::TextButton& btn) {
        btn.setColour (juce::TextButton::buttonColourId,  juce::Colour (0xff313244));
        btn.setColour (juce::TextButton::textColourOffId, kFg);
        addAndMakeVisible (btn);
    };
    styleTab (tabProcess);
    styleTab (tabGenerate);
    tabProcess .onClick = [this] { currentTab = 0; updateTabVisibility(); };
    tabGenerate.onClick = [this] { currentTab = 1; updateTabVisibility(); };

    auto makeLabel = [&] (juce::Label& l, const juce::String& text) {
        l.setText (text, juce::dontSendNotification);
        l.setJustificationType (juce::Justification::centred);
        l.setColour (juce::Label::textColourId, kFg);
        addAndMakeVisible (l);
    };

    // ── Tab 1: Process & Train ────────────────────────────────────────────────
    lblFolder.setText (proc.audioFolder.isNotEmpty() ? proc.audioFolder : "No folder selected",
                       juce::dontSendNotification);
    lblFolder.setColour (juce::Label::textColourId, kFg);
    lblFolder.setJustificationType (juce::Justification::centredLeft);
    addAndMakeVisible (lblFolder);

    btnBrowseFolder.onClick = [this] { browseFolder(); };
    addAndMakeVisible (btnBrowseFolder);

    lblInstruments.setText ("Instruments to include:", juce::dontSendNotification);
    lblInstruments.setColour (juce::Label::textColourId, kFg);
    addAndMakeVisible (lblInstruments);

    for (auto* chk : { &chkLeadVox, &chkHarmVox, &chkGuitar, &chkBass, &chkDrums, &chkOther }) {
        chk->setToggleState (true, juce::dontSendNotification);
        chk->setColour (juce::ToggleButton::textColourId, kFg);
        chk->setLookAndFeel (smallToggleLAF.get());
        addAndMakeVisible (chk);
    }

    makeKnob (sldSeqLen, 64, 1024, (double) proc.seqLen, 64);
    sldSeqLen.onValueChange = [this] { proc.seqLen = (int) sldSeqLen.getValue(); };
    makeLabel (lblSeqLen, "Seq Len");

    btnRunProcess.onClick = [this] {
        if (proc.audioFolder.isEmpty()) { browseFolder (true); return; }
        proc.selectedTracks = buildTracksString();
        proc.startProcess (proc.audioFolder);
    };
    btnTrain.onClick = [this] {
        if (! proc.isTrainingDataReady()) {
            localErrorMessage = "No training data, run \"Process Audio\" first.";
            updateStatusLabel();
            return;
        }
        localErrorMessage.clear();
        proc.startTrain();
    };
    addAndMakeVisible (btnRunProcess);
    addAndMakeVisible (btnTrain);

    // ── Tab 2: Generate ───────────────────────────────────────────────────────
    lblCkpt.setText (proc.ckptPath.isNotEmpty() ? proc.ckptPath : "No checkpoint selected",
                     juce::dontSendNotification);
    lblCkpt.setColour (juce::Label::textColourId, kFg);
    lblCkpt.setJustificationType (juce::Justification::centredLeft);
    addAndMakeVisible (lblCkpt);

    btnBrowseCkpt.onClick = [this] { browseCheckpoint(); };
    addAndMakeVisible (btnBrowseCkpt);

    makeKnob (sldTemperature, 0.1, 2.0, proc.temperature, 0.01);
    makeKnob (sldTopP,        0.1, 1.0, proc.topP,        0.01);
    makeKnob (sldMaxTokens,   64,  2048, proc.maxTokens,  64);
    makeKnob (sldTempo,       40,  240,  proc.tempoBpm,   0.5);

    sldTemperature.onValueChange = [this] { proc.temperature = (float) sldTemperature.getValue(); };
    sldTopP       .onValueChange = [this] { proc.topP        = (float) sldTopP.getValue(); };
    sldMaxTokens  .onValueChange = [this] { proc.maxTokens   = (int)   sldMaxTokens.getValue(); updateTokenWarning(); };
    sldTempo      .onValueChange = [this] { proc.tempoBpm    = (float) sldTempo.getValue(); };

    makeLabel (lblTemperature, "Creativity");
    makeLabel (lblTopP,        "Variety");
    makeLabel (lblMaxTokens,   "Length");
    makeLabel (lblTempo,       "Tempo");

    cmbSubdivision.addItem ("1/4",  24);
    cmbSubdivision.addItem ("1/8",  12);
    cmbSubdivision.addItem ("1/16",  6);
    cmbSubdivision.addItem ("1/32",  3);
    cmbSubdivision.setSelectedId (proc.gridSubdivision, juce::dontSendNotification);
    cmbSubdivision.onChange = [this] { proc.gridSubdivision = cmbSubdivision.getSelectedId(); };
    cmbSubdivision.setColour (juce::ComboBox::backgroundColourId, juce::Colour (0xff313244));
    cmbSubdivision.setColour (juce::ComboBox::textColourId, kFg);
    cmbSubdivision.setColour (juce::ComboBox::arrowColourId, kAcc);
    addAndMakeVisible (cmbSubdivision);
    makeLabel (lblSubdivision, "Subdiv");
    lblSubdivision.setFont (juce::Font (11.5f));

    btnTriplets.setToggleState (proc.allowTriplets, juce::dontSendNotification);
    btnTriplets.setColour (juce::ToggleButton::textColourId, kFg);
    btnTriplets.setLookAndFeel (smallToggleLAF.get());
    btnTriplets.onStateChange = [this] { proc.allowTriplets = btnTriplets.getToggleState(); };
    addAndMakeVisible (btnTriplets);

    btnSeedFromData.setToggleState (proc.seedFromData, juce::dontSendNotification);
    btnSeedFromData.setColour (juce::ToggleButton::textColourId, kFg);
    btnSeedFromData.setLookAndFeel (smallToggleLAF.get());
    btnSeedFromData.onStateChange = [this] { proc.seedFromData = btnSeedFromData.getToggleState(); };
    addAndMakeVisible (btnSeedFromData);

    btnSyncTempo.setToggleState (proc.syncTempo, juce::dontSendNotification);
    btnSyncTempo.setColour (juce::ToggleButton::textColourId, kFg);
    btnSyncTempo.setLookAndFeel (smallToggleLAF.get());
    btnSyncTempo.onStateChange = [this] {
        proc.syncTempo = btnSyncTempo.getToggleState();
        sldTempo.setEnabled (! proc.syncTempo);
        if (proc.syncTempo)
            sldTempo.setValue (proc.getHostBpm(), juce::dontSendNotification);
    };
    sldTempo.setEnabled (! proc.syncTempo);
    addAndMakeVisible (btnSyncTempo);

    btnGenerate.onClick = [this] {
        if (proc.ckptPath.isEmpty()) {
            localErrorMessage = "No model loaded, use \"Select Model\" to choose a .pt checkpoint.";
            updateStatusLabel();
            return;
        }
        localErrorMessage.clear();
        proc.startGenerate();
    };
    addAndMakeVisible (btnGenerate);

    // ── Shared ────────────────────────────────────────────────────────────────
    btnCancel.onClick = [this] { localErrorMessage.clear(); proc.cancelJob(); };
    btnCancel.setColour (juce::TextButton::buttonColourId, juce::Colour (0xff313244));
    addAndMakeVisible (btnCancel);

    lblStatus.setColour (juce::Label::textColourId, kFg);
    lblStatus.setJustificationType (juce::Justification::centredLeft);
    addAndMakeVisible (lblStatus);

    lblMessage.setColour (juce::Label::textColourId, kAcc);
    lblMessage.setJustificationType (juce::Justification::centredLeft);
    addAndMakeVisible (lblMessage);

    lblTokenWarning.setColour (juce::Label::textColourId, juce::Colour (0xffff9900));
    lblTokenWarning.setFont (juce::Font (11.0f));
    lblTokenWarning.setVisible (false);
    addAndMakeVisible (lblTokenWarning);

    btnShowMidi.setVisible (false);
    btnShowMidi.onClick = [this] {
        if (lastMidiPath.isNotEmpty())
            juce::File (lastMidiPath).revealToUser();
    };
    btnShowMidi.addMouseListener (this, false);
    addAndMakeVisible (btnShowMidi);

    updateTabVisibility();
    startTimer (1500);
}

AIMusicEditor::~AIMusicEditor()
{
    stopTimer();
    btnTriplets    .setLookAndFeel (nullptr);
    btnSeedFromData.setLookAndFeel (nullptr);
    btnSyncTempo   .setLookAndFeel (nullptr);
    for (auto* chk : { &chkLeadVox, &chkHarmVox, &chkGuitar, &chkBass, &chkDrums, &chkOther })
        chk->setLookAndFeel (nullptr);
}

void AIMusicEditor::makeKnob (juce::Slider& s, double mn, double mx, double def, double step)
{
    s.setSliderStyle (juce::Slider::RotaryVerticalDrag);
    s.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 60, 16);
    s.setRange (mn, mx, step);
    s.setValue (def, juce::dontSendNotification);
    s.setColour (juce::Slider::rotarySliderFillColourId,    kAcc);
    s.setColour (juce::Slider::rotarySliderOutlineColourId, kFg.withAlpha (0.3f));
    s.setColour (juce::Slider::textBoxTextColourId,         kFg);
    s.setColour (juce::Slider::textBoxOutlineColourId,      juce::Colours::transparentBlack);
    addAndMakeVisible (s);
}

void AIMusicEditor::paint (juce::Graphics& g)
{
    g.fillAll (kBg);

    // ── Orbiting sparkles around title (phase driven by MirrorMirror's 24fps timer) ──
    float tPhase = static_cast<MirrorMirror*> (mirrorAnim.get())->phase;
    float titleCx = getWidth() * 0.5f;
    float titleCy = 18.f;
    for (int i = 0; i < 9; ++i)
    {
        float sp  = tPhase * 2.0f + i * juce::MathConstants<float>::twoPi / 9.f;
        float alp = std::max (0.f, std::sin (sp));
        if (alp < 0.05f) continue;
        float sa  = i * juce::MathConstants<float>::twoPi / 9.f + tPhase * 0.22f;
        float spx = titleCx + std::cos (sa) * (78.f + (i % 3) * 10.f);
        float spy = titleCy + std::sin (sa) * (11.f + (i % 2) * 4.f);
        float sz  = 2.0f * alp;
        g.setColour (juce::Colour (0xffFFEE88).withAlpha (alp * 0.85f));
        g.fillEllipse (spx - sz, spy - sz, sz * 2.f, sz * 2.f);
        g.setColour (juce::Colour (0xffFFFFCC).withAlpha (alp * 0.55f));
        g.drawLine (spx - sz * 1.8f, spy, spx + sz * 1.8f, spy, 0.7f);
        g.drawLine (spx, spy - sz * 1.8f, spx, spy + sz * 1.8f, 0.7f);
    }

    // ── Title ─────────────────────────────────────────────────────────────────
    auto titleRect = getLocalBounds().removeFromTop (36);
    // Soft purple glow behind text
    g.setColour (juce::Colour (0xffaa77ff).withAlpha (0.18f + 0.07f * std::sin (tPhase * 0.4f)));
    g.setFont (juce::Font (16.5f, juce::Font::bold));
    g.drawText ("Mirror Mirror", titleRect.translated (0, 1), juce::Justification::centred);
    g.setColour (kFg);
    g.drawText ("Mirror Mirror", titleRect, juce::Justification::centred);
    // Tab underline
    auto tabLine = getLocalBounds().reduced (12);
    tabLine.removeFromTop (36 + 28);
    g.setColour (kAcc.withAlpha (0.3f));
    g.drawHorizontalLine (tabLine.getY(), (float) tabLine.getX(), (float) tabLine.getRight());
}

void AIMusicEditor::resized()
{
    auto area = getLocalBounds().reduced (12);

    // Mirror + its two action buttons stacked just above it
    constexpr int kMirrorW = 120, kMirrorH = 100;
    int mirrorX = getWidth() - kMirrorW - 4;
    int mirrorY = getHeight() - kMirrorH - 6;
    mirrorAnim ->setBounds (mirrorX, mirrorY,           kMirrorW, kMirrorH);
    btnCancel  .setBounds  (mirrorX, mirrorY - 26,      kMirrorW, 22);
    btnShowMidi.setBounds  (mirrorX, mirrorY - 26 - 26, kMirrorW, 22);

    area.removeFromTop (36); // title

    // Tab bar
    auto tabRow = area.removeFromTop (28);
    int  tabW   = tabRow.getWidth() / 2;
    tabProcess .setBounds (tabRow.removeFromLeft (tabW));
    tabGenerate.setBounds (tabRow);
    area.removeFromTop (6);

    // Reserve shared status from bottom (status + msg + warning + cancel)
    auto statusArea = area.removeFromBottom (90);

    if (currentTab == 0)
    {
        // ── Process & Train tab ──────────────────────────────────────────────
        auto folderRow = area.removeFromTop (24);
        btnBrowseFolder.setBounds (folderRow.removeFromRight (120));
        folderRow.removeFromRight (4);
        lblFolder.setBounds (folderRow);
        area.removeFromTop (6);

        lblInstruments.setBounds (area.removeFromTop (16));
        area.removeFromTop (4);

        auto stemRow = area.removeFromTop (24);
        constexpr int kStemGap = 4;
        int stemW = (stemRow.getWidth() - kStemGap * 5) / 6;
        int ix = 0;
        for (auto* chk : { &chkLeadVox, &chkHarmVox, &chkGuitar, &chkBass, &chkDrums, &chkOther }) {
            if (ix++ > 0) stemRow.removeFromLeft (kStemGap);
            chk->setBounds (stemRow.removeFromLeft (stemW));
        }
        area.removeFromTop (10);

        auto seqArea = area.removeFromTop (80);
        int seqW = seqArea.getWidth() / 5; // same column width as generate tab knobs
        auto seqCol = seqArea.removeFromLeft (seqW);
        lblSeqLen.setBounds (seqCol.removeFromBottom (18));
        sldSeqLen.setBounds (seqCol);
        area.removeFromTop (10);

        auto btnRow = area.removeFromTop (34);
        btnRunProcess.setBounds (btnRow.removeFromLeft (140));
        btnRow.removeFromLeft (6);
        btnTrain.setBounds (btnRow.removeFromLeft (90));
    }
    else
    {
        // ── Generate tab ─────────────────────────────────────────────────────
        auto ckptRow = area.removeFromTop (24);
        btnBrowseCkpt.setBounds (ckptRow.removeFromRight (90));
        ckptRow.removeFromRight (4);
        lblCkpt.setBounds (ckptRow);
        area.removeFromTop (8);

        auto knobArea = area.removeFromTop (112);
        int  knobW    = knobArea.getWidth() / 5;
        using KP = std::pair<juce::Slider*, juce::Label*>;
        // All four knobs identical height — Sync sits in its own row below
        for (auto pair : { KP {&sldTemperature, &lblTemperature},
                           KP {&sldTopP,         &lblTopP},
                           KP {&sldMaxTokens,    &lblMaxTokens},
                           KP {&sldTempo,        &lblTempo} })
        {
            auto col = knobArea.removeFromLeft (knobW);
            pair.second->setBounds (col.removeFromBottom (18));
            pair.first ->setBounds (col);
        }
        // Subdiv column
        {
            auto col = knobArea;
            lblSubdivision.setBounds (col.removeFromBottom (18));
            btnTriplets   .setBounds (col.removeFromBottom (22));
            cmbSubdivision.setBounds (col.reduced (2, 6));
        }

        // Sync toggle centered under the Tempo knob column
        area.removeFromTop (4);
        {
            auto syncRow = area.removeFromTop (20);
            syncRow.removeFromLeft (knobW * 3);
            btnSyncTempo.setBounds (syncRow.removeFromLeft (knobW)
                                           .withSizeKeepingCentre (64, 20));
        }
        area.removeFromTop (2);
        btnSeedFromData.setBounds (area.removeFromTop (22));
        area.removeFromTop (8);

        btnGenerate.setBounds (area.removeFromTop (34).removeFromLeft (140));
    }

    // ── Shared status (labels only; buttons are above the mirror) ───────────
    // Constrain labels so they don't extend under the mirror on the right.
    int labelMaxRight = mirrorX - 8;
    auto sa = statusArea.withRight (labelMaxRight);
    sa.removeFromTop (6);
    lblStatus.setBounds (sa.removeFromTop (22));
    sa.removeFromTop (4);
    lblMessage.setBounds (sa.removeFromTop (22));
    sa.removeFromTop (2);
    lblTokenWarning.setBounds (sa.removeFromTop (18));
}

void AIMusicEditor::updateTabVisibility()
{
    bool onProcess = (currentTab == 0);

    tabProcess .setColour (juce::TextButton::buttonColourId,
                           onProcess ? kAcc.withAlpha (0.25f) : juce::Colour (0xff313244));
    tabGenerate.setColour (juce::TextButton::buttonColourId,
                           !onProcess ? kAcc.withAlpha (0.25f) : juce::Colour (0xff313244));

    for (juce::Component* c : std::initializer_list<juce::Component*> {
             &lblFolder, &btnBrowseFolder, &lblInstruments,
             &chkLeadVox, &chkHarmVox, &chkGuitar, &chkBass, &chkDrums, &chkOther,
             &sldSeqLen, &lblSeqLen, &btnRunProcess, &btnTrain })
        c->setVisible (onProcess);

    for (juce::Component* c : std::initializer_list<juce::Component*> {
             &lblCkpt, &btnBrowseCkpt,
             &sldTemperature, &sldTopP, &sldMaxTokens, &sldTempo,
             &lblTemperature, &lblTopP, &lblMaxTokens, &lblTempo,
             &btnSyncTempo, &cmbSubdivision, &btnTriplets, &lblSubdivision,
             &btnSeedFromData, &btnGenerate })
        c->setVisible (!onProcess);

    resized();
}

juce::String AIMusicEditor::buildTracksString() const
{
    juce::StringArray tracks;
    if (chkLeadVox.getToggleState()) tracks.add ("voxlead");
    if (chkHarmVox.getToggleState()) tracks.add ("voxharm");
    if (chkGuitar .getToggleState()) tracks.add ("guitar");
    if (chkBass   .getToggleState()) tracks.add ("bass");
    if (chkDrums  .getToggleState()) tracks.add ("drums");
    if (chkOther  .getToggleState()) tracks.add ("other");
    if (tracks.size() == 6) return {};  // all selected = no filter
    return tracks.joinIntoString (",");
}

void AIMusicEditor::timerCallback()
{
    if (proc.syncTempo)
        sldTempo.setValue (proc.getHostBpm(), juce::dontSendNotification);
    updateTokenWarning();
    updateStatusLabel();

    auto& mm       = *static_cast<MirrorMirror*> (mirrorAnim.get());
    auto  curStage = proc.lastStatus.stage;

    mm.isError = (curStage == "error") || localErrorMessage.isNotEmpty();

    // Nod when a new job starts (transition into a running state)
    const juce::StringArray kRunning { "processing", "training", "generating" };
    if (kRunning.contains (curStage) && ! kRunning.contains (prevStage))
        mm.triggerNod();

    // Celebration burst + wink when a job finishes
    if (curStage == "done" && kRunning.contains (prevStage))
        mm.triggerCelebration();

    prevStage = curStage;
}

void AIMusicEditor::updateStatusLabel()
{
    // Client-side validation errors persist until the user clears them,
    // ignoring whatever the server timer says in the meantime.
    if (localErrorMessage.isNotEmpty())
    {
        lblStatus .setText ("Status: error",    juce::dontSendNotification);
        lblMessage.setText (localErrorMessage,  juce::dontSendNotification);
        btnCancel .setVisible (true);
        btnCancel .setButtonText ("Clear");
        btnShowMidi.setVisible (false);
        btnRunProcess.setEnabled (true);
        btnTrain     .setEnabled (true);
        btnGenerate  .setEnabled (true);
        return;
    }

    auto& s = proc.lastStatus;
    lblStatus.setText ("Status: " + s.stage, juce::dontSendNotification);

    if (s.stage == "training" && s.epoch >= 0)
    {
        juce::String ep = "Epoch " + juce::String (s.epoch);
        if (s.totalEpochs > 0) ep += " / " + juce::String (s.totalEpochs);
        if (s.valLoss >= 0)    ep += "   val loss: " + juce::String (s.valLoss, 4);
        lblMessage.setText (ep, juce::dontSendNotification);
    }
    else
    {
        // For error states: prefer the descriptive message; fall back to the short error string
        auto detail = (s.stage == "error" && s.message.isEmpty()) ? s.error : s.message;
        lblMessage.setText (detail, juce::dontSendNotification);
    }

    if (s.stage == "done" && s.message.startsWith ("midi_id="))
    {
        auto jobId = s.message.fromFirstOccurrenceOf ("midi_id=", false, false);
        if (jobId.isNotEmpty())
        {
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

    bool busy = (s.stage == "processing" || s.stage == "training" || s.stage == "generating");
    btnCancel.setVisible (busy || s.stage == "error");
    btnCancel.setButtonText (busy ? "Cancel" : "Clear");

    bool serverReady = (s.stage == "idle" || s.stage == "done" || s.stage == "error");
    btnRunProcess.setEnabled (true);
    btnTrain     .setEnabled (serverReady);
    btnGenerate  .setEnabled (serverReady);
}

void AIMusicEditor::updateTokenWarning()
{
    int  ctx  = proc.trainingCtxLen;
    bool over = ctx > 0 && (int) sldMaxTokens.getValue() > ctx;
    lblMaxTokens.setColour (juce::Label::textColourId, over ? juce::Colour (0xffff9900) : kFg);
    lblMaxTokens.setText (over ? "Length (!)" : "Length", juce::dontSendNotification);
    lblTokenWarning.setVisible (over);
    if (over)
        lblTokenWarning.setText ("Heads up: generating past the training length ("
                                 + juce::String (ctx) + " tokens) may sound unpredictable",
                                 juce::dontSendNotification);
}

void AIMusicEditor::mouseDrag (const juce::MouseEvent& e)
{
    if (e.eventComponent == &btnShowMidi && lastMidiPath.isNotEmpty())
        performExternalDragDropOfFiles (juce::StringArray { lastMidiPath }, false);
}

void AIMusicEditor::browseFolder (bool startAfterSelect)
{
    auto lastDir  = proc.getPref ("lastAudioDir");
    auto startDir = lastDir.isNotEmpty() ? juce::File (lastDir)
                                         : juce::File::getSpecialLocation (juce::File::userMusicDirectory);

    auto chooser = std::make_shared<juce::FileChooser> ("Select audio folder", startDir);
    chooser->launchAsync (juce::FileBrowserComponent::openMode |
                          juce::FileBrowserComponent::canSelectDirectories,
        [this, chooser, startAfterSelect] (const juce::FileChooser& fc)
        {
            auto folder = fc.getResult();
            if (folder.isDirectory())
            {
                proc.audioFolder = folder.getFullPathName();
                proc.setPref ("lastAudioDir", folder.getFullPathName());
                lblFolder.setText (folder.getFullPathName(), juce::dontSendNotification);
                if (startAfterSelect)
                {
                    proc.selectedTracks = buildTracksString();
                    proc.startProcess (proc.audioFolder);
                }
            }
        });
}

void AIMusicEditor::browseCheckpoint()
{
    auto lastDir  = proc.getPref ("lastCkptDir");
    auto startDir = lastDir.isNotEmpty() ? juce::File (lastDir)
                                         : juce::File::getSpecialLocation (juce::File::userHomeDirectory);

    auto chooser = std::make_shared<juce::FileChooser> ("Select model checkpoint (.pt)", startDir, "*.pt");
    chooser->launchAsync (juce::FileBrowserComponent::openMode |
                          juce::FileBrowserComponent::canSelectFiles,
        [this, chooser] (const juce::FileChooser& fc)
        {
            auto f = fc.getResult();
            if (f.existsAsFile())
            {
                proc.ckptPath = f.getFullPathName();
                proc.setPref ("lastCkptDir", f.getParentDirectory().getFullPathName());
                lblCkpt.setText (f.getFullPathName(), juce::dontSendNotification);
                proc.loadCheckpointInfo();
                updateTokenWarning();
            }
        });
}
