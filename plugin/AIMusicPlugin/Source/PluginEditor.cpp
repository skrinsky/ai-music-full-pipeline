#include "PluginEditor.h"

static const juce::Colour kBg  { 0xff1e1e2e };
static const juce::Colour kFg  { 0xffcdd6f4 };
static const juce::Colour kAcc { 0xff89b4fa };

// ── Mirror Mirror animation ───────────────────────────────────────────────────
struct MirrorMirror : public juce::Component, public juce::TooltipClient, private juce::Timer
{
    float phase          = 0.0f;
    bool  isError        = false;
    int   errorHoldFrames { 0 };   // counts down after error clears, giving a delay
    float nodPhase       { -1.f }; // -1 = inactive; >=0 = nod in progress
    float shakePhase     { -1.f }; // -1 = inactive; >=0 = head-shake (error)
    float celebPhase     { -1.f }; // -1 = inactive; >=0 = celebration burst + wink
    float boopPhase      { -1.f }; // -1 = inactive; >=0 = tickled shimmy

    void triggerNod()         { nodPhase   = 0.f; }
    void triggerShake()       { shakePhase = 0.f; }
    void triggerCelebration() { celebPhase = 0.f; }
    void triggerBoop()        { boopPhase  = 0.f; }

    juce::Point<float> noseCenter() const
    {
        float w  = (float) getWidth(), h = (float) getHeight();
        float ow = 62.f, oh = h * 0.76f, oy = 12.f;
        float ox = (w - ow) * 0.5f;
        auto surf = juce::Rectangle<float> (ox + 8.f, oy + 8.f, ow - 16.f, oh - 16.f);
        float sry = surf.getHeight() * 0.5f;
        return { surf.getCentreX(), surf.getCentreY() + sry * 0.06f };
    }

    juce::String getTooltip() override
    {
        if (noseCenter().getDistanceFrom (getMouseXYRelative().toFloat()) < 9.f)
            return "boop me!";
        return {};
    }

    void mouseDown (const juce::MouseEvent& e) override
    {
        if (noseCenter().getDistanceFrom (e.position) < 9.f)
            triggerBoop();
    }

    MirrorMirror()  { startTimerHz (24); }
    ~MirrorMirror() override { stopTimer(); }

    void paint (juce::Graphics& g) override
    {
        float w = (float) getWidth(), h = (float) getHeight();
        float cx = w * 0.5f;

        // Mirror oval -right portion of component; left ~36px reserved for person
        float ow = 62.f, oh = h * 0.76f;
        float oy = 12.f;
        float ox = (w - ow) * 0.5f;  // centered -no person zone needed
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

        // Nod (up-down) and shake (left-right): damped sine, same envelope
        float nodY = 0.f;
        if (nodPhase >= 0.f)
            nodY = sry * 0.28f * std::sin (nodPhase * 3.5f)
                               * std::exp  (-nodPhase * 0.28f);
        float shakeX = 0.f;
        if (shakePhase >= 0.f)
            shakeX = srx * 0.28f * std::sin (shakePhase * 3.5f)
                                 * std::exp  (-shakePhase * 0.28f);
        // Boop: fast multi-axis jiggle at two incommensurate frequencies, decays quickly
        float boopX = 0.f, boopY = 0.f;
        if (boopPhase >= 0.f)
        {
            float env = std::exp (-boopPhase * 1.4f);
            boopX = srx * 0.10f * std::sin (boopPhase * 13.1f) * env;
            boopY = sry * 0.08f * std::sin (boopPhase * 17.7f) * env;
        }
        float fcx = scx + shakeX + boopX;  // face center x

        // Tickle shimmer flash
        if (boopPhase >= 0.f)
        {
            float bf = std::exp (-boopPhase * 2.5f);
            g.setColour (juce::Colour (0xffFF99CC).withAlpha (bf * 0.28f));
            g.fillEllipse (surf_rect);
        }

        float ey  = scy - sry * 0.14f + nodY + boopY;
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
        g.fillEllipse (fcx - ex - es * 2.2f, ey - es * 2.2f, es * 4.4f, es * 4.4f);
        g.setColour (juce::Colour (0xffaaddff).withAlpha (vis));
        g.fillEllipse (fcx - ex - es, ey - es, es * 2.f, es * 2.f);
        g.setColour (juce::Colour (0xff0a1a2e).withAlpha (vis * 0.95f));
        {
            auto lOff = pupilOff (fcx - ex, ey);
            g.fillEllipse (fcx - ex + lOff.x - ps, ey + lOff.y - ps, ps * 2.f, ps * 2.f);
        }

        // ── right eye (squishes closed on wink) ───────────────────────────────
        float rEyeHScale = 1.f - winkClose * 0.96f;
        g.setColour (juce::Colour (0xff4488ff).withAlpha (vis * 0.35f));
        g.fillEllipse (fcx + ex - es * 2.2f, ey - es * 2.2f * rEyeHScale,
                       es * 4.4f,             es * 4.4f * rEyeHScale);
        g.setColour (juce::Colour (0xffaaddff).withAlpha (vis));
        g.fillEllipse (fcx + ex - es, ey - es * rEyeHScale,
                       es * 2.f,      es * 2.f * rEyeHScale);
        if (rEyeHScale > 0.12f)   // pupil disappears as eye closes
        {
            g.setColour (juce::Colour (0xff0a1a2e).withAlpha (vis * 0.95f));
            auto rOff = pupilOff (fcx + ex, ey);
            g.fillEllipse (fcx + ex + rOff.x - ps,
                           ey + rOff.y - ps * rEyeHScale,
                           ps * 2.f, ps * 2.f * rEyeHScale);
        }
        // Eyelid crease appears as the eye closes
        if (winkClose > 0.45f)
        {
            float lidVis = (winkClose - 0.45f) / 0.55f;
            juce::Path lid;
            lid.startNewSubPath (fcx + ex - es, ey);
            lid.quadraticTo     (fcx + ex,      ey - es * 0.55f, fcx + ex + es, ey);
            g.setColour (juce::Colour (0xffcceeFF).withAlpha (vis * lidVis * 0.85f));
            g.strokePath (lid, juce::PathStrokeType (1.3f));
        }

        float fmy = scy + sry * 0.24f + nodY + boopY;
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
                g.fillEllipse (fcx - omw, fmy - omh * 0.5f, omw * 2.f, omh * 2.f);

                // ── violet swirl -pulses at a different rate ──────────────
                float swirl = 0.45f + 0.40f * std::sin (phase * 3.1f);
                g.setColour (juce::Colour (0xff9922ee).withAlpha (vis * swirl * 0.55f * mouthOpen * errScale));
                g.fillEllipse (fcx - omw * 0.58f, fmy - omh * 0.40f,
                               omw * 1.16f, omh * 0.80f);

                // ── bright inner rim ───────────────────────────────────────
                g.setColour (juce::Colour (0xff99ddff).withAlpha (vis * mouthOpen * errScale));
                g.drawEllipse (fcx - omw, fmy - omh * 0.5f,
                               omw * 2.f, omh * 2.f, 1.6f);

                // ── outer aura ring ────────────────────────────────────────
                g.setColour (juce::Colour (0xff5533cc).withAlpha (vis * mouthOpen * 0.45f * errScale));
                g.drawEllipse (fcx - omw - 2.f, fmy - omh * 0.5f - 1.2f,
                               omw * 2.f + 4.f, omh * 2.f + 2.4f, 2.8f);
            }
            else
            {
                // Almost-closed -thin line so it doesn't snap to smile
                g.setColour (juce::Colour (0xff88bbff).withAlpha (vis * 0.30f * errScale));
                g.drawLine (fcx - omw * 0.5f, fmy, fcx + omw * 0.5f, fmy, 1.0f);
            }
        }
        else
        {
            // Normal smile arc
            juce::Path mouth;
            float mw = srx * 0.40f;
            mouth.startNewSubPath (fcx - mw, fmy);
            mouth.quadraticTo (fcx, fmy + sry * 0.16f, fcx + mw, fmy);
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
        if (shakePhase >= 0.f)
        {
            shakePhase += 0.065f;
            if (shakePhase > 12.0f) shakePhase = -1.f;
        }
        if (celebPhase >= 0.f)
        {
            celebPhase += 0.065f;
            if (celebPhase > 5.0f) celebPhase = -1.f;
        }
        if (boopPhase >= 0.f)
        {
            boopPhase += 0.075f;
            if (boopPhase > 4.0f) boopPhase = -1.f;
        }
        // Drive all animated painting in the parent editor (title sparkles + button pulses)
        if (auto* parent = getParentComponent())
            parent->repaint();
        repaint();
    }
};

// ── Eye knob -used for the lone Seq Len knob on the Process & Train tab ─────
struct MirrorEyeKnobLAF : public juce::LookAndFeel_V4
{
    void drawRotarySlider (juce::Graphics& g, int x, int y, int width, int height,
                           float sliderPos, float startAngle, float endAngle,
                           juce::Slider&) override
    {
        auto constexpr twoPi  = juce::MathConstants<float>::twoPi;

        float cx = x + width  * 0.5f;
        float cy = y + height * 0.5f;
        float r  = std::min (width, height) * 0.5f - 3.f;
        if (r < 5.f) return;

        // Same outer purple glow as orb knobs
        g.setColour (juce::Colour (0xff6633cc).withAlpha (0.22f));
        g.fillEllipse (cx - r - 4.f, cy - r - 4.f, (r + 4.f) * 2.f, (r + 4.f) * 2.f);

        // Same gold frame gradient -visual family connection
        juce::ColourGradient frameGrad (
            juce::Colour (0xffFFE566), cx, cy - r,
            juce::Colour (0xff7A4E00), cx, cy + r, false);
        frameGrad.addColour (0.30, juce::Colour (0xffFFD700));
        frameGrad.addColour (0.70, juce::Colour (0xffCE9E22));
        g.setGradientFill (frameGrad);
        g.fillEllipse (cx - r, cy - r, r * 2.f, r * 2.f);

        // ── Iris surface ───────────────────────��──────────────────────────────
        float sr = r - 5.f;
        juce::ColourGradient irisBase (
            juce::Colour (0xff0d0510), cx, cy - sr,
            juce::Colour (0xff1e0a28), cx, cy + sr, false);
        g.setGradientFill (irisBase);
        g.fillEllipse (cx - sr, cy - sr, sr * 2.f, sr * 2.f);

        // Radial striations -thin rays from pupil edge outward
        int nRays = 36;
        for (int i = 0; i < nRays; ++i)
        {
            float a      = i * twoPi / nRays;
            float inner  = sr * 0.30f;
            bool  major  = (i % 3 == 0);
            float outer  = sr * (major ? 0.96f : 0.88f);
            float thick  = major ? 0.9f : 0.5f;
            float alpha  = major ? 0.38f : 0.18f;
            g.setColour (juce::Colour (0xffB8860B).withAlpha (alpha));
            g.drawLine (cx + std::cos (a) * inner, cy + std::sin (a) * inner,
                        cx + std::cos (a) * outer, cy + std::sin (a) * outer, thick);
        }

        // Collarette ring (just outside pupil -like real iris anatomy)
        float colR = sr * 0.36f;
        g.setColour (juce::Colour (0xffCE9E22).withAlpha (0.25f));
        g.drawEllipse (cx - colR, cy - colR, colR * 2.f, colR * 2.f, 1.0f);

        // Outer limbal shadow (darkens the iris rim, like a real eye)
        g.setColour (juce::Colour (0xff000000).withAlpha (0.30f));
        g.drawEllipse (cx - sr + 1.f, cy - sr + 1.f, (sr - 1.f) * 2.f, (sr - 1.f) * 2.f, 3.0f);

        // ── Pupil ─────────────────────────────────────────────────────────────
        float pr = sr * 0.28f;
        g.setColour (juce::Colour (0xff030108));
        g.fillEllipse (cx - pr, cy - pr, pr * 2.f, pr * 2.f);
        // Deep purple core glow
        g.setColour (juce::Colour (0xff7711cc).withAlpha (0.60f));
        g.fillEllipse (cx - pr * 0.75f, cy - pr * 0.75f, pr * 1.5f, pr * 1.5f);
        // Bright inner spark
        g.setColour (juce::Colour (0xff99bbff).withAlpha (0.40f));
        g.fillEllipse (cx - pr * 0.38f, cy - pr * 0.38f, pr * 0.75f, pr * 0.75f);

        // ── Value pointer -gold spoke from pupil edge to iris ────────────────
        float curAngle = startAngle + (endAngle - startAngle) * sliderPos;
        float si = pr * 1.05f,  so = sr * 0.87f;
        float sx1 = cx + std::sin (curAngle) * si,  sy1 = cy - std::cos (curAngle) * si;
        float sx2 = cx + std::sin (curAngle) * so,  sy2 = cy - std::cos (curAngle) * so;
        // Glow
        g.setColour (juce::Colour (0xffFFD700).withAlpha (0.22f));
        g.drawLine (sx1, sy1, sx2, sy2, 4.2f);
        // Bright spoke
        g.setColour (juce::Colour (0xffFFEE88).withAlpha (0.92f));
        g.drawLine (sx1, sy1, sx2, sy2, 1.2f);
        // Tip dot
        g.setColour (juce::Colours::white.withAlpha (0.85f));
        g.fillEllipse (sx2 - 2.f, sy2 - 2.f, 4.f, 4.f);

        // ── Specular glint (light catching the eye surface) ───────────────────
        g.setColour (juce::Colours::white.withAlpha (0.50f));
        g.fillEllipse (cx - sr * 0.28f, cy - sr * 0.62f, sr * 0.22f, sr * 0.11f);
        g.setColour (juce::Colours::white.withAlpha (0.20f));
        g.fillEllipse (cx - sr * 0.16f, cy - sr * 0.50f, sr * 0.11f, sr * 0.07f);
    }
};

// ── Magical knob LookAndFeel ─────────────────────────────────────────────────
struct MirrorKnobLAF : public juce::LookAndFeel_V4
{
    void drawRotarySlider (juce::Graphics& g, int x, int y, int width, int height,
                           float sliderPos, float startAngle, float endAngle,
                           juce::Slider& slider) override
    {
        auto constexpr twoPi  = juce::MathConstants<float>::twoPi;
        auto constexpr halfPi = juce::MathConstants<float>::halfPi;

        float cx = x + width  * 0.5f;
        float cy = y + height * 0.5f;
        float r  = std::min (width, height) * 0.5f - 3.f;
        if (r < 5.f) return;

        // outer purple glow -very subtle slow pulse
        {
            float t    = (float) (juce::Time::getMillisecondCounterHiRes() * 0.001);
            float glow = 0.19f + 0.03f * std::sin (t * 0.32f + sliderPos * 5.1f);
            g.setColour (juce::Colour (0xff6633cc).withAlpha (glow));
            g.fillEllipse (cx - r - 4.f, cy - r - 4.f, (r + 4.f) * 2.f, (r + 4.f) * 2.f);
        }

        // gold frame
        juce::ColourGradient frameGrad (
            juce::Colour (0xffFFE566), cx, cy - r,
            juce::Colour (0xff7A4E00), cx, cy + r, false);
        frameGrad.addColour (0.30, juce::Colour (0xffFFD700));
        frameGrad.addColour (0.70, juce::Colour (0xffCE9E22));
        g.setGradientFill (frameGrad);
        g.fillEllipse (cx - r, cy - r, r * 2.f, r * 2.f);

        // dark magical surface
        float sr = r - 5.f;
        juce::ColourGradient surf (
            juce::Colour (0xff1a0a38), cx - sr * 0.3f, cy - sr,
            juce::Colour (0xff091525), cx, cy + sr, false);
        surf.addColour (0.45, juce::Colour (0xff23104a));
        g.setGradientFill (surf);
        g.fillEllipse (cx - sr, cy - sr, sr * 2.f, sr * 2.f);

        // shimmer highlight -very slowly breathes and drifts
        {
            float t  = (float) (juce::Time::getMillisecondCounterHiRes() * 0.001);
            float s1 = std::sin (t * 0.31f);          // primary breath
            float s2 = std::sin (t * 0.19f + 0.8f);  // secondary drift, different rate

            // Main highlight: alpha breathes 0.04–0.07, x position drifts slightly
            float a1  = 0.055f + 0.018f * s1;
            float ox1 = sr * 0.03f * s2;
            g.setColour (juce::Colours::white.withAlpha (a1));
            g.fillEllipse (cx - sr * 0.18f + ox1, cy - sr * 0.92f, sr * 0.44f, sr * 1.3f);

            // Tiny secondary glint that slides a little further, nearly invisible
            float a2  = 0.018f + 0.010f * s2;
            float ox2 = sr * 0.10f * s1;
            g.setColour (juce::Colours::white.withAlpha (a2));
            g.fillEllipse (cx - sr * 0.08f + ox2, cy - sr * 0.78f, sr * 0.20f, sr * 0.55f);

        }

        // inner bevel
        g.setColour (juce::Colour (0xff2a1800).withAlpha (0.4f));
        g.drawEllipse (cx - sr, cy - sr, sr * 2.f, sr * 2.f, 1.5f);

        // gem dots around frame
        g.setColour (juce::Colour (0xff8A5E00));
        g.drawEllipse (cx - r + 0.8f, cy - r + 0.8f, (r - 0.8f) * 2.f, (r - 0.8f) * 2.f, 0.8f);
        for (int i = 0; i < 8; ++i)
        {
            float a  = i * twoPi / 8.f - halfPi;
            float px = cx + std::cos (a) * (r - 3.f);
            float py = cy + std::sin (a) * (r - 3.f);
            bool  big = (i % 2 == 0);
            g.setColour (big ? juce::Colour (0xffFFEE44) : juce::Colour (0xffAA8800));
            float dr = big ? 2.2f : 1.5f;
            g.fillEllipse (px - dr, py - dr, dr * 2.f, dr * 2.f);
        }

        // track arc (dim)
        float arcR = sr - 2.5f;
        {
            juce::Path track;
            track.addArc (cx - arcR, cy - arcR, arcR * 2.f, arcR * 2.f,
                          startAngle, endAngle, true);
            g.setColour (juce::Colour (0xff4433aa).withAlpha (0.30f));
            g.strokePath (track, juce::PathStrokeType (1.5f, juce::PathStrokeType::curved,
                                                        juce::PathStrokeType::rounded));
        }

        // value arc -glow
        float curAngle = startAngle + (endAngle - startAngle) * sliderPos;
        if (sliderPos > 0.001f)
        {
            juce::Path val;
            val.addArc (cx - arcR, cy - arcR, arcR * 2.f, arcR * 2.f,
                        startAngle, curAngle, true);
            // soft halo
            g.setColour (juce::Colour (0xffaaddff).withAlpha (0.30f));
            g.strokePath (val, juce::PathStrokeType (3.8f, juce::PathStrokeType::curved,
                                                      juce::PathStrokeType::rounded));
            // bright core
            g.setColour (juce::Colour (0xff89b4fa).withAlpha (0.85f));
            g.strokePath (val, juce::PathStrokeType (1.8f, juce::PathStrokeType::curved,
                                                      juce::PathStrokeType::rounded));
        }

        // pointer orb
        float dotR = sr * 0.60f;
        float dotX = cx + std::sin (curAngle) * dotR;
        float dotY = cy - std::cos (curAngle) * dotR;
        g.setColour (juce::Colour (0xff89b4fa).withAlpha (0.35f));
        g.fillEllipse (dotX - 5.5f, dotY - 5.5f, 11.f, 11.f);
        g.setColour (juce::Colour (0xffcce8ff).withAlpha (0.95f));
        g.fillEllipse (dotX - 2.8f, dotY - 2.8f, 5.6f, 5.6f);
        g.setColour (juce::Colours::white.withAlpha (0.90f));
        g.fillEllipse (dotX - 1.4f, dotY - 1.4f, 2.8f, 2.8f);

        // Dancing reflection twinkle -sliderPos is naturally different per knob
        // (temp ~0.34, topP ~0.94, length ~0.23, tempo ~0.4) so it seeds unique timing.
        {
            float t  = (float) (juce::Time::getMillisecondCounterHiRes() * 0.001);
            float ph = sliderPos * 7.3f;  // spread the four knobs well apart in phase

            // Beat of two incommensurable rates → irregular on/off timing
            float beat = std::sin (t * 0.73f + ph) * std::sin (t * 1.17f + ph + 0.9f);
            float tw   = std::max (0.f, beat * beat * beat * beat);  // sharp, brief

            if (tw > 0.01f)
            {
                // Position wanders independently in x/y within the existing lit zone
                float wx = cx - sr * 0.15f + sr * 0.10f * std::sin (t * 0.31f + ph);
                float wy = cy - sr * 0.55f + sr * 0.07f * std::sin (t * 0.21f + ph + 1.2f);

                // Pale blue-lavender -same family as the knob's existing shimmer
                auto col = juce::Colour (0xffcce6ff);

                g.setColour (col.withAlpha (tw * 0.10f));
                g.fillEllipse (wx - sr * 0.17f, wy - sr * 0.10f, sr * 0.34f, sr * 0.20f);

                g.setColour (col.withAlpha (tw * 0.30f));
                g.fillEllipse (wx - sr * 0.06f, wy - sr * 0.04f, sr * 0.12f, sr * 0.08f);

                g.setColour (juce::Colours::white.withAlpha (tw * 0.45f));
                float cr = sr * 0.028f;
                g.fillEllipse (wx - cr, wy - cr, cr * 2.f, cr * 2.f);
            }
        }

        // Frozen-state overlay -rendered on top of the knob when disabled
        if (! slider.isEnabled())
        {
            // Dim veil -bg colour at ~66% alpha kills the glow and colour
            g.setColour (juce::Colour (0xa81e1e2e));
            g.fillEllipse (cx - r - 4.f, cy - r - 4.f, (r + 4.f) * 2.f, (r + 4.f) * 2.f);
            // Ice-blue rim signals "frozen", not just "turned off"
            g.setColour (juce::Colour (0x5589b4fa));
            g.drawEllipse (cx - r, cy - r, r * 2.f, r * 2.f, 1.5f);
        }
    }
};

// ── Magical toggle buttons -glowing orb instead of a tick box ───────────────
struct SmallToggleLAF : public juce::LookAndFeel_V4
{
    void drawTickBox (juce::Graphics& g, juce::Component& /*component*/,
                      float x, float y, float w, float h,
                      bool ticked, bool isEnabled,
                      bool /*highlighted*/, bool /*down*/) override
    {
        float cx = x + w * 0.5f, cy = y + h * 0.5f;
        float r  = std::min (w, h) * 0.5f - 0.5f;

        // Outer ring -dim gold, brightens when checked
        g.setColour (juce::Colour (ticked ? 0xffFFD700u : 0xffCE9E22u)
                         .withAlpha (isEnabled ? (ticked ? 0.80f : 0.42f) : 0.20f));
        g.drawEllipse (cx - r, cy - r, r * 2.f, r * 2.f, 1.1f);

        if (ticked)
        {
            // Soft glow fill
            g.setColour (juce::Colour (0xff89b4fa).withAlpha (0.18f));
            g.fillEllipse (cx - r, cy - r, r * 2.f, r * 2.f);
            // Bright inner orb
            float ir = r * 0.62f;
            g.setColour (juce::Colour (0xffaaddff).withAlpha (isEnabled ? 0.90f : 0.40f));
            g.fillEllipse (cx - ir, cy - ir, ir * 2.f, ir * 2.f);
            // White highlight
            float cr = ir * 0.42f;
            g.setColour (juce::Colours::white.withAlpha (0.80f));
            g.fillEllipse (cx - cr, cy - cr, cr * 2.f, cr * 2.f);
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
        auto col = btn.findColour (juce::ToggleButton::textColourId);
        if (! btn.isEnabled()) col = col.withAlpha (0.45f);
        g.setColour (col);
        g.setFont (fontSize);
        g.drawFittedText (btn.getButtonText(),
                          btn.getLocalBounds()
                             .withTrimmedLeft (juce::roundToInt (tickWidth) + 10)
                             .withTrimmedRight (2),
                          juce::Justification::centredLeft, 10);
    }
};

// ── Magical UI LookAndFeel -buttons, combo boxes, popup menus ────────────────
struct MirrorUILAF : public juce::LookAndFeel_V4
{
    MirrorUILAF()
    {
        setColour (juce::TextButton::textColourOffId,        kFg);
        setColour (juce::TextButton::textColourOnId,         juce::Colour (0xffFFEE88));
        setColour (juce::ComboBox::textColourId,             kFg);
        setColour (juce::PopupMenu::textColourId,            kFg);
        setColour (juce::PopupMenu::backgroundColourId,      juce::Colour (0xff130820));
        setColour (juce::PopupMenu::highlightedBackgroundColourId, juce::Colour (0xff6633cc));
    }

    // ── TextButton ────────────────────────────────────────────────────────────
    void drawButtonBackground (juce::Graphics& g, juce::Button& btn,
                                const juce::Colour& bgColour,
                                bool highlighted, bool down) override
    {
        auto b = btn.getLocalBounds().toFloat().reduced (0.5f);
        constexpr float corner = 5.f;

        // Dark magical base gradient
        juce::ColourGradient base (
            juce::Colour (0xff1e1040), b.getCentreX(), b.getY(),
            juce::Colour (0xff0c0818), b.getCentreX(), b.getBottom(), false);
        g.setGradientFill (base);
        g.fillRoundedRectangle (b, corner);

        // Low-alpha tint = accent overlay (e.g., active tab -set via setColour)
        if (bgColour.getAlpha() < 200)
        {
            g.setColour (bgColour);
            g.fillRoundedRectangle (b, corner);
        }

        // Hover / press purple wash
        if (highlighted || down)
        {
            g.setColour (juce::Colour (0xff6633cc).withAlpha (down ? 0.28f : 0.14f));
            g.fillRoundedRectangle (b, corner);
        }

        // Gold border -dims when idle, brightens on hover
        float ba = down ? 1.0f : (highlighted ? 0.80f : 0.40f);
        juce::ColourGradient border (
            juce::Colour (0xffFFE566).withAlpha (ba), b.getCentreX(), b.getY(),
            juce::Colour (0xff9A6B00).withAlpha (ba), b.getCentreX(), b.getBottom(), false);
        border.addColour (0.5, juce::Colour (0xffFFD700).withAlpha (ba));
        g.setGradientFill (border);
        g.drawRoundedRectangle (b, corner, 1.0f);
    }

    void drawButtonText (juce::Graphics& g, juce::TextButton& btn,
                          bool highlighted, bool /*down*/) override
    {
        auto col = btn.findColour (btn.getToggleState() ? juce::TextButton::textColourOnId
                                                        : juce::TextButton::textColourOffId);
        if (! btn.isEnabled()) col = col.withAlpha (0.40f);
        else if (highlighted)  col = col.brighter (0.25f);
        g.setColour (col);
        g.setFont (juce::Font (12.5f));
        g.drawFittedText (btn.getButtonText(),
                          btn.getLocalBounds().reduced (4, 0),
                          juce::Justification::centred, 2);
    }

    // ── ComboBox ──────────────────────────────────────────────────────────────
    void drawComboBox (juce::Graphics& g, int width, int height, bool /*isDown*/,
                        int /*bx*/, int /*by*/, int /*bw*/, int /*bh*/,
                        juce::ComboBox& /*box*/) override
    {
        auto b = juce::Rectangle<float> (0.f, 0.f, (float) width, (float) height).reduced (0.5f);
        constexpr float corner = 4.f;

        juce::ColourGradient base (
            juce::Colour (0xff1e1040), b.getCentreX(), b.getY(),
            juce::Colour (0xff0c0818), b.getCentreX(), b.getBottom(), false);
        g.setGradientFill (base);
        g.fillRoundedRectangle (b, corner);

        juce::ColourGradient border (
            juce::Colour (0xffFFE566).withAlpha (0.50f), b.getCentreX(), b.getY(),
            juce::Colour (0xff9A6B00).withAlpha (0.50f), b.getCentreX(), b.getBottom(), false);
        g.setGradientFill (border);
        g.drawRoundedRectangle (b, corner, 1.0f);

        // Gold chevron arrow
        float ax = (float) width - 14.f, ay = (float) height * 0.5f;
        juce::Path arrow;
        arrow.startNewSubPath (ax - 4.5f, ay - 3.f);
        arrow.lineTo           (ax,        ay + 3.f);
        arrow.lineTo           (ax + 4.5f, ay - 3.f);
        g.setColour (juce::Colour (0xffFFD700).withAlpha (0.85f));
        g.strokePath (arrow, juce::PathStrokeType (1.4f, juce::PathStrokeType::mitered,
                                                    juce::PathStrokeType::square));
    }

    juce::Font getComboBoxFont (juce::ComboBox&) override { return juce::Font (12.f); }

    // ── Popup menu ────────────────────────────────────────────────────────────
    void drawPopupMenuBackground (juce::Graphics& g, int width, int height) override
    {
        auto b = juce::Rectangle<float> (0.f, 0.f, (float) width, (float) height);
        g.setColour (juce::Colour (0xff130820));
        g.fillRoundedRectangle (b, 4.f);
        g.setColour (juce::Colour (0xffFFD700).withAlpha (0.38f));
        g.drawRoundedRectangle (b.reduced (0.5f), 4.f, 1.f);
    }

    void drawPopupMenuItem (juce::Graphics& g, const juce::Rectangle<int>& area,
                             bool isSeparator, bool isActive, bool isHighlighted,
                             bool isTicked, bool /*hasSubMenu*/,
                             const juce::String& text, const juce::String& /*shortcut*/,
                             const juce::Drawable* /*icon*/,
                             const juce::Colour* /*textCol*/) override
    {
        if (isSeparator)
        {
            g.setColour (juce::Colour (0xffFFD700).withAlpha (0.18f));
            g.drawHorizontalLine (area.getCentreY(),
                                  (float) area.getX() + 4.f, (float) area.getRight() - 4.f);
            return;
        }

        if (isHighlighted)
        {
            g.setColour (juce::Colour (0xff6633cc).withAlpha (0.32f));
            g.fillRoundedRectangle (area.toFloat().reduced (2.f, 1.f), 3.f);
            g.setColour (juce::Colour (0xffFFD700).withAlpha (0.18f));
            g.drawRoundedRectangle (area.toFloat().reduced (2.f, 1.f), 3.f, 0.7f);
        }

        auto col = isHighlighted ? juce::Colour (0xffFFEE88) : kFg;
        if (! isActive) col = col.withAlpha (0.40f);
        g.setColour (col);
        g.setFont (juce::Font (12.f));
        g.drawFittedText (text, area.reduced (8, 0), juce::Justification::centredLeft, 1);

        // Gold dot for the ticked (currently selected) item
        if (isTicked)
        {
            float cx = (float) area.getRight() - 12.f;
            float cy = (float) area.getCentreY();
            g.setColour (juce::Colour (0xff89b4fa).withAlpha (0.35f));
            g.fillEllipse (cx - 4.f, cy - 4.f, 8.f, 8.f);
            g.setColour (juce::Colour (0xffFFD700).withAlpha (0.90f));
            g.fillEllipse (cx - 2.5f, cy - 2.5f, 5.f, 5.f);
        }
    }

    juce::Font getPopupMenuFont() override { return juce::Font (12.f); }
};

// ── Key button LookAndFeel -horizontal key, gold on purple glow ─────────────
struct VanityButtonLAF : public juce::LookAndFeel_V4
{
    float phase { 0.0f };

    void drawButtonBackground (juce::Graphics& g, juce::Button& btn,
                               const juce::Colour&, bool isOver, bool isDown) override
    {
        float w = (float) btn.getWidth(), h = (float) btn.getHeight();

        // ── purple glow ───────────────────────────────────────────────────────
        float ga = isDown ? 0.70f : (isOver ? 0.48f : 0.24f);
        g.setColour (juce::Colour (0xff5522bb).withAlpha (ga));
        g.fillEllipse (1.f, 1.f, w - 2.f, h - 2.f);

        // ── shared gold gradient ──────────────────────────────────────────────
        float cx = w * 0.5f, cy = h * 0.5f;
        juce::ColourGradient gold (
            juce::Colour (0xff9A6B00), cx, h * 0.10f,
            juce::Colour (0xffFFE566), cx, h * 0.90f, false);
        gold.addColour (0.30, juce::Colour (0xffFFD700));
        gold.addColour (0.65, juce::Colour (0xffDDA020));
        juce::Colour outline { juce::Colour (0xff6B4800).withAlpha (0.55f) };

        // ══ HAIRBRUSH (left portion, rotated 30°) ════════════════════════════
        float bCx = w * 0.27f, bCy = cy;
        float ang = juce::MathConstants<float>::pi * 0.167f;   // 30°
        auto  rot = juce::AffineTransform::rotation (ang, bCx, bCy);

        float headCy  = bCy - h * 0.110f;
        float headRx  = w * 0.192f;
        float headRy  = h * 0.246f;
        float padRx   = headRx * 0.68f;
        float padRy   = headRy * 0.68f;
        float handleHW = w * 0.052f;
        float handleT  = headCy + headRy * 0.62f;
        float handleB  = bCy + h * 0.380f;

        juce::Path handle;
        handle.addRoundedRectangle (bCx - handleHW, handleT,
                                    handleHW * 2.f, handleB - handleT,
                                    handleHW * 0.75f);
        handle.applyTransform (rot);
        g.setGradientFill (gold);
        g.fillPath (handle);

        juce::Path outer;
        outer.addEllipse (bCx - headRx, headCy - headRy,
                          headRx * 2.f, headRy * 2.f);
        outer.applyTransform (rot);
        g.setGradientFill (gold);
        g.fillPath (outer);

        juce::Path pad;
        pad.addEllipse (bCx - padRx, headCy - padRy,
                        padRx * 2.f, padRy * 2.f);
        pad.applyTransform (rot);
        g.setColour (juce::Colour (0xff0d0820));
        g.fillPath (pad);

        float bLen = h * 0.068f;
        g.setColour (juce::Colour (0xffFFD700).withAlpha (0.90f));
        for (int i = 0; i < 12; ++i)
        {
            float a   = i * juce::MathConstants<float>::twoPi / 12.f;
            float bx0 = bCx    + std::cos (a) * padRx;
            float by0 = headCy + std::sin (a) * padRy;
            float bx1 = bCx    + std::cos (a) * (padRx - bLen);
            float by1 = headCy + std::sin (a) * (padRy - bLen);
            juce::Path bl;
            bl.startNewSubPath (bx0, by0);
            bl.lineTo          (bx1, by1);
            bl.applyTransform  (rot);
            g.strokePath (bl, juce::PathStrokeType (0.60f));
        }

        g.setColour (outline);
        g.strokePath (handle, juce::PathStrokeType (0.70f));
        g.strokePath (outer,  juce::PathStrokeType (0.80f));
        juce::Path padOut;
        padOut.addEllipse (bCx - padRx, headCy - padRy, padRx * 2.f, padRy * 2.f);
        padOut.applyTransform (rot);
        g.setColour (outline.withAlpha (0.30f));
        g.strokePath (padOut, juce::PathStrokeType (0.50f));

        float gleam = 0.38f + 0.22f * std::sin (phase * 1.5f);
        juce::Path arc;
        arc.addArc (bCx - headRx * 0.76f, headCy - headRy * 0.76f,
                    headRx * 1.52f,        headRy * 1.52f,
                    -juce::MathConstants<float>::pi * 0.85f,
                    -juce::MathConstants<float>::pi * 0.10f, true);
        arc.applyTransform (rot);
        g.setColour (juce::Colours::white.withAlpha (gleam));
        g.strokePath (arc, juce::PathStrokeType (1.0f));

        // ══ COMB (right portion, -15° tilt) ══════════════════════════════════
        // Spine (backbone) on the right, prong teeth extending left,
        // handle tapering below with a rounded end -like the reference.
        float cCx  = w * 0.810f, cCy = cy;
        float cAng = -juce::MathConstants<float>::pi / 12.f;   // -15°
        // Flip horizontally around cCx so teeth point right (away from brush), then rotate
        auto  cRot = juce::AffineTransform::scale (-1.f, 1.f, cCx, cCy)
                         .followedBy (juce::AffineTransform::rotation (cAng, cCx, cCy));

        float spineW  = w * 0.110f;                            // backbone width
        float toothL  = w * 0.094f;                            // prong length (extends left)
        float toothH  = h * 0.037f;                            // prong height
        float gapH    = h * 0.022f;                            // gap between prongs
        int   nTeeth  = 8;
        float headH   = nTeeth * toothH + (nTeeth + 1) * gapH; // head section total height
        float handleH = h * 0.175f;                            // handle length
        float handleW = spineW * 0.65f;                        // handle narrower than spine

        float headTop = cCy - (headH + handleH) * 0.5f;
        float headBot = headTop + headH;
        float handleBot = headBot + handleH;

        // Handle: tapers from spine width down to handleW, rounded tip
        juce::Path hndl;
        hndl.startNewSubPath (cCx - spineW * 0.5f, headBot);
        hndl.lineTo          (cCx + spineW * 0.5f, headBot);
        hndl.quadraticTo     (cCx + spineW * 0.5f,  headBot + handleH * 0.32f,
                              cCx + handleW * 0.5f,  headBot + handleH * 0.38f);
        hndl.lineTo          (cCx + handleW * 0.5f,  handleBot - handleW * 0.5f);
        hndl.quadraticTo     (cCx + handleW * 0.5f,  handleBot,
                              cCx,                   handleBot);
        hndl.quadraticTo     (cCx - handleW * 0.5f,  handleBot,
                              cCx - handleW * 0.5f,  handleBot - handleW * 0.5f);
        hndl.lineTo          (cCx - handleW * 0.5f,  headBot + handleH * 0.38f);
        hndl.quadraticTo     (cCx - spineW * 0.5f,   headBot + handleH * 0.32f,
                              cCx - spineW * 0.5f,   headBot);
        hndl.closeSubPath();
        hndl.applyTransform (cRot);
        g.setGradientFill (gold);
        g.fillPath (hndl);
        g.setColour (outline);
        g.strokePath (hndl, juce::PathStrokeType (0.70f));

        // Spine (backbone of head section)
        juce::Path spn;
        spn.addRoundedRectangle (cCx - spineW * 0.5f, headTop,
                                 spineW, headH, spineW * 0.38f);
        spn.applyTransform (cRot);
        g.setGradientFill (gold);
        g.fillPath (spn);

        // Prong teeth -solid filled rectangles with gaps between them
        float ty = headTop + gapH;
        for (int i = 0; i < nTeeth; ++i, ty += toothH + gapH)
        {
            juce::Path tooth;
            tooth.addRoundedRectangle (cCx - spineW * 0.5f - toothL,
                                       ty, toothL, toothH,
                                       toothH * 0.45f);
            tooth.applyTransform (cRot);
            g.setGradientFill (gold);
            g.fillPath (tooth);
            g.setColour (outline.withAlpha (0.38f));
            g.strokePath (tooth, juce::PathStrokeType (0.48f));
        }

        // Spine outline drawn last so it sits cleanly over tooth roots
        g.setColour (outline);
        g.strokePath (spn, juce::PathStrokeType (0.75f));

    }

    void drawButtonText (juce::Graphics&, juce::TextButton&, bool, bool) override {}
};



// ── Advanced Settings Panel ───────────────────────────────────────────────────
// ── Piano roll component ──────────────────────────────────────────────────────
struct PianoRollView : public juce::Component
{
    struct Note { float t, dur; int pitch, vel, inst; float score; };
    const std::vector<Note>* notes { nullptr };
    float threshold { 0.35f };

    static const juce::Colour kInstCols[6];

    void setData (const std::vector<Note>* n, float th)
    {
        notes = n;  threshold = th;  repaint();
    }
    void setThreshold (float th) { threshold = th;  repaint(); }

    void paint (juce::Graphics& g) override
    {
        g.fillAll (juce::Colour (0xff080614));

        if (! notes || notes->empty())
        {
            g.setColour (juce::Colour (0xff4a4a6a));
            g.setFont (11.f);
            g.drawText ("no preview data", getLocalBounds(), juce::Justification::centred);
            return;
        }

        float tMin = 1e9f, tMax = -1e9f;
        int   pMin = 127,  pMax = 0;
        for (auto& n : *notes)
        {
            tMin = std::min (tMin, n.t);
            tMax = std::max (tMax, n.t + n.dur);
            pMin = std::min (pMin, n.pitch);
            pMax = std::max (pMax, n.pitch);
        }
        pMin = std::max (0,   pMin - 2);
        pMax = std::min (127, pMax + 2);
        int   pRange = std::max (1, pMax - pMin);
        float tRange = std::max (0.001f, tMax - tMin);
        float w = (float) getWidth(), h = (float) getHeight();
        float rowH = h / (float) pRange;

        // faint horizontal grid lines every 12 semitones (one octave)
        g.setColour (juce::Colour (0xffffffff).withAlpha (0.04f));
        for (int p = pMin; p <= pMax; p += 12)
        {
            float y = h - ((float)(p - pMin) + 0.5f) / (float) pRange * h;
            g.drawHorizontalLine ((int) y, 0.f, w);
        }

        for (auto& n : *notes)
        {
            float nx = (n.t - tMin) / tRange * w;
            float nw = std::max (1.5f, n.dur / tRange * w);
            float ny = h - ((float)(n.pitch - pMin) + 1.f) / (float) pRange * h;
            float nh = std::max (1.5f, rowH * 0.82f);

            bool kept = n.score >= threshold;
            int  ci   = std::clamp (n.inst, 0, 5);

            if (kept)
            {
                float lift = 0.55f + 0.45f * juce::jmin (1.f,
                    (n.score - threshold) / (1.f - threshold + 0.001f));
                g.setColour (kInstCols[ci].withAlpha (lift));
                g.fillRect  (nx, ny, nw, nh);
                // tiny bright top edge
                g.setColour (juce::Colours::white.withAlpha (lift * 0.4f));
                g.fillRect  (nx, ny, nw, 1.f);
            }
            else
            {
                float dim = 0.10f + 0.12f * (n.score / std::max (0.001f, threshold));
                g.setColour (juce::Colour (0xff4b2a6b).withAlpha (dim));
                g.fillRect  (nx, ny, nw, nh);
            }
        }

        // threshold label
        juce::String thr = juce::String ((int) std::round (threshold * 100)) + "% threshold";
        g.setColour  (juce::Colour (0xffFFD700).withAlpha (0.55f));
        g.setFont    (9.5f);
        g.drawText   (thr, 4, 3, 120, 13, juce::Justification::left);
    }
};

const juce::Colour PianoRollView::kInstCols[6] = {
    juce::Colour (0xffcc88ff),   // 0 vox lead -purple
    juce::Colour (0xffaa66dd),   // 1 vox harm -soft purple
    juce::Colour (0xff88dd44),   // 2 guitar -green-gold
    juce::Colour (0xff44aaff),   // 3 other -blue
    juce::Colour (0xffFFAA22),   // 4 bass -warm gold
    juce::Colour (0xff888899),   // 5 drums -gray
};

// ── Reprocess Dialog ──────────────────────────────────────────────────────────
struct ReprocessDialog : public juce::Component
{
    juce::StringArray              files;
    juce::Array<bool>              reprocessFlags;
    std::function<void(juce::StringArray)> onConfirm;
    std::function<void()>          onCancel;

    int  hoverRow     { -1 };
    bool hoverConfirm { false }, hoverCancel { false };
    bool hoverUp      { false }, hoverDown   { false };
    int  scrollOffset { 0 };
    int  dragStartY   { -1 };
    int  dragStartOff { 0 };
    bool wasDrag      { false };

    explicit ReprocessDialog (const juce::StringArray& f) : files (f)
    {
        for (int i = 0; i < f.size(); ++i)
            reprocessFlags.add (false);
        setInterceptsMouseClicks (true, false);
    }

    int maxVisibleRows() const
    {
        int maxRows = (getHeight() - 40 - 78 - 54) / 32;
        return juce::jmax (2, juce::jmin ((int) files.size(), maxRows));
    }
    bool needsScroll() const { return (int) files.size() > maxVisibleRows(); }
    int  maxScroll()   const { return juce::jmax (0, (int) files.size() - maxVisibleRows()); }

    juce::Rectangle<int> panelBounds() const
    {
        int pw = 420, ph = 78 + maxVisibleRows() * 32 + 54;
        return { (getWidth() - pw) / 2, (getHeight() - ph) / 2, pw, ph };
    }
    // List rows live between y=70 and y=70+visible*32 inside the panel
    juce::Rectangle<int> listArea() const
    {
        auto p = panelBounds();
        return { p.getX(), p.getY() + 70, p.getWidth(), maxVisibleRows() * 32 };
    }
    juce::Rectangle<int> toggleBounds (int i) const
    {
        auto p = panelBounds();
        return { p.getRight() - 118, p.getY() + 74 + (i - scrollOffset) * 32, 104, 24 };
    }
    juce::Rectangle<int> upArrowBounds() const
    {
        auto p = panelBounds();
        return { p.getRight() - 22, p.getY() + 70, 16, 16 };
    }
    juce::Rectangle<int> downArrowBounds() const
    {
        auto p  = panelBounds();
        int  la = maxVisibleRows() * 32;
        return { p.getRight() - 22, p.getY() + 70 + la - 16, 16, 16 };
    }
    juce::Rectangle<int> confirmBounds() const
    {
        auto p = panelBounds();
        return { p.getRight() - 112, p.getBottom() - 44, 96, 30 };
    }
    juce::Rectangle<int> cancelBounds() const
    {
        auto p = panelBounds();
        return { p.getX() + 16, p.getBottom() - 44, 96, 30 };
    }

    void scroll (int delta)
    {
        scrollOffset = juce::jlimit (0, maxScroll(), scrollOffset + delta);
        hoverRow = -1;
        repaint();
    }

    void mouseWheelMove (const juce::MouseEvent&, const juce::MouseWheelDetails& wheel) override
    {
        if (needsScroll())
            scroll (wheel.deltaY < 0 ? 1 : -1);
    }

    void paint (juce::Graphics& g) override
    {
        g.fillAll (juce::Colour (0xcc060412));

        auto panel = panelBounds().toFloat();
        g.setColour (juce::Colour (0xff130a24));
        g.fillRoundedRectangle (panel, 12.f);
        g.setColour (juce::Colour (0xff5522bb).withAlpha (0.75f));
        g.drawRoundedRectangle (panel, 12.f, 1.5f);

        // Title
        g.setColour (juce::Colour (0xffFFD700));
        g.setFont (juce::Font (14.f, juce::Font::bold));
        g.drawText ("ALREADY PROCESSED", panelBounds().withHeight (44),
                    juce::Justification::centred);

        // Subtitle
        g.setColour (juce::Colours::white.withAlpha (0.45f));
        g.setFont (juce::Font (10.5f));
        g.drawText ("Gold = will be reprocessed   |   Dark = use existing",
                    panelBounds().withTrimmedTop (40).withHeight (24),
                    juce::Justification::centred);

        // Divider
        g.setColour (juce::Colour (0xff5522bb).withAlpha (0.28f));
        g.drawLine ((float) panelBounds().getX() + 18, (float) panelBounds().getY() + 68,
                    (float) panelBounds().getRight() - 18, (float) panelBounds().getY() + 68, 1.f);

        // Rows
        int visible = maxVisibleRows();
        int end     = juce::jmin ((int) files.size(), scrollOffset + visible);
        for (int i = scrollOffset; i < end; ++i)
        {
            bool rep  = reprocessFlags[i];
            int  visI = i - scrollOffset;

            if (hoverRow == i)
            {
                g.setColour (juce::Colour (0xff1e1040));
                g.fillRoundedRectangle ((float) panelBounds().getX() + 10,
                                        (float) panelBounds().getY() + 70 + visI * 32,
                                        (float) panelBounds().getWidth() - 20, 28.f, 5.f);
            }

            g.setColour (juce::Colours::white.withAlpha (rep ? 1.0f : 0.72f));
            g.setFont (juce::Font (10.5f));
            g.drawText (files[i],
                        panelBounds().getX() + 18,
                        panelBounds().getY() + 74 + visI * 32,
                        panelBounds().getWidth() - 142, 22,
                        juce::Justification::centredLeft, true);

            auto tb = toggleBounds (i).toFloat();
            g.setColour (rep ? juce::Colour (0xffFFD700) : juce::Colour (0xff1a1040));
            g.fillRoundedRectangle (tb, 5.f);
            g.setColour (rep ? juce::Colour (0xffAA8800).withAlpha (0.7f)
                             : juce::Colour (0xff5522bb).withAlpha (0.45f));
            g.drawRoundedRectangle (tb, 5.f, 1.f);
            g.setColour (rep ? juce::Colour (0xff0d0820) : juce::Colours::white.withAlpha (0.50f));
            g.setFont (juce::Font (9.f, juce::Font::bold));
            g.drawText (rep ? "REPROCESS" : "USE EXISTING",
                        toggleBounds (i), juce::Justification::centred);
        }

        // Up / down arrows + scrollbar (shown when list overflows)
        if (needsScroll())
        {
            // Scrollbar track
            auto  p   = panelBounds();
            int   sbX = p.getRight() - 10;
            int   sbY = p.getY() + 70;
            int   sbH = visible * 32;
            float ratio = (float) visible / (float) files.size();
            float posF  = maxScroll() > 0 ? (float) scrollOffset / (float) maxScroll() : 0.f;
            int   tH    = juce::jmax (16, (int) (sbH * ratio));
            int   tY    = sbY + (int) ((sbH - tH) * posF);

            g.setColour (juce::Colour (0xff5522bb).withAlpha (0.18f));
            g.fillRoundedRectangle ((float) sbX, (float) sbY, 4.f, (float) sbH, 2.f);
            g.setColour (juce::Colour (0xff5522bb).withAlpha (0.65f));
            g.fillRoundedRectangle ((float) sbX, (float) tY, 4.f, (float) tH, 2.f);

            // Up arrow
            auto drawArrow = [&] (juce::Rectangle<int> r, bool pointUp, bool hover, bool enabled)
            {
                float alpha = enabled ? (hover ? 0.95f : 0.55f) : 0.18f;
                g.setColour (juce::Colour (0xff5522bb).withAlpha (alpha * 0.5f));
                g.fillRoundedRectangle (r.toFloat(), 3.f);
                g.setColour (juce::Colour (0xffFFD700).withAlpha (alpha));
                juce::Path tri;
                float cx = r.getCentreX(), cy = r.getCentreY();
                float hw = 4.f, hh = 3.f;
                if (pointUp)
                {
                    tri.startNewSubPath (cx, cy - hh);
                    tri.lineTo (cx - hw, cy + hh);
                    tri.lineTo (cx + hw, cy + hh);
                }
                else
                {
                    tri.startNewSubPath (cx, cy + hh);
                    tri.lineTo (cx - hw, cy - hh);
                    tri.lineTo (cx + hw, cy - hh);
                }
                tri.closeSubPath();
                g.fillPath (tri);
            };
            drawArrow (upArrowBounds(),   true,  hoverUp,   scrollOffset > 0);
            drawArrow (downArrowBounds(), false, hoverDown, scrollOffset < maxScroll());
        }

        // Cancel button
        {
            auto cb = cancelBounds().toFloat();
            g.setColour (juce::Colour (0xff1e1040).withAlpha (hoverCancel ? 0.95f : 0.75f));
            g.fillRoundedRectangle (cb, 6.f);
            g.setColour (juce::Colour (0xff5522bb).withAlpha (hoverCancel ? 0.8f : 0.45f));
            g.drawRoundedRectangle (cb, 6.f, 1.f);
            g.setColour (juce::Colours::white.withAlpha (hoverCancel ? 0.85f : 0.55f));
            g.setFont (juce::Font (10.f, juce::Font::bold));
            g.drawText ("CANCEL", cancelBounds(), juce::Justification::centred);
        }

        // Confirm button
        {
            auto cb = confirmBounds().toFloat();
            g.setColour (juce::Colour (0xff5522bb).withAlpha (hoverConfirm ? 1.0f : 0.80f));
            g.fillRoundedRectangle (cb, 6.f);
            g.setColour (juce::Colour (0xffFFD700).withAlpha (hoverConfirm ? 0.9f : 0.55f));
            g.drawRoundedRectangle (cb, 6.f, 1.f);
            g.setColour (juce::Colour (0xffFFD700));
            g.setFont (juce::Font (10.f, juce::Font::bold));
            g.drawText ("CONFIRM", confirmBounds(), juce::Justification::centred);
        }
    }

    void resized() override {}

    void mouseMove (const juce::MouseEvent& e) override
    {
        int visible = maxVisibleRows();
        int end     = juce::jmin ((int) files.size(), scrollOffset + visible);
        int nr = -1;
        for (int i = scrollOffset; i < end; ++i)
            if (toggleBounds (i).contains (e.getPosition()))
                nr = i;
        bool nc = confirmBounds().contains  (e.getPosition());
        bool nx = cancelBounds().contains   (e.getPosition());
        bool nu = upArrowBounds().contains  (e.getPosition());
        bool nd = downArrowBounds().contains (e.getPosition());
        if (nr != hoverRow || nc != hoverConfirm || nx != hoverCancel
            || nu != hoverUp || nd != hoverDown)
        {
            hoverRow = nr; hoverConfirm = nc; hoverCancel = nx;
            hoverUp = nu; hoverDown = nd;
            repaint();
        }
    }

    void mouseExit (const juce::MouseEvent&) override
    {
        hoverRow = -1; hoverConfirm = hoverCancel = hoverUp = hoverDown = false;
        dragStartY = -1;
        repaint();
    }

    void mouseDown (const juce::MouseEvent& e) override
    {
        wasDrag = false;

        // Arrow buttons scroll immediately
        if (needsScroll())
        {
            if (upArrowBounds().contains (e.getPosition()))   { scroll (-1); return; }
            if (downArrowBounds().contains (e.getPosition())) { scroll ( 1); return; }
        }

        // Confirm / cancel
        if (confirmBounds().contains (e.getPosition()) ||
            cancelBounds().contains  (e.getPosition()))
            return;  // handled in mouseUp so we can distinguish from drag

        // Start drag-to-scroll if inside list area
        if (listArea().contains (e.getPosition()))
        {
            dragStartY   = e.getPosition().y;
            dragStartOff = scrollOffset;
        }
    }

    void mouseDrag (const juce::MouseEvent& e) override
    {
        if (! needsScroll() || dragStartY < 0) return;
        int delta = dragStartY - e.getPosition().y;
        if (std::abs (delta) > 4) wasDrag = true;
        scrollOffset = juce::jlimit (0, maxScroll(), dragStartOff + delta / 32);
        repaint();
    }

    void mouseUp (const juce::MouseEvent& e) override
    {
        dragStartY = -1;
        if (wasDrag) { wasDrag = false; return; }

        // Toggle row
        int visible = maxVisibleRows();
        int end     = juce::jmin ((int) files.size(), scrollOffset + visible);
        for (int i = scrollOffset; i < end; ++i)
        {
            if (toggleBounds (i).contains (e.getPosition()))
            {
                reprocessFlags.set (i, ! reprocessFlags[i]);
                repaint();
                return;
            }
        }

        if (confirmBounds().contains (e.getPosition()))
        {
            juce::StringArray toSkip;
            for (int i = 0; i < files.size(); ++i)
                if (! reprocessFlags[i])
                    toSkip.add (files[i]);
            auto cb = onConfirm;
            juce::MessageManager::callAsync ([this] {
                if (auto* p = getParentComponent()) p->removeChildComponent (this);
                delete this;
            });
            if (cb) cb (toSkip);
            return;
        }

        if (cancelBounds().contains (e.getPosition()))
        {
            auto cb = onCancel;
            juce::MessageManager::callAsync ([this] {
                if (auto* p = getParentComponent()) p->removeChildComponent (this);
                delete this;
            });
            if (cb) cb();
        }
    }
};


// ── Advanced Settings Panel ───────────────────────────────────────────────────
struct AdvancedPanel : public juce::Component
{
    AIMusicProcessor& proc;

    // ── existing controls ─────────────────────────────────────────────────────
    juce::Label        lblDisc          { {}, "Note Filter" };
    juce::ToggleButton chkDisc          { "Enable" };
    juce::Label        lblIntensity     { {}, "Intensity" };
    juce::Slider       sldIntensity;
    juce::Label        lblIntensityHint { {}, "low = gentle,  high = strict" };

    juce::Label        lblSeq           { {}, "Seq Length (training)" };
    juce::Slider       sldSeqLen;
    juce::Label        lblSeqHint       { {}, "tokens per training window" };

    // ── fine-tune from checkpoint ─────────────────────────────────────────────
    juce::Label        lblFineTune      { {}, "Fine-tune" };
    juce::ToggleButton chkFineTune      { "From checkpoint" };
    juce::Label        lblBaseCkpt      { {}, "Base model" };
    juce::TextEditor   edtBaseCkpt;
    juce::TextButton   btnBrowseBase    { "..." };

    // ── piano roll preview ────────────────────────────────────────────────────
    juce::Label        lblPreview       { {}, "Filter Preview" };
    juce::ComboBox     cmbSong;
    juce::TextButton   btnLoad          { "Load" };
    juce::Label        lblStatus        { {}, "Enable filter above, then re-process to unlock" };
    PianoRollView      pianoRoll;

    struct SongData { juce::String name; std::vector<PianoRollView::Note> notes; };
    std::vector<SongData>            songs;
    std::shared_ptr<std::atomic<bool>> alive { std::make_shared<std::atomic<bool>> (true) };

    static const juce::Colour kBg2;
    static const juce::Colour kFg2;
    static const juce::Colour kAcc2;

    explicit AdvancedPanel (AIMusicProcessor& p) : proc (p)
    {
        setSize (400, 550);

        auto styleLabel = [&] (juce::Label& l, bool small = false) {
            l.setColour (juce::Label::textColourId, kFg2);
            if (small) l.setFont (juce::Font (10.5f));
            addAndMakeVisible (l);
        };
        auto styleSlider = [&] (juce::Slider& s) {
            s.setSliderStyle (juce::Slider::LinearHorizontal);
            s.setTextBoxStyle (juce::Slider::TextBoxRight, false, 48, 18);
            s.setColour (juce::Slider::textBoxTextColourId,       kFg2);
            s.setColour (juce::Slider::thumbColourId,             kAcc2);
            s.setColour (juce::Slider::trackColourId,             kAcc2.withAlpha (0.4f));
            s.setColour (juce::Slider::textBoxBackgroundColourId, juce::Colour (0xff313244));
            s.setColour (juce::Slider::textBoxOutlineColourId,    juce::Colours::transparentBlack);
            addAndMakeVisible (s);
        };

        // Note filter
        styleLabel (lblDisc);
        chkDisc.setToggleState (proc.discIntensity > 0.0f, juce::dontSendNotification);
        chkDisc.setColour (juce::ToggleButton::textColourId, kFg2);
        chkDisc.onStateChange = [this] {
            bool on = chkDisc.getToggleState();
            sldIntensity.setEnabled (on);
            proc.discIntensity = on ? (float) sldIntensity.getValue() : 0.0f;
            updatePianoRoll();
        };
        addAndMakeVisible (chkDisc);

        styleLabel (lblIntensity);
        sldIntensity.setRange (0.01, 1.0, 0.01);
        sldIntensity.setValue (proc.discIntensity > 0.0f ? (double) proc.discIntensity : 0.25,
                               juce::dontSendNotification);
        sldIntensity.setEnabled (proc.discIntensity > 0.0f);
        sldIntensity.onValueChange = [this] {
            if (chkDisc.getToggleState())
                proc.discIntensity = (float) sldIntensity.getValue();
            updatePianoRoll();
        };
        styleSlider (sldIntensity);
        styleLabel (lblIntensityHint, true);

        // Seq length
        styleLabel (lblSeq);
        sldSeqLen.setRange (128, 1024, 128);
        sldSeqLen.setValue (proc.seqLen, juce::dontSendNotification);
        sldSeqLen.onValueChange = [this] { proc.seqLen = (int) sldSeqLen.getValue(); };
        styleSlider (sldSeqLen);
        styleLabel (lblSeqHint, true);

        // Fine-tune from checkpoint
        styleLabel (lblFineTune);
        // Always default to the Lakh base model — not saved to DAW state so this runs fresh each session
        if (proc.repoRoot.exists())
        {
            auto candidate = proc.repoRoot.getChildFile ("runs/checkpoints/es_model.pt");
            if (candidate.existsAsFile())
                proc.pretrainCkpt = candidate.getFullPathName();
        }
        // Lock seq_len to match the base checkpoint; unlock when fine-tune is off
        auto applyFineTuneLock = [this] {
            bool on = chkFineTune.getToggleState();
            proc.pretrainCkpt = on ? edtBaseCkpt.getText().trim() : juce::String{};
            edtBaseCkpt .setEnabled (on);
            btnBrowseBase.setEnabled (on);
            if (on && proc.pretrainCkpt.isNotEmpty())
            {
                int ckptSeq = proc.fetchSeqLenForCkpt (proc.pretrainCkpt);
                if (ckptSeq > 0)
                {
                    sldSeqLen.setValue (ckptSeq, juce::sendNotification);
                    sldSeqLen.setEnabled (false);
                    sldSeqLen.setTooltip ("Locked to " + juce::String (ckptSeq)
                                          + " - must match the base checkpoint's training length.");
                    return;
                }
            }
            sldSeqLen.setEnabled (true);
            sldSeqLen.setTooltip ("Number of tokens per training window. "
                                  "512 works well for most datasets; use 1024 for longer musical phrases.");
        };

        chkFineTune.setToggleState (proc.pretrainCkpt.isNotEmpty(), juce::dontSendNotification);
        chkFineTune.setColour (juce::ToggleButton::textColourId, kFg2);
        chkFineTune.onStateChange = [applyFineTuneLock] { applyFineTuneLock(); };
        addAndMakeVisible (chkFineTune);

        styleLabel (lblBaseCkpt);
        edtBaseCkpt.setText (proc.pretrainCkpt, false);
        edtBaseCkpt.setMultiLine (false);
        edtBaseCkpt.setReturnKeyStartsNewLine (false);
        edtBaseCkpt.setSelectAllWhenFocused (true);
        edtBaseCkpt.setEnabled (proc.pretrainCkpt.isNotEmpty());
        edtBaseCkpt.setColour (juce::TextEditor::backgroundColourId,    juce::Colour (0xff252535));
        edtBaseCkpt.setColour (juce::TextEditor::textColourId,          kFg2);
        edtBaseCkpt.setColour (juce::TextEditor::outlineColourId,       kAcc2.withAlpha (0.35f));
        edtBaseCkpt.setColour (juce::TextEditor::focusedOutlineColourId, kAcc2.withAlpha (0.75f));
        edtBaseCkpt.onTextChange = [this, applyFineTuneLock] {
            if (chkFineTune.getToggleState()) applyFineTuneLock();
        };
        addAndMakeVisible (edtBaseCkpt);

        // Apply lock immediately if fine-tune is already on at open time
        applyFineTuneLock();

        btnBrowseBase.setEnabled (proc.pretrainCkpt.isNotEmpty());
        btnBrowseBase.onClick = [this] {
            auto chooser = std::make_shared<juce::FileChooser> (
                "Select base checkpoint (.pt)", juce::File{}, "*.pt");
            chooser->launchAsync (
                juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                [this, chooser] (const juce::FileChooser& fc) {
                    auto result = fc.getResult();
                    if (result.existsAsFile())
                    {
                        edtBaseCkpt.setText (result.getFullPathName(), false);
                        proc.pretrainCkpt = result.getFullPathName();
                    }
                });
        };
        addAndMakeVisible (btnBrowseBase);

        // Preview section
        styleLabel (lblPreview);

        cmbSong.setColour (juce::ComboBox::backgroundColourId,  juce::Colour (0xff313244));
        cmbSong.setColour (juce::ComboBox::textColourId,         kFg2);
        cmbSong.setColour (juce::ComboBox::outlineColourId,      kAcc2.withAlpha (0.3f));
        cmbSong.onChange = [this] {
            int idx = cmbSong.getSelectedId() - 1;
            if (idx >= 0 && idx < (int) songs.size())
                updatePianoRoll (idx);
        };
        addAndMakeVisible (cmbSong);

        addAndMakeVisible (btnLoad);
        btnLoad.onClick = [this] { loadPreview(); };

        styleLabel (lblStatus, true);

        addAndMakeVisible (pianoRoll);

        // Tooltips
        chkDisc      .setTooltip ("Enable the AI note filter, which uses a trained discriminator to remove "
                                  "low-quality or atypical notes from your training data before training.");
        sldIntensity .setTooltip ("How aggressively to filter notes. "
                                  "Low (~0.05) removes only the worst 5%; high (~1.0) removes up to 50%. "
                                  "Start low and preview the result before reprocessing.");
        sldSeqLen    .setTooltip ("Number of tokens per training window. "
                                  "Longer sequences (512–1024) capture more musical context but require more memory and are slower to train.");
        chkFineTune  .setTooltip ("Start training from an existing checkpoint rather than from scratch. "
                                  "Useful for adapting a previously trained model to new material.");
        edtBaseCkpt  .setTooltip ("Path to the base checkpoint (.pt file) to fine-tune from.");
        btnBrowseBase.setTooltip ("Browse for a base model checkpoint to fine-tune from.");
        cmbSong      .setTooltip ("Select a song to preview how the note filter affects its notes.");
        btnLoad      .setTooltip ("Load the filter preview for the selected song. "
                                  "Run Process Audio with the filter enabled first to unlock this.");
    }

    ~AdvancedPanel() override { alive->store (false); }

    void loadPreview()
    {
        btnLoad.setEnabled (false);
        lblStatus.setText ("loading...", juce::dontSendNotification);
        auto alivePtr = alive;
        juce::Thread::launch ([this, alivePtr] {
            auto evDir   = proc.fetchLatestEvents();
            auto jsonStr = proc.fetchDiscPreview (evDir);
            juce::MessageManager::callAsync ([this, alivePtr, jsonStr] {
                if (! alivePtr->load()) return;
                btnLoad.setEnabled (true);
                parsePreview (jsonStr);
            });
        });
    }

    void parsePreview (const juce::String& jsonStr)
    {
        if (jsonStr.isEmpty() || jsonStr.startsWith ("{\"detail\""))
        {
            lblStatus.setText ("No data -process with filter enabled first",
                               juce::dontSendNotification);
            return;
        }
        auto json  = juce::JSON::parse (jsonStr);
        auto* root = json.getDynamicObject();
        if (! root) { lblStatus.setText ("parse error", juce::dontSendNotification); return; }

        auto songsVar = root->getProperty ("songs");
        auto* arr     = songsVar.getArray();
        if (! arr || arr->isEmpty())
        {
            lblStatus.setText ("no songs in preview data", juce::dontSendNotification);
            return;
        }

        songs.clear();
        cmbSong.clear (juce::dontSendNotification);

        for (int si = 0; si < arr->size(); ++si)
        {
            if (auto* sobj = (*arr)[si].getDynamicObject())
            {
                SongData sd;
                sd.name   = sobj->getProperty ("name").toString();
                auto* narr = sobj->getProperty ("notes").getArray();
                if (narr)
                {
                    for (auto& nv : *narr)
                    {
                        if (auto* no = nv.getDynamicObject())
                        {
                            PianoRollView::Note nd;
                            nd.t     = (float)(double) no->getProperty ("t");
                            nd.dur   = (float)(double) no->getProperty ("dur");
                            nd.pitch = (int)            no->getProperty ("p");
                            nd.vel   = (int)            no->getProperty ("v");
                            nd.inst  = (int)            no->getProperty ("inst");
                            nd.score = (float)(double) no->getProperty ("score");
                            sd.notes.push_back (nd);
                        }
                    }
                }
                cmbSong.addItem (sd.name.isEmpty() ? ("Song " + juce::String (si + 1)) : sd.name,
                                 si + 1);
                songs.push_back (std::move (sd));
            }
        }

        if (! songs.empty())
        {
            cmbSong.setSelectedId (1, juce::dontSendNotification);
            updatePianoRoll (0);
            lblStatus.setText ({}, juce::dontSendNotification);
        }
    }

    void updatePianoRoll (int songIdx = -1)
    {
        if (songIdx >= 0) cmbSong.setSelectedId (songIdx + 1, juce::dontSendNotification);
        int idx = cmbSong.getSelectedId() - 1;
        if (idx < 0 || idx >= (int) songs.size()) return;
        float intensity = (float) sldIntensity.getValue();
        float thresh    = 0.10f + intensity * 0.45f;
        pianoRoll.setData (&songs[idx].notes, thresh);
    }

    void paint (juce::Graphics& g) override
    {
        g.fillAll (kBg2);
        float sepY = (float) getHeight() * 0.40f;
        g.setColour (kFg2.withAlpha (0.12f));
        g.drawHorizontalLine ((int) sepY, 12.f, (float) getWidth() - 12.f);
        // piano roll border
        g.setColour (kAcc2.withAlpha (0.18f));
        g.drawRect (pianoRoll.getBounds().expanded (1), 1);
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced (14, 12);
        int rowH  = 22, gap = 4;

        // ── Note Filter ───────────────────────────────────────────────────────
        auto discRow = area.removeFromTop (rowH);
        lblDisc .setBounds (discRow.removeFromLeft (90));
        chkDisc .setBounds (discRow.removeFromLeft (70));
        area.removeFromTop (gap);
        auto intRow = area.removeFromTop (rowH);
        lblIntensity.setBounds (intRow.removeFromLeft (68));
        intRow.removeFromLeft (4);
        sldIntensity.setBounds (intRow);
        area.removeFromTop (2);
        lblIntensityHint.setBounds (area.removeFromTop (14));

        area.removeFromTop (10);

        // ── Seq Length ────────────────────────────────────────────────────────
        lblSeq.setBounds (area.removeFromTop (rowH));
        area.removeFromTop (gap);
        sldSeqLen.setBounds (area.removeFromTop (rowH));
        area.removeFromTop (2);
        lblSeqHint.setBounds (area.removeFromTop (14));

        area.removeFromTop (12);

        // ── Fine-tune from checkpoint ─────────────────────────────────────────
        auto ftRow = area.removeFromTop (rowH);
        lblFineTune.setBounds (ftRow.removeFromLeft (72));
        chkFineTune.setBounds (ftRow);
        area.removeFromTop (gap);
        auto baseRow = area.removeFromTop (rowH);
        lblBaseCkpt  .setBounds (baseRow.removeFromLeft (72));
        baseRow.removeFromLeft (4);
        btnBrowseBase.setBounds (baseRow.removeFromRight (28));
        baseRow.removeFromRight (4);
        edtBaseCkpt  .setBounds (baseRow);

        area.removeFromTop (12);

        // ── Preview header ────────────────────────────────────────────────────
        auto hdr = area.removeFromTop (rowH);
        lblPreview.setBounds (hdr.removeFromLeft (90));
        btnLoad   .setBounds (hdr.removeFromRight (50));
        hdr.removeFromRight (4);
        cmbSong   .setBounds (hdr);

        area.removeFromTop (4);
        lblStatus .setBounds (area.removeFromTop (14));
        area.removeFromTop (6);

        // ── Piano roll ────────────────────────────────────────────────────────
        pianoRoll.setBounds (area);
    }
};

const juce::Colour AdvancedPanel::kBg2  { 0xff1e1e2e };
const juce::Colour AdvancedPanel::kFg2  { 0xffcdd6f4 };
const juce::Colour AdvancedPanel::kAcc2 { 0xff89b4fa };

AIMusicEditor::AIMusicEditor (AIMusicProcessor& p)
    : AudioProcessorEditor (&p), proc (p),
      mirrorAnim     (std::make_unique<MirrorMirror>()),
      mirrorUILAF       (std::make_unique<MirrorUILAF>()),
      smallToggleLAF    (std::make_unique<SmallToggleLAF>()),
      mirrorKnobLAF     (std::make_unique<MirrorKnobLAF>()),
      keyButtonLAF      (std::make_unique<VanityButtonLAF>())
{
    setSize (480, 440);
    setLookAndFeel (mirrorUILAF.get());   // global -cascades to all children without explicit LAF
    addAndMakeVisible (*mirrorAnim);

    // ── Tab bar ───────────────────────────────────────────────────────────────
    auto styleTab = [&] (juce::TextButton& btn) {
        addAndMakeVisible (btn);
    };
    styleTab (tabProcess);
    styleTab (tabGenerate);
    tabProcess .onClick = [this] { currentTab = 0; updateTabVisibility(); };
    tabGenerate.onClick = [this] { currentTab = 1; updateTabVisibility(); };

    // ── Project name ──────────────────────────────────────────────────────────
    lblProjectName.setText ("Project", juce::dontSendNotification);
    lblProjectName.setColour (juce::Label::textColourId, kFg.withAlpha (0.70f));
    lblProjectName.setJustificationType (juce::Justification::centredRight);
    addAndMakeVisible (lblProjectName);

    edtProjectName.setText (proc.projectName, false);
    edtProjectName.setMultiLine (false);
    edtProjectName.setReturnKeyStartsNewLine (false);
    edtProjectName.setSelectAllWhenFocused (true);
    edtProjectName.setColour (juce::TextEditor::backgroundColourId,  juce::Colour (0xff252535));
    edtProjectName.setColour (juce::TextEditor::textColourId,         kFg);
    edtProjectName.setColour (juce::TextEditor::outlineColourId,      kAcc.withAlpha (0.35f));
    edtProjectName.setColour (juce::TextEditor::focusedOutlineColourId, kAcc.withAlpha (0.75f));
    edtProjectName.onTextChange = [this] {
        proc.projectName = edtProjectName.getText().trim();
    };
    addAndMakeVisible (edtProjectName);

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

    btnRunProcess.onClick = [this] {
        if (proc.audioFolder.isEmpty()) { browseFolder (true); return; }
        proc.selectedTracks = buildTracksString();

        auto existing = proc.fetchExistingProcessed();
        if (existing.isEmpty())
        {
            proc.startProcess (proc.audioFolder, {});
            return;
        }

        auto* dlg = new ReprocessDialog (existing);
        dlg->setBounds (getLocalBounds());
        dlg->onConfirm = [this] (juce::StringArray filesToSkip) {
            proc.startProcess (proc.audioFolder, filesToSkip);
        };
        dlg->onCancel = [] {};
        addAndMakeVisible (dlg);
        dlg->toFront (false);
    };
    btnTrain.onClick = [this] { browseEventsAndTrain(); };
    addAndMakeVisible (btnRunProcess);
    addAndMakeVisible (btnTrain);

    // ── Advanced settings button (key icon) ───────────────────────────────────
    btnAdvanced.setLookAndFeel (keyButtonLAF.get());

    btnAdvanced.onClick = [this] {
        juce::DialogWindow::LaunchOptions opts;
        opts.dialogTitle             = "Advanced Settings";
        opts.content.setOwned        (new AdvancedPanel (proc));
        opts.dialogBackgroundColour  = juce::Colour (0xff1e1e2e);
        opts.useNativeTitleBar       = false;
        opts.resizable               = false;
        opts.componentToCentreAround = this;
        opts.launchAsync();
    };
    addAndMakeVisible (btnAdvanced);

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
    addAndMakeVisible (cmbSubdivision);
    makeLabel (lblSubdivision, "Subdiv");
    lblSubdivision.setFont (juce::Font (11.5f));

    btnTriplets.setToggleState (proc.allowTriplets, juce::dontSendNotification);
    btnTriplets.setColour (juce::ToggleButton::textColourId, kFg);
    btnTriplets.setLookAndFeel (smallToggleLAF.get());
    btnTriplets.onStateChange = [this] { proc.allowTriplets = btnTriplets.getToggleState(); };
    addAndMakeVisible (btnTriplets);

    btnQuantize.setToggleState (proc.quantize, juce::dontSendNotification);
    btnQuantize.setColour (juce::ToggleButton::textColourId, kFg);
    btnQuantize.setLookAndFeel (smallToggleLAF.get());
    btnQuantize.onStateChange = [this] {
        proc.quantize = btnQuantize.getToggleState();
        bool q = proc.quantize;
        cmbSubdivision.setEnabled (q);
        btnTriplets   .setEnabled (q);
    };
    cmbSubdivision.setEnabled (proc.quantize);
    btnTriplets   .setEnabled (proc.quantize);
    addAndMakeVisible (btnQuantize);

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

    // ── Preset bar ────────────────────────────────────────────────────────────
    btnSavePreset.onClick = [this] { savePreset(); };
    btnLoadPreset.onClick = [this] { loadPreset(); };
    addAndMakeVisible (btnSavePreset);
    addAndMakeVisible (btnLoadPreset);

    proc.onStateLoaded = [this] { refreshFromProcessor(); };

    // ── Shared ────────────────────────────────────────────────────────────────
    btnCancel.onClick = [this] { localErrorMessage.clear(); proc.cancelJob(); };
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

    btnPreview.setVisible (false);
    btnPreview.onClick = [this] {
        if (proc.isPreviewPlaying())
            proc.stopPreview();
        else if (lastMidiPath.isNotEmpty())
            proc.startPreview (lastMidiPath);
    };
    addAndMakeVisible (btnPreview);

    proc.onPreviewStateChanged = [this] (bool playing) {
        btnPreview.setButtonText (playing ? "Stop" : "Preview");
        repaint();
    };

    // ── Tooltips ──────────────────────────────────────────────────────────────
    // Tab buttons
    tabProcess .setTooltip ("Process your audio files into training data, then train an AI model on them.");
    tabGenerate.setTooltip ("Generate new MIDI using a trained model. Switch here after training finishes.");

    // Project name
    lblProjectName.setTooltip ("A name for this project. Processed data and the trained model are saved "
                                "under this name so you can have multiple projects side by side.");
    edtProjectName.setTooltip ("A name for this project. Processed data and the trained model are saved "
                                "under this name so you can have multiple projects side by side.");

    // Process & Train tab
    btnBrowseFolder.setTooltip ("Choose the folder containing your audio files (.wav/.mp3). "
                                "The plugin will separate each file into stems and convert them to MIDI.");
    chkLeadVox .setTooltip ("Include lead vocals in the training data. Uncheck to leave this instrument out of the model.");
    chkHarmVox .setTooltip ("Include harmony / backing vocals in the training data.");
    chkGuitar  .setTooltip ("Include guitar in the training data.");
    chkBass    .setTooltip ("Include bass in the training data.");
    chkDrums   .setTooltip ("Include drums in the training data.");
    chkOther   .setTooltip ("Include other / miscellaneous instruments in the training data.");
    btnRunProcess.setTooltip ("Separate your audio files into stems, convert to MIDI events, "
                              "and prepare the dataset for training. Run this before hitting Train.");
    btnTrain     .setTooltip ("Train an AI model on the processed MIDI data. "
                              "The model learns the musical style of your audio files.");

    // Generate tab
    btnBrowseCkpt.setTooltip ("Select a trained model file (.pt) to generate from. "
                              "After training, the model is saved in your project folder.");
    lblCkpt      .setTooltip ("The trained model currently selected for generation.");
    lblTemperature.setTooltip ("Controls randomness. Lower values (0.5) are more predictable; "
                               "higher values (1.2+) are more experimental.");
    sldTemperature.setTooltip ("Controls randomness. Lower values (0.5) are more predictable; "
                               "higher values (1.2+) are more experimental.");
    lblTopP.setTooltip ("Nucleus sampling threshold - only the most likely tokens up to this "
                        "cumulative probability are considered. Lower = safer, higher = more varied.");
    sldTopP.setTooltip ("Nucleus sampling threshold - only the most likely tokens up to this "
                        "cumulative probability are considered. Lower = safer, higher = more varied.");
    lblMaxTokens.setTooltip ("Maximum number of musical events to generate. More tokens = longer output.");
    sldMaxTokens.setTooltip ("Maximum number of musical events to generate. More tokens = longer output.");
    lblTempo.setTooltip ("Tempo in BPM for the generated MIDI.");
    sldTempo    .setTooltip ("Tempo in BPM for the generated MIDI.");
    btnSyncTempo.setTooltip ("Lock the tempo to your DAW's current BPM.");
    lblSubdivision.setTooltip ("Grid subdivision used to snap note timings when Quantize is on.");
    cmbSubdivision.setTooltip ("Grid subdivision used to snap note timings when Quantize is on.");
    btnTriplets   .setTooltip ("Allow triplet subdivisions (e.g. 1/8T) in the generated rhythm.");
    btnQuantize   .setTooltip ("Snap all generated notes to the nearest grid subdivision.");
    btnSeedFromData.setTooltip ("Seed generation from a random phrase in your training data "
                                "instead of starting from scratch - tends to stay closer to your style.");
    btnGenerate.setTooltip ("Generate a new MIDI sequence using the selected model and settings.");

    // Title bar / shared
    btnAdvanced.setTooltip ("Advanced settings - note filter, sequence length, fine-tune from a base checkpoint.");
    btnSavePreset.setTooltip ("Save the current settings to a preset file.");
    btnLoadPreset.setTooltip ("Load settings from a previously saved preset file.");
    btnCancel  .setTooltip ("Cancel the currently running job.");
    btnShowMidi.setTooltip ("Show the generated MIDI file in Finder.");
    btnPreview .setTooltip ("Play back the generated MIDI through a basic synth for a quick listen.");

    addMouseListener (&longPressHelper, true);  // touch long-press tooltips

    updateTabVisibility();
    startTimer (1500);
}

// ── LongPressHelper ───────────────────────────────────────────────────────────

juce::String AIMusicEditor::LongPressHelper::findTooltip (juce::Component* c)
{
    while (c != nullptr)
    {
        if (auto* tc = dynamic_cast<juce::TooltipClient*> (c))
        {
            auto t = tc->getTooltip();
            if (t.isNotEmpty()) return t;
        }
        c = c->getParentComponent();
    }
    return {};
}

void AIMusicEditor::LongPressHelper::mouseDown (const juce::MouseEvent& e)
{
    pressedOn = e.eventComponent;
    pressPos  = e.position;
    startTimer (600);
}

void AIMusicEditor::LongPressHelper::mouseUp (const juce::MouseEvent&)
{
    stopTimer();
}

void AIMusicEditor::LongPressHelper::mouseDrag (const juce::MouseEvent& e)
{
    if (e.getDistanceFromDragStart() > 8.f)
        stopTimer();
}

void AIMusicEditor::LongPressHelper::timerCallback()
{
    stopTimer();
    if (pressedOn == nullptr) return;

    auto text = findTooltip (pressedOn);
    if (text.isEmpty()) return;

    auto* bubble = new juce::BubbleMessageComponent();
    owner.addAndMakeVisible (bubble);
    bubble->setColour (juce::BubbleComponent::backgroundColourId, juce::Colour (0xff252535));
    bubble->setColour (juce::BubbleComponent::outlineColourId,    juce::Colour (0xff89b4fa));

    juce::AttributedString str;
    str.setWordWrap     (juce::AttributedString::byWord);
    str.setJustification (juce::Justification::centredLeft);
    str.append (text, juce::Font (12.0f), juce::Colour (0xffcdd6f4));

    auto pos = owner.getLocalPoint (pressedOn, pressPos).toInt();
    bubble->showAt (juce::Rectangle<int> (pos.x, pos.y, 1, 1), str,
                    3000 /*ms*/, true /*hideOnClick*/, true /*deleteSelf*/);
}

// ─────────────────────────────────────────────────────────────────────────────

AIMusicEditor::~AIMusicEditor()
{
    stopTimer();
    proc.onStateLoaded          = nullptr;
    proc.onPreviewStateChanged  = nullptr;
    proc.stopPreview();
    setLookAndFeel (nullptr);    // must clear before LAF is destroyed
    for (auto* s : { &sldTemperature, &sldTopP, &sldMaxTokens, &sldTempo })
        s->setLookAndFeel (nullptr);
    btnTriplets    .setLookAndFeel (nullptr);
    btnQuantize    .setLookAndFeel (nullptr);
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
    s.setColour (juce::Slider::textBoxTextColourId,          kFg);
    s.setColour (juce::Slider::textBoxOutlineColourId,       juce::Colours::transparentBlack);
    s.setColour (juce::Slider::textBoxBackgroundColourId,    kBg);
    s.setColour (juce::Slider::textBoxHighlightColourId,     kAcc.withAlpha (0.3f));
    s.setLookAndFeel (mirrorKnobLAF.get());
    addAndMakeVisible (s);
}

void AIMusicEditor::paint (juce::Graphics& g)
{
    // Pull phase first -drives both the background pulse and everything else
    float tPhase = static_cast<MirrorMirror*> (mirrorAnim.get())->phase;

    // ── Slowly pulsing diagonal background gradient ───────────────────────────
    {
        float pulse  = 0.5f + 0.5f * std::sin (tPhase * 0.13f);   // ~48-s cycle
        float pulse2 = 0.5f + 0.5f * std::sin (tPhase * 0.09f + 1.1f);
        float w = (float) getWidth(), h = (float) getHeight();

        // Primary diagonal: top-left (soft blue-indigo) → bottom-right (mid indigo)
        auto topLeft = juce::Colour (0xff2e2e48).interpolatedWith (
                           juce::Colour (0xff2a2844), pulse);
        juce::ColourGradient bg (
            topLeft,                   0.f, 0.f,
            juce::Colour (0xff191926), w,   h,   false);
        bg.addColour (0.45, juce::Colour (0xff222236));
        g.setGradientFill (bg);
        g.fillRect (getLocalBounds());

        // Cross-diagonal overlay: bottom-left (muted indigo tint) → top-right (transparent)
        auto botLeft = juce::Colour (0xff1e1832).withAlpha (0.45f + 0.12f * pulse2);
        juce::ColourGradient bg2 (
            botLeft,                          0.f, h,
            juce::Colours::transparentBlack,  w,   0.f, false);
        g.setGradientFill (bg2);
        g.fillRect (getLocalBounds());

        // Slow drifting radial bloom -very faint so it reads as atmosphere not colour
        float rx = w * (0.25f + 0.35f * (0.5f + 0.5f * std::sin (tPhase * 0.07f)));
        float ry = h * 0.65f;
        juce::ColourGradient radial (
            juce::Colour (0xff6644cc).withAlpha (0.05f + 0.03f * pulse),
            rx, ry,
            juce::Colours::transparentBlack,
            rx + w * 0.5f, ry, true);
        g.setGradientFill (radial);
        g.fillRect (getLocalBounds());
    }

    // ── Orbiting sparkles around title (phase driven by MirrorMirror's 24fps timer) ──
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

    // ── Pulsing halos for action buttons (drawn behind children) ──────────────
    float ph = static_cast<MirrorMirror*> (mirrorAnim.get())->phase;

    auto drawPulse = [&] (juce::Component& btn, juce::Colour glowCol, juce::Colour ringCol)
    {
        if (! btn.isVisible()) return;
        float pulse = 0.5f + 0.5f * std::sin (ph * 2.8f);
        auto outer = btn.getBounds().toFloat().expanded (4.f + pulse * 4.f);
        auto ring  = btn.getBounds().toFloat().expanded (1.5f + pulse * 1.5f);
        g.setColour (glowCol.withAlpha (pulse * 0.26f));
        g.fillRoundedRectangle (outer, 7.f);
        g.setColour (ringCol.withAlpha (0.35f + pulse * 0.45f));
        g.drawRoundedRectangle (ring, 6.f, 1.5f);
    };

    // "Clear" -gold pulse (error context, tap to dismiss)
    if (btnCancel.getButtonText() == "Clear")
        drawPulse (btnCancel,  juce::Colour (0xffFFD700), juce::Colour (0xffFFBB44));

    // "Show MIDI" -blue pulse (success, tap to reveal)
    drawPulse (btnShowMidi, juce::Colour (0xff89b4fa), juce::Colour (0xffaaddff));

    // "Preview" -mauve pulse while playing (matches dark theme palette)
    if (proc.isPreviewPlaying())
        drawPulse (btnPreview, juce::Colour (0xffcba6f7), juce::Colour (0xffe0b8ff));

    // ── Progress bar (processing / training / generating) ────────────────────
    auto& ps = proc.lastStatus;
    bool isProcessing  = (ps.stage == "processing" && ps.progress >= 0.f);
    bool isTraining    = (ps.stage == "training");
    bool isGenerating  = (ps.stage == "generating");

    if (isProcessing || isTraining || isGenerating)
    {
        // Each bar row is 11px tall: 8pt label on left, 4px bar track on right centred vertically
        constexpr int kRowH  = 11;
        constexpr int kBarH  = 4;
        constexpr int kLblW  = 38;   // width of the text column
        constexpr int kGap   = 4;

        int rowX = lblStatus.getX();
        int rowW = lblStatus.getWidth();
        int row1Y = lblStatus.getBottom() + 2;
        int row2Y = row1Y + kRowH + 2;

        // Bar track rectangles (right of label)
        int trackX = rowX + kLblW + kGap;
        int trackW = rowW - kLblW - kGap;

        juce::Rectangle<int> barBounds  (trackX, row1Y + (kRowH - kBarH) / 2, trackW, kBarH);
        juce::Rectangle<int> batchBounds (trackX, row2Y + (kRowH - kBarH) / 2, trackW, kBarH);

        // Continuous left-to-right chase (wraps, never bounces)
        auto drawChase = [&] (juce::Rectangle<int> r)
        {
            double t   = std::fmod (juce::Time::getMillisecondCounterHiRes() / 1200.0, 1.0);
            int    sw  = juce::roundToInt (r.getWidth() * 0.30f);
            int    sx  = r.getX() + juce::roundToInt ((r.getWidth() + sw) * (float) t) - sw;
            // Fade the leading edge so it looks like a comet, not a hard block
            for (int i = 0; i < sw; ++i)
            {
                float alpha = (float)(i + 1) / (float) sw * 0.8f;
                g.setColour (kAcc.withAlpha (alpha));
                g.fillRect (juce::Rectangle<int> (sx + i, r.getY(), 1, r.getHeight()));
            }
        };

        // ── Labels ───────────────────────────────────────────────────────────
        g.setFont (juce::Font (8.0f));
        g.setColour (kFg.withAlpha (0.45f));
        if (isTraining)
        {
            g.drawText ("epochs", juce::Rectangle<int> (rowX, row1Y, kLblW, kRowH),
                        juce::Justification::centredRight, false);
            g.drawText ("batch",  juce::Rectangle<int> (rowX, row2Y, kLblW, kRowH),
                        juce::Justification::centredRight, false);
        }

        // ── Main bar: epoch progress ─────────────────────────────────────────
        g.setColour (juce::Colour (0xff313244));
        g.fillRoundedRectangle (barBounds.toFloat(), 2.f);

        if (isProcessing)
        {
            auto filled = barBounds.withWidth (juce::roundToInt (barBounds.getWidth() * ps.progress));
            g.setColour (kAcc);
            g.fillRoundedRectangle (filled.toFloat(), 2.f);
        }
        else if (isTraining && ps.totalEpochs > 0)
        {
            // Solid fill for completed epochs (may be 0 while in first epoch)
            float frac  = juce::jlimit (0.f, 1.f, (float) ps.epoch / (float) ps.totalEpochs);
            auto filled = barBounds.withWidth (juce::roundToInt (barBounds.getWidth() * frac));
            g.setColour (kAcc);
            g.fillRoundedRectangle (filled.toFloat(), 2.f);
        }
        else
        {
            // Generating or training without epoch info -animate the main bar
            drawChase (barBounds);
        }

        // ── Batch bar: within-epoch progress ────────────────────────────────
        if (isTraining)
        {
            g.setColour (juce::Colour (0xff313244));
            g.fillRoundedRectangle (batchBounds.toFloat(), 2.f);

            if (ps.batchProgress >= 0.f)
            {
                // Deterministic fill: how far through current epoch's batches
                auto bFilled = batchBounds.withWidth (juce::roundToInt (batchBounds.getWidth() * ps.batchProgress));
                g.setColour (kAcc.withAlpha (0.65f));
                g.fillRoundedRectangle (bFilled.toFloat(), 2.f);
            }
            else
            {
                // Waiting for first batch report -chase shows training is alive
                drawChase (batchBounds);
            }
        }
        else if (isGenerating)
        {
            // Single animated bar for generating
            drawChase (barBounds);
        }

        // ── Epoch / val loss line below the bars ─────────────────────────────
        if (isTraining)
        {
            int textY = row2Y + kRowH + 3;
            juce::String epochLine;
            if (ps.epoch >= 0 && ps.totalEpochs > 0)
                epochLine = "Epoch " + juce::String (ps.epoch) + " / " + juce::String (ps.totalEpochs);
            if (ps.valLoss >= 0.0)
            {
                if (epochLine.isNotEmpty()) epochLine += "   ";
                epochLine += "val loss: " + juce::String (ps.valLoss, 4);
            }
            if (epochLine.isNotEmpty())
            {
                g.setFont (juce::Font (9.5f));
                g.setColour (kFg.withAlpha (0.45f));
                g.drawText (epochLine,
                            juce::Rectangle<int> (rowX, textY, rowW, 12),
                            juce::Justification::centred, false);
            }
        }
    }
}

void AIMusicEditor::resized()
{
    auto area = getLocalBounds().reduced (12);

    // Title strip -key (upper-left) and Save/Load (upper-right)
    {
        auto titleStrip = getLocalBounds().removeFromTop (36).reduced (8, 7);
        btnAdvanced.setBounds (titleStrip.removeFromLeft (26).withSizeKeepingCentre (26, 26));
        titleStrip.removeFromLeft (4);
        btnLoadPreset.setBounds (titleStrip.removeFromRight (42));
        titleStrip.removeFromRight (4);
        btnSavePreset.setBounds (titleStrip.removeFromRight (42));
    }

    // Mirror + its two action buttons stacked just above it
    constexpr int kMirrorW = 120, kMirrorH = 100;
    int mirrorX = getWidth() - kMirrorW - 4;
    int mirrorY = getHeight() - kMirrorH - 6;
    mirrorAnim ->setBounds (mirrorX, mirrorY,                kMirrorW, kMirrorH);
    btnCancel  .setBounds  (mirrorX, mirrorY - 26,          kMirrorW, 22);
    btnShowMidi.setBounds  (mirrorX, mirrorY - 26 - 26,     kMirrorW, 22);
    btnPreview .setBounds  (mirrorX, mirrorY - 26 - 26 - 26 - 6, kMirrorW, 22);

    area.removeFromTop (36); // title

    // Tab bar
    auto tabRow = area.removeFromTop (28);
    int  tabW   = tabRow.getWidth() / 2;
    tabProcess .setBounds (tabRow.removeFromLeft (tabW));
    tabGenerate.setBounds (tabRow);
    area.removeFromTop (5);


    // Reserve shared status from bottom (status + msg + warning + cancel)
    auto statusArea = area.removeFromBottom (90);

    if (currentTab == 0)
    {
        // ── Process & Train tab ──────────────────────────────────────────────
        // Project name at the very top of this tab
        {
            auto projRow = area.removeFromTop (24);
            lblProjectName.setBounds (projRow.removeFromLeft (52));
            projRow.removeFromLeft (5);
            edtProjectName.setBounds (projRow.removeFromLeft (210));
        }
        area.removeFromTop (6);

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

        auto btnRow = area.removeFromTop (34);
        btnRunProcess.setBounds (btnRow.removeFromLeft (140));
        btnRow.removeFromLeft (6);
        btnTrain.setBounds (btnRow.removeFromLeft (90));
        area.removeFromTop (8);
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
        // All four knobs identical height -Sync sits in its own row below
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
            btnQuantize   .setBounds (col.removeFromBottom (22));
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
             &lblProjectName, &edtProjectName,
             &lblFolder, &btnBrowseFolder, &lblInstruments,
             &chkLeadVox, &chkHarmVox, &chkGuitar, &chkBass, &chkDrums, &chkOther,
             &btnRunProcess, &btnTrain })
        c->setVisible (onProcess);

    for (juce::Component* c : std::initializer_list<juce::Component*> {
             &lblCkpt, &btnBrowseCkpt,
             &sldTemperature, &sldTopP, &sldMaxTokens, &sldTempo,
             &lblTemperature, &lblTopP, &lblMaxTokens, &lblTempo,
             &btnSyncTempo, &cmbSubdivision, &btnTriplets, &btnQuantize, &lblSubdivision,
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

    bool curIsError = (curStage == "error") || localErrorMessage.isNotEmpty();
    mm.isError = curIsError;

    // Shake "no" on fresh error
    if (curIsError && ! prevIsError)
        mm.triggerShake();

    // Nod when a new job starts (transition into a running state)
    const juce::StringArray kRunning { "processing", "training", "generating" };
    if (kRunning.contains (curStage) && ! kRunning.contains (prevStage))
        mm.triggerNod();

    // Celebration burst + wink when a job finishes
    if (curStage == "done" && kRunning.contains (prevStage))
    {
        mm.triggerCelebration();
        // Auto-update checkpoint path when training completes via a named project
        if (prevStage == "training" && proc.lastStatus.ckptPath.isNotEmpty())
        {
            proc.ckptPath = proc.lastStatus.ckptPath;
            lblCkpt.setText (proc.ckptPath, juce::dontSendNotification);
        }
    }

    prevIsError = curIsError;
    prevStage   = curStage;

    // Animate vanity icon shimmer in sync with MirrorMirror phase
    static_cast<VanityButtonLAF*> (keyButtonLAF.get())->phase = mm.phase;
    btnAdvanced.repaint();

    // Keep the progress bar repainting while a job is running
    if (kRunning.contains (curStage))
        repaint();
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
        btnPreview  .setVisible (false);
        btnRunProcess.setEnabled (true);
        btnTrain     .setEnabled (true);
        btnGenerate  .setEnabled (true);
        return;
    }

    auto& s = proc.lastStatus;
    lblStatus.setText ("Status: " + s.stage, juce::dontSendNotification);

    if (s.stage == "training")
    {
        lblMessage.setText ({}, juce::dontSendNotification);
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
                bool midiReady = juce::File (lastMidiPath).existsAsFile();
                btnShowMidi.setVisible (midiReady);
                btnPreview  .setVisible (midiReady);
            }
        }
    }
    else if (s.stage != "done")
    {
        btnShowMidi.setVisible (false);
        btnPreview  .setVisible (false);
    }

    bool busy = (s.stage == "processing" || s.stage == "training" || s.stage == "generating");
    btnCancel.setVisible (busy || s.stage == "error");
    btnCancel.setButtonText (busy ? "Cancel" : "Clear");

    bool serverReady  = (s.stage == "idle" || s.stage == "done" || s.stage == "error");
    bool generating   = (s.stage == "generating");

    btnRunProcess.setEnabled (true);
    btnTrain     .setEnabled (serverReady);
    btnGenerate  .setEnabled (serverReady);

    // Freeze all generation parameters while inference is running
    btnBrowseCkpt  .setEnabled (! generating);
    sldTemperature .setEnabled (! generating);
    sldTopP        .setEnabled (! generating);
    sldMaxTokens   .setEnabled (! generating);
    sldTempo       .setEnabled (! generating && ! proc.syncTempo);
    btnSyncTempo   .setEnabled (! generating);
    btnSeedFromData.setEnabled (! generating);
    btnQuantize    .setEnabled (! generating);
    cmbSubdivision .setEnabled (! generating && proc.quantize);
    btnTriplets    .setEnabled (! generating && proc.quantize);
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
                    proc.startProcess (proc.audioFolder, {});
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

void AIMusicEditor::browseEventsAndTrain()
{
    // Default to the most recently created events folder; fall back to runs/events/
    auto latest   = proc.fetchLatestEvents();
    auto startDir = latest.isNotEmpty()
                        ? juce::File (latest)
                        : juce::File::getSpecialLocation (juce::File::userHomeDirectory);

    auto chooser = std::make_shared<juce::FileChooser> (
        "Select events folder to train on", startDir);

    chooser->launchAsync (juce::FileBrowserComponent::openMode |
                          juce::FileBrowserComponent::canSelectDirectories,
        [this, chooser] (const juce::FileChooser& fc)
        {
            auto folder = fc.getResult();
            if (! folder.isDirectory()) return;

            if (! folder.getChildFile ("events_train.pkl").existsAsFile())
            {
                localErrorMessage = "Selected folder has no events_train.pkl -run Process Audio first.";
                updateStatusLabel();
                return;
            }
            localErrorMessage.clear();
            proc.startTrain (folder.getFullPathName());
        });
}

void AIMusicEditor::savePreset()
{
    auto startDir = proc.getPref ("lastPresetDir");
    auto dir = startDir.isNotEmpty() ? juce::File (startDir)
                                     : juce::File::getSpecialLocation (juce::File::userDocumentsDirectory);

    auto chooser = std::make_shared<juce::FileChooser> ("Save Mirror Mirror Preset", dir, "*.mmpreset");
    chooser->launchAsync (juce::FileBrowserComponent::saveMode |
                          juce::FileBrowserComponent::canSelectFiles,
        [this, chooser] (const juce::FileChooser& fc)
        {
            auto f = fc.getResult().withFileExtension (".mmpreset");
            if (f.getFullPathName().isEmpty()) return;

            juce::XmlElement xml ("MirrorMirrorPreset");
            xml.setAttribute ("version",        1);
            xml.setAttribute ("temperature",    proc.temperature);
            xml.setAttribute ("topP",           proc.topP);
            xml.setAttribute ("tempoBpm",       proc.tempoBpm);
            xml.setAttribute ("gridSubdivision", proc.gridSubdivision);
            xml.setAttribute ("allowTriplets",  proc.allowTriplets ? 1 : 0);
            xml.setAttribute ("maxTokens",      proc.maxTokens);
            xml.setAttribute ("syncTempo",      proc.syncTempo    ? 1 : 0);
            xml.setAttribute ("seedFromData",   proc.seedFromData ? 1 : 0);
            xml.setAttribute ("quantize",       proc.quantize     ? 1 : 0);
            xml.setAttribute ("ckptPath",       proc.ckptPath);
            xml.setAttribute ("audioFolder",    proc.audioFolder);
            xml.setAttribute ("selectedTracks", proc.selectedTracks);
            xml.setAttribute ("discIntensity",  proc.discIntensity);
            xml.setAttribute ("seqLen",         proc.seqLen);
            xml.writeTo (f);

            proc.setPref ("lastPresetDir", f.getParentDirectory().getFullPathName());
        });
}

void AIMusicEditor::loadPreset()
{
    auto startDir = proc.getPref ("lastPresetDir");
    auto dir = startDir.isNotEmpty() ? juce::File (startDir)
                                     : juce::File::getSpecialLocation (juce::File::userDocumentsDirectory);

    auto chooser = std::make_shared<juce::FileChooser> ("Load Mirror Mirror Preset", dir, "*.mmpreset");
    chooser->launchAsync (juce::FileBrowserComponent::openMode |
                          juce::FileBrowserComponent::canSelectFiles,
        [this, chooser] (const juce::FileChooser& fc)
        {
            auto f = fc.getResult();
            if (! f.existsAsFile()) return;

            auto xml = juce::XmlDocument::parse (f);
            if (xml == nullptr || xml->getTagName() != "MirrorMirrorPreset") return;

            proc.temperature     = (float) xml->getDoubleAttribute ("temperature",    proc.temperature);
            proc.topP            = (float) xml->getDoubleAttribute ("topP",           proc.topP);
            proc.tempoBpm        = (float) xml->getDoubleAttribute ("tempoBpm",       proc.tempoBpm);
            proc.gridSubdivision =         xml->getIntAttribute    ("gridSubdivision", proc.gridSubdivision);
            proc.allowTriplets   =         xml->getIntAttribute    ("allowTriplets",  proc.allowTriplets ? 1 : 0) != 0;
            proc.maxTokens       =         xml->getIntAttribute    ("maxTokens",      proc.maxTokens);
            proc.syncTempo       =         xml->getIntAttribute    ("syncTempo",      proc.syncTempo    ? 1 : 0) != 0;
            proc.seedFromData    =         xml->getIntAttribute    ("seedFromData",   proc.seedFromData ? 1 : 0) != 0;
            proc.quantize        =         xml->getIntAttribute    ("quantize",       proc.quantize     ? 1 : 0) != 0;
            proc.ckptPath        =         xml->getStringAttribute ("ckptPath",       proc.ckptPath);
            proc.audioFolder     =         xml->getStringAttribute ("audioFolder",    proc.audioFolder);
            proc.selectedTracks  =         xml->getStringAttribute ("selectedTracks", proc.selectedTracks);
            proc.discIntensity   = (float) xml->getDoubleAttribute ("discIntensity",  proc.discIntensity);
            proc.seqLen          =         xml->getIntAttribute    ("seqLen",         proc.seqLen);

            proc.setPref ("lastPresetDir", f.getParentDirectory().getFullPathName());
            refreshFromProcessor();
        });
}

void AIMusicEditor::refreshFromProcessor()
{
    sldTemperature.setValue (proc.temperature,    juce::dontSendNotification);
    sldTopP       .setValue (proc.topP,           juce::dontSendNotification);
    sldMaxTokens  .setValue (proc.maxTokens,      juce::dontSendNotification);
    sldTempo      .setValue (proc.tempoBpm,       juce::dontSendNotification);

    btnSyncTempo   .setToggleState (proc.syncTempo,     juce::dontSendNotification);
    btnSeedFromData.setToggleState (proc.seedFromData,  juce::dontSendNotification);
    btnQuantize    .setToggleState (proc.quantize,      juce::dontSendNotification);
    btnTriplets    .setToggleState (proc.allowTriplets, juce::dontSendNotification);

    cmbSubdivision.setSelectedId (proc.gridSubdivision, juce::dontSendNotification);

    sldTempo      .setEnabled (! proc.syncTempo);
    cmbSubdivision.setEnabled (proc.quantize);
    btnTriplets   .setEnabled (proc.quantize);

    lblCkpt  .setText (proc.ckptPath.isNotEmpty()    ? proc.ckptPath    : "No checkpoint selected",
                       juce::dontSendNotification);
    lblFolder.setText (proc.audioFolder.isNotEmpty() ? proc.audioFolder : "No folder selected",
                       juce::dontSendNotification);
    edtProjectName.setText (proc.projectName, false);

    if (proc.selectedTracks.isEmpty())
    {
        for (auto* chk : { &chkLeadVox, &chkHarmVox, &chkGuitar, &chkBass, &chkDrums, &chkOther })
            chk->setToggleState (true, juce::dontSendNotification);
    }
    else
    {
        auto tracks = juce::StringArray::fromTokens (proc.selectedTracks, ",", "");
        chkLeadVox.setToggleState (tracks.contains ("voxlead"), juce::dontSendNotification);
        chkHarmVox.setToggleState (tracks.contains ("voxharm"), juce::dontSendNotification);
        chkGuitar .setToggleState (tracks.contains ("guitar"),  juce::dontSendNotification);
        chkBass   .setToggleState (tracks.contains ("bass"),    juce::dontSendNotification);
        chkDrums  .setToggleState (tracks.contains ("drums"),   juce::dontSendNotification);
        chkOther  .setToggleState (tracks.contains ("other"),   juce::dontSendNotification);
    }

    updateTokenWarning();
}

// ── Boop gold confetti — drawn over all children so nothing occludes it ───────
void AIMusicEditor::paintOverChildren (juce::Graphics& g)
{
    auto& mm = *static_cast<MirrorMirror*> (mirrorAnim.get());
    if (mm.boopPhase < 0.f) return;

    // Burst origin: nose position translated to editor coordinates
    auto  mb  = mirrorAnim->getBounds();
    auto  np  = mm.noseCenter();
    float cx  = (float) mb.getX() + np.x;
    float cy  = (float) mb.getY() + np.y;

    // Golden palette
    static const juce::Colour kRC[] = {
        juce::Colour (0xffFFD700), juce::Colour (0xffFFEC6E),
        juce::Colour (0xffFFF3B0), juce::Colour (0xffFFC200),
        juce::Colour (0xffDAA520), juce::Colour (0xffFFE08A),
    };

    // 20 particles, angles spread in all directions (degrees -> radians baked in)
    // Each has: angle (rad), speed (px/s), delay (s)
    static const float kAngle[] = {
         0.00f,  0.31f,  0.63f,  0.94f,  1.26f,  1.57f,  1.88f,  2.20f,
         2.51f,  2.83f,  3.14f,  3.46f,  3.77f,  4.08f,  4.40f,  4.71f,
         5.03f,  5.34f,  5.65f,  5.97f
    };
    static const float kSpeed[] = {
        110.f, 85.f, 130.f, 70.f, 100.f, 120.f, 80.f, 115.f,
         90.f, 105.f, 95.f, 125.f, 75.f, 108.f, 88.f, 118.f,
         72.f, 132.f, 98.f,  82.f
    };
    static const float kDelay[] = {
        0.00f, 0.02f, 0.00f, 0.03f, 0.01f, 0.00f, 0.02f, 0.04f,
        0.01f, 0.03f, 0.00f, 0.02f, 0.04f, 0.01f, 0.03f, 0.00f,
        0.02f, 0.01f, 0.03f, 0.02f
    };
    constexpr int kN = 20;

    for (int i = 0; i < kN; ++i)
    {
        float pt = mm.boopPhase - kDelay[i];
        if (pt <= 0.f) continue;

        float vx = kSpeed[i] * std::cos (kAngle[i]);
        float vy = kSpeed[i] * std::sin (kAngle[i]);
        float px = cx + vx * pt + std::sin (pt * 3.1f + i) * 3.f;
        float py = cy + vy * pt + 40.f * pt * pt;  // gentle gravity on all

        float alpha = std::max (0.f, 1.f - pt / 2.0f);
        if (alpha < 0.02f) continue;

        // Pop open fast, then shrink with shimmer
        float sz_base = (pt < 0.25f) ? (8.f * pt / 0.25f)
                                     : (8.f * std::exp (-(pt - 0.25f) * 2.5f));
        float pulse = 1.f + 0.32f * std::sin (pt * 18.f + (float) i * 2.1f);
        float sz = std::max (0.f, sz_base * pulse);

        auto col = kRC[i % 6];
        g.setColour (col.withAlpha (alpha));
        g.fillEllipse (px - sz, py - sz, sz * 2.f, sz * 2.f);
        // Cross arms
        g.setColour (col.withAlpha (alpha * 0.6f));
        float arm = sz * 2.2f;
        g.drawLine (px - arm, py, px + arm, py, 1.0f);
        g.drawLine (px, py - arm, px, py + arm, 1.0f);
    }
}
