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
    float shakePhase     { -1.f }; // -1 = inactive; ≥0 = head-shake (error)
    float celebPhase     { -1.f }; // -1 = inactive; ≥0 = celebration burst + wink

    void triggerNod()         { nodPhase   = 0.f; }
    void triggerShake()       { shakePhase = 0.f; }
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

        // Nod (up-down) and shake (left-right): damped sine, same envelope
        float nodY = 0.f;
        if (nodPhase >= 0.f)
            nodY = sry * 0.28f * std::sin (nodPhase * 3.5f)
                               * std::exp  (-nodPhase * 0.28f);
        float shakeX = 0.f;
        if (shakePhase >= 0.f)
            shakeX = srx * 0.28f * std::sin (shakePhase * 3.5f)
                                 * std::exp  (-shakePhase * 0.28f);
        float fcx = scx + shakeX;  // face center x — shifts left-right on shake

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
                g.fillEllipse (fcx - omw, fmy - omh * 0.5f, omw * 2.f, omh * 2.f);

                // ── violet swirl — pulses at a different rate ──────────────
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
                // Almost-closed — thin line so it doesn't snap to smile
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
        // Drive all animated painting in the parent editor (title sparkles + button pulses)
        if (auto* parent = getParentComponent())
            parent->repaint();
        repaint();
    }
};

// ── Eye knob — used for the lone Seq Len knob on the Process & Train tab ─────
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

        // Same gold frame gradient — visual family connection
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

        // Radial striations — thin rays from pupil edge outward
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

        // Collarette ring (just outside pupil — like real iris anatomy)
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

        // ── Value pointer — gold spoke from pupil edge to iris ────────────────
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

        // outer purple glow — very subtle slow pulse
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

        // shimmer highlight — very slowly breathes and drifts
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

        // value arc — glow
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

        // Dancing reflection twinkle — sliderPos is naturally different per knob
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

                // Pale blue-lavender — same family as the knob's existing shimmer
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

        // Frozen-state overlay — rendered on top of the knob when disabled
        if (! slider.isEnabled())
        {
            // Dim veil — bg colour at ~66% alpha kills the glow and colour
            g.setColour (juce::Colour (0xa81e1e2e));
            g.fillEllipse (cx - r - 4.f, cy - r - 4.f, (r + 4.f) * 2.f, (r + 4.f) * 2.f);
            // Ice-blue rim signals "frozen", not just "turned off"
            g.setColour (juce::Colour (0x5589b4fa));
            g.drawEllipse (cx - r, cy - r, r * 2.f, r * 2.f, 1.5f);
        }
    }
};

// ── Magical toggle buttons — glowing orb instead of a tick box ───────────────
struct SmallToggleLAF : public juce::LookAndFeel_V4
{
    void drawTickBox (juce::Graphics& g, juce::Component& /*component*/,
                      float x, float y, float w, float h,
                      bool ticked, bool isEnabled,
                      bool /*highlighted*/, bool /*down*/) override
    {
        float cx = x + w * 0.5f, cy = y + h * 0.5f;
        float r  = std::min (w, h) * 0.5f - 0.5f;

        // Outer ring — dim gold, brightens when checked
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

// ── Magical UI LookAndFeel — buttons, combo boxes, popup menus ────────────────
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

        // Low-alpha tint = accent overlay (e.g., active tab — set via setColour)
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

        // Gold border — dims when idle, brightens on hover
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

AIMusicEditor::AIMusicEditor (AIMusicProcessor& p)
    : AudioProcessorEditor (&p), proc (p),
      mirrorAnim     (std::make_unique<MirrorMirror>()),
      mirrorUILAF       (std::make_unique<MirrorUILAF>()),
      smallToggleLAF    (std::make_unique<SmallToggleLAF>()),
      mirrorKnobLAF     (std::make_unique<MirrorKnobLAF>())
{
    setSize (480, 440);
    setLookAndFeel (mirrorUILAF.get());   // global — cascades to all children without explicit LAF
    addAndMakeVisible (*mirrorAnim);

    // ── Tab bar ───────────────────────────────────────────────────────────────
    auto styleTab = [&] (juce::TextButton& btn) {
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

    btnRunProcess.onClick = [this] {
        if (proc.audioFolder.isEmpty()) { browseFolder (true); return; }
        proc.selectedTracks = buildTracksString();
        proc.startProcess (proc.audioFolder);
    };
    btnTrain.onClick = [this] { browseEventsAndTrain(); };
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

    updateTabVisibility();
    startTimer (1500);
}

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
    // Pull phase first — drives both the background pulse and everything else
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

        // Slow drifting radial bloom — very faint so it reads as atmosphere not colour
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

    // "Clear" — gold pulse (error context, tap to dismiss)
    if (btnCancel.getButtonText() == "Clear")
        drawPulse (btnCancel,  juce::Colour (0xffFFD700), juce::Colour (0xffFFBB44));

    // "Show MIDI" — blue pulse (success, tap to reveal)
    drawPulse (btnShowMidi, juce::Colour (0xff89b4fa), juce::Colour (0xffaaddff));

    // "Preview" — mauve pulse while playing (matches dark theme palette)
    if (proc.isPreviewPlaying())
        drawPulse (btnPreview, juce::Colour (0xffcba6f7), juce::Colour (0xffe0b8ff));

    // ── Preprocessing progress bar ────────────────────────────────────────────
    auto& ps = proc.lastStatus;
    if (ps.stage == "processing" && ps.progress >= 0.f)
    {
        juce::Rectangle<int> barBounds (lblStatus.getX(),
                                        lblStatus.getBottom() + 1,
                                        lblStatus.getWidth(),
                                        4);
        g.setColour (juce::Colour (0xff313244));
        g.fillRoundedRectangle (barBounds.toFloat(), 2.f);
        auto filled = barBounds.withWidth (juce::roundToInt (barBounds.getWidth() * ps.progress));
        g.setColour (kAcc);
        g.fillRoundedRectangle (filled.toFloat(), 2.f);
    }
}

void AIMusicEditor::resized()
{
    auto area = getLocalBounds().reduced (12);

    // Preset Save/Load — top-right corner of the title strip
    {
        auto titleStrip = getLocalBounds().removeFromTop (36).reduced (8, 7);
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
        mm.triggerCelebration();

    prevIsError = curIsError;
    prevStage   = curStage;
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
                localErrorMessage = "Selected folder has no events_train.pkl — run Process Audio first.";
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
