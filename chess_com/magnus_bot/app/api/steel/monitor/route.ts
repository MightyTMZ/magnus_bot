import { NextResponse } from "next/server";
import { chromium } from "playwright";
import Anthropic from "@anthropic-ai/sdk";
import { writeFile, mkdir } from "fs/promises";
import path from "path";

const anthropic = new Anthropic();

// In-memory store for the previous screenshot, per session AND per consumer: the page and the
// arm loop (consumer "arms") both poll, and with one shared baseline whichever polls first eats the move.
const previousScreenshots = new Map<string, Buffer>();

/**
 * Compare two PNG buffers by raw pixel bytes.
 * Returns the fraction of bytes that differ (0.0 = identical, 1.0 = completely different).
 */
function bufferDiffRatio(a: Buffer, b: Buffer): number {
  // Quick check: if lengths differ, something changed
  if (a.length !== b.length) return 1.0;

  let diffCount = 0;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) diffCount++;
  }
  return diffCount / a.length;
}

const DIFF_THRESHOLD = 0.001; // 0.1% of bytes must differ to trigger vision check

const VISION_PROMPT = `You are monitoring a chess.com game. You are given two screenshots: BEFORE and AFTER.

Your job: determine if a new chess move was played by WHITE between these two screenshots.

Rules:
- ONLY detect moves made by WHITE. Ignore any moves made by Black.
- Be CONSERVATIVE. Only report a move if you are confident one was played.
- If you're unsure, return an empty string.
- If a White move was played, return it in UCI format (e.g., "e2e4", "g1f3", "e1g1" for castling).
- Look at the board position and the move list on the right side of the screen.
- Ignore changes in ads, popups, clocks, chat, or other non-game UI elements.
- Only return the SINGLE most recent new White move, not the full move list.

Respond with ONLY the UCI move string, or an empty string if no new White move was detected. No explanation, no punctuation, no formatting — just the move or nothing.`;

export async function POST(request: Request) {
  const { sessionId, consumer } = await request.json();
  const key = `${sessionId}:${consumer ?? "page"}`;
  const apiKey = process.env.STEEL_API_KEY;

  if (!sessionId) {
    return NextResponse.json(
      { error: "sessionId is required" },
      { status: 400 }
    );
  }

  let browser;
  try {
    browser = await chromium.connectOverCDP(
      `wss://connect.steel.dev?apiKey=${apiKey}&sessionId=${sessionId}`
    );

    const context = browser.contexts()[0];
    const page = context.pages()[0];
    if (!page) {
      return NextResponse.json({ error: "No page open" }, { status: 400 });
    }

    // Take current screenshot
    const currentBuffer = await page.screenshot({ type: "png", fullPage: false });
    const previousBuffer = previousScreenshots.get(key);

    // Store current as the new "previous" for next poll
    previousScreenshots.set(key, currentBuffer);

    // First call: no previous screenshot to compare against
    if (!previousBuffer) {
      return NextResponse.json({ move: "", reason: "first_screenshot" });
    }

    // Check pixel diff
    const diffRatio = bufferDiffRatio(previousBuffer, currentBuffer);

    if (diffRatio < DIFF_THRESHOLD) {
      return NextResponse.json({ move: "", reason: "no_change", diffRatio });
    }

    console.log(`[monitor] Pixel diff detected (ratio: ${diffRatio.toFixed(4)}), sending to vision...`);

    // // Save before/after locally for debugging
    // const debugDir = path.join(process.cwd(), "debug-screenshots");
    // await mkdir(debugDir, { recursive: true });
    // const ts = Date.now();
    // await writeFile(path.join(debugDir, `monitor-${ts}-before.png`), previousBuffer);
    // await writeFile(path.join(debugDir, `monitor-${ts}-after.png`), currentBuffer);

    // Screenshots differ: ask Claude Vision to detect the move
    const beforeBase64 = previousBuffer.toString("base64");
    const afterBase64 = currentBuffer.toString("base64");

    const response = await anthropic.messages.create({
      model: "claude-sonnet-4-6",
      max_tokens: 20,
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: VISION_PROMPT },
            {
              type: "image",
              source: {
                type: "base64",
                media_type: "image/png",
                data: beforeBase64,
              },
            },
            {
              type: "image",
              source: {
                type: "base64",
                media_type: "image/png",
                data: afterBase64,
              },
            },
          ],
        },
      ],
    });

    const rawText =
      response.content[0].type === "text" ? response.content[0].text.trim() : "";

    // Validate: should be empty or a UCI move (2-5 chars, e.g. e2e4, e7e8q)
    const uciPattern = /^[a-h][1-8][a-h][1-8][qrbn]?$/;
    const move = uciPattern.test(rawText) ? rawText : "";

    if (move) {
      console.log(`[monitor] Move detected: ${move}`);
    } else if (rawText) {
      console.log(`[monitor] Vision returned non-move text: "${rawText}"`);
    } else {
      console.log(`[monitor] Vision returned empty: no move detected`);
    }

    return NextResponse.json({ move, diffRatio });
  } catch (error) {
    console.error("Monitor error:", error);
    return NextResponse.json(
      { error: "Monitor failed" },
      { status: 500 }
    );
  } finally {
    if (browser) {
      await browser.close().catch(() => {});
    }
  }
}
