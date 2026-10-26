import { NextResponse } from "next/server";
import { chromium } from "playwright";
import Anthropic from "@anthropic-ai/sdk";
import { writeFile, mkdir } from "fs/promises";
import path from "path";

const anthropic = new Anthropic();

const VISION_PROMPT = `You are looking at a chess.com game in a browser. I need you to identify the pixel coordinates of two squares on the board so I can click them to make a move.

The move I need to play: {MOVE}
- Source square: {SRC}
- Destination square: {DST}

Return the CENTER pixel coordinates (x, y) of each square. The coordinates should be relative to the top-left corner of the screenshot.

You MUST respond in exactly this JSON format and nothing else:
{"src": {"x": <number>, "y": <number>}, "dst": {"x": <number>, "y": <number>}}

Important:
- Look at the board carefully — it may be flipped (black's perspective).
- The coordinates must be the CENTER of each square, not the edge.
- Return ONLY the JSON, no explanation, no markdown.`;

export async function POST(request: Request) {
  const { sessionId, move } = await request.json();
  const apiKey = process.env.STEEL_API_KEY;

  if (!sessionId || !move) {
    return NextResponse.json(
      { error: "sessionId and move are required" },
      { status: 400 }
    );
  }

  // Validate UCI format
  const uciPattern = /^[a-h][1-8][a-h][1-8][qrbn]?$/;
  if (!uciPattern.test(move)) {
    return NextResponse.json(
      { error: "Invalid UCI move format" },
      { status: 400 }
    );
  }

  const src = move.slice(0, 2);
  const dst = move.slice(2, 4);

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

    // Take screenshot
    const buffer = await page.screenshot({ type: "png", fullPage: false });
    const base64 = buffer.toString("base64");

    // // Save locally for debugging
    // const debugDir = path.join(process.cwd(), "debug-screenshots");
    // await mkdir(debugDir, { recursive: true });
    // await writeFile(path.join(debugDir, `play-move-${move}-${Date.now()}.png`), buffer);

    // Ask Claude for coordinates
    const prompt = VISION_PROMPT
      .replace("{MOVE}", move)
      .replace("{SRC}", src)
      .replace("{DST}", dst);

    const response = await anthropic.messages.create({
      model: "claude-sonnet-4-6",
      max_tokens: 200,
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: prompt },
            {
              type: "image",
              source: {
                type: "base64",
                media_type: "image/png",
                data: base64,
              },
            },
          ],
        },
      ],
    });

    const block = response.content.find((b) => b.type === "text");
    const rawText = block?.type === "text" ? block.text.trim() : "";

    // Parse coordinates. The model sometimes wraps the JSON in a ```json fence or adds a
    // sentence around it, so pull out the outermost {...} instead of parsing the raw text.
    let coords: { src: { x: number; y: number }; dst: { x: number; y: number } };
    try {
      const start = rawText.indexOf("{");
      const end = rawText.lastIndexOf("}");
      coords = JSON.parse(rawText.slice(start, end + 1));
    } catch {
      console.error(`[play-move] ${move}: unparseable vision reply:`, rawText);
      return NextResponse.json(
        { error: "Failed to parse coordinates from vision model", raw: rawText },
        { status: 500 }
      );
    }

    // Validate coordinates are reasonable (positive numbers)
    if (
      typeof coords.src?.x !== "number" || typeof coords.src?.y !== "number" ||
      typeof coords.dst?.x !== "number" || typeof coords.dst?.y !== "number" ||
      coords.src.x < 0 || coords.src.y < 0 ||
      coords.dst.x < 0 || coords.dst.y < 0
    ) {
      console.error(`[play-move] ${move}: invalid coordinates:`, rawText);
      return NextResponse.json(
        { error: "Invalid coordinates from vision model", coords },
        { status: 500 }
      );
    }

    // Click source square
    await page.mouse.click(coords.src.x, coords.src.y);

    // Small delay to let chess.com register the selection
    await page.waitForTimeout(300);

    // Click destination square
    await page.mouse.click(coords.dst.x, coords.dst.y);

    return NextResponse.json({
      success: true,
      move,
      coords,
    });
  } catch (error) {
    console.error("Play move error:", error);
    return NextResponse.json(
      { error: "Failed to play move", detail: String(error) },
      { status: 500 }
    );
  } finally {
    if (browser) {
      await browser.close().catch(() => {});
    }
  }
}
