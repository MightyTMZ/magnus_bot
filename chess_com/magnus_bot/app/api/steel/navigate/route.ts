import { NextResponse } from "next/server";
import { chromium } from "playwright";

export async function POST(request: Request) {
  const { sessionId, url } = await request.json();
  const apiKey = process.env.STEEL_API_KEY;

  if (!sessionId || !url) {
    return NextResponse.json(
      { error: "sessionId and url are required" },
      { status: 400 }
    );
  }

  let browser;
  try {
    browser = await chromium.connectOverCDP(
      `wss://connect.steel.dev?apiKey=${apiKey}&sessionId=${sessionId}`
    );

    const context = browser.contexts()[0];
    const page = context.pages()[0] ?? (await context.newPage());
    await page.goto(url, { waitUntil: "domcontentloaded" });

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Navigate error:", error);
    return NextResponse.json(
      { error: "Failed to navigate" },
      { status: 500 }
    );
  } finally {
    if (browser) {
      await browser.close().catch(() => {});
    }
  }
}
