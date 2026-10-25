import { NextResponse } from "next/server";
import { chromium } from "playwright";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const sessionId = searchParams.get("sessionId");
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

    const buffer = await page.screenshot({ type: "png", fullPage: false });
    const base64 = buffer.toString("base64");

    return NextResponse.json({ screenshot: base64 });
  } catch (error) {
    console.error("Screenshot error:", error);
    return NextResponse.json(
      { error: "Failed to take screenshot" },
      { status: 500 }
    );
  } finally {
    if (browser) {
      await browser.close().catch(() => {});
    }
  }
}
