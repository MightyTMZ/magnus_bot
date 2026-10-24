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
    const url = page?.url() ?? "";

    return NextResponse.json({ url });
  } catch (error) {
    console.error("Failed to get URL via CDP:", error);
    return NextResponse.json(
      { error: "Failed to get current URL" },
      { status: 500 }
    );
  } finally {
    if (browser) {
      await browser.close().catch(() => {});
    }
  }
}
