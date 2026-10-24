import Steel from "steel-sdk";
import { NextResponse, NextRequest } from "next/server";

const steel = new Steel({ steelAPIKey: process.env.STEEL_API_KEY });

// Check if an existing session is still alive
export async function GET(request: NextRequest) {
  const sessionId = request.nextUrl.searchParams.get("sessionId");
  if (!sessionId) {
    return NextResponse.json({ error: "Missing sessionId" }, { status: 400 });
  }
  try {
    const session = await steel.sessions.retrieve(sessionId);
    return NextResponse.json({
      sessionId: session.id,
      viewerUrl: session.debugUrl,
      websocketUrl: session.websocketUrl,
      status: session.status,
    });
  } catch {
    return NextResponse.json({ error: "Session not found" }, { status: 404 });
  }
}

export async function POST() {
  try {
    const session = await steel.sessions.create({
      useProxy: true,
      timeout: 15 * 60 * 1000, // 15 minutes
    });

    return NextResponse.json({
      sessionId: session.id,
      viewerUrl: session.debugUrl,
      websocketUrl: session.websocketUrl,
    });
  } catch (error) {
    console.error("Failed to create Steel session:", error);
    return NextResponse.json(
      { error: "Failed to create Steel session" },
      { status: 500 }
    );
  }
}

export async function DELETE(request: Request) {
  try {
    const { sessionId } = await request.json();
    await steel.sessions.release(sessionId);
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Failed to release Steel session:", error);
    return NextResponse.json(
      { error: "Failed to release Steel session" },
      { status: 500 }
    );
  }
}
