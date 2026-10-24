import Steel from "steel-sdk";
import { NextResponse } from "next/server";

const steel = new Steel({ steelAPIKey: process.env.STEEL_API_KEY });

export async function POST() {
  try {
    const session = await steel.sessions.create({
      useProxy: true,
      timeout: 1800000, // 30 minutes
    });

    return NextResponse.json({
      sessionId: session.id,
      sessionViewerUrl: session.sessionViewerUrl,
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
