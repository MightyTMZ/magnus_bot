import { NextResponse } from "next/server";

export async function POST(request: Request) {
  const { sessionId, url } = await request.json();
  const apiKey = process.env.STEEL_API_KEY;

  if (!sessionId || !url) {
    return NextResponse.json(
      { error: "sessionId and url are required" },
      { status: 400 }
    );
  }

  try {
    const response = await fetch(
      `https://api.steel.dev/v1/sessions/${sessionId}/actions/navigate`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "steel-api-key": apiKey!,
        },
        body: JSON.stringify({ url }),
      }
    );

    if (!response.ok) {
      const text = await response.text();
      console.error("Steel navigate failed:", text);
      return NextResponse.json(
        { error: "Failed to navigate" },
        { status: 500 }
      );
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Navigate error:", error);
    return NextResponse.json(
      { error: "Failed to navigate" },
      { status: 500 }
    );
  }
}
