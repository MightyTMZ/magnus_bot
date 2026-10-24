import { NextResponse } from "next/server";
import WebSocket from "ws";

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

  try {
    const url = await getCdpCurrentUrl(sessionId, apiKey!);
    return NextResponse.json({ url });
  } catch (error) {
    console.error("Failed to get URL via CDP:", error);
    return NextResponse.json(
      { error: "Failed to get current URL" },
      { status: 500 }
    );
  }
}

function getCdpCurrentUrl(
  sessionId: string,
  apiKey: string
): Promise<string> {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(
      `wss://connect.steel.dev?apiKey=${apiKey}&sessionId=${sessionId}`
    );

    const timeout = setTimeout(() => {
      ws.close();
      reject(new Error("CDP connection timed out"));
    }, 5000);

    let messageId = 1;

    ws.on("open", () => {
      // Get the current page URL via CDP
      ws.send(
        JSON.stringify({
          id: messageId,
          method: "Target.getTargets",
        })
      );
    });

    ws.on("message", (data: WebSocket.Data) => {
      const msg = JSON.parse(data.toString());
      if (msg.id === messageId && msg.result?.targetInfos) {
        const pageTarget = msg.result.targetInfos.find(
          (t: { type: string }) => t.type === "page"
        );
        clearTimeout(timeout);
        ws.close();
        resolve(pageTarget?.url ?? "");
      }
    });

    ws.on("error", (err: Error) => {
      clearTimeout(timeout);
      reject(err);
    });
  });
}
