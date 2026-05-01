import { promises as fs } from "fs";
import path from "path";
import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

export async function GET() {
  const filePath = path.resolve(process.cwd(), "..", "data", "viz", "graph.json");
  try {
    const data = await fs.readFile(filePath, "utf-8");
    return new NextResponse(data, {
      headers: {
        "Content-Type": "application/json",
        "Cache-Control": "no-store",
      },
    });
  } catch (err) {
    return NextResponse.json(
      { error: `Could not read ${filePath}: ${(err as Error).message}` },
      { status: 500 }
    );
  }
}
