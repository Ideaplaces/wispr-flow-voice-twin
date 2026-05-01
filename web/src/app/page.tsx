"use client";

import dynamic from "next/dynamic";

const GraphExplorer = dynamic(
  () => import("./components/GraphExplorer"),
  { ssr: false, loading: () => <Loading /> }
);

function Loading() {
  return (
    <div
      style={{
        height: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        color: "#888",
        fontSize: 13,
      }}
    >
      Loading graph...
    </div>
  );
}

export default function Page() {
  return <GraphExplorer />;
}
