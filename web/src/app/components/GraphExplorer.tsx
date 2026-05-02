"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type Node = {
  id: string;
  x: number;
  y: number;
  label: string;
  color: string;
  topic_id: number;
  topic_label: string;
  ctx: string;
  ts: string;
  text: string;
};

type Edge = {
  source: string;
  target: string;
  weight: number;
};

type Topic = {
  id: number;
  label: string;
  color: string;
  count: number;
};

type GraphData = {
  nodes: Node[];
  edges: Edge[];
  topics: Topic[];
  stats: { n_nodes: number; n_edges: number; n_topics: number };
};

export default function GraphExplorer() {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const sigmaRef = useRef<unknown>(null);
  const graphRef = useRef<unknown>(null);
  const nodeIndexRef = useRef<Map<string, Node>>(new Map());

  const [data, setData] = useState<GraphData | null>(null);
  const [selected, setSelected] = useState<Node | null>(null);
  const [hovered, setHovered] = useState<Node | null>(null);
  const [topicFilter, setTopicFilter] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadProgress, setLoadProgress] = useState<string>("Fetching graph...");

  // Neighbor index: id -> set of neighbor ids, built once after graph load.
  const neighborsRef = useRef<Map<string, Set<string>>>(new Map());

  useEffect(() => {
    let cancelled = false;
    let resizeObserver: ResizeObserver | null = null;

    (async () => {
      try {
        setLoadProgress("Fetching graph artifact...");
        const res = await fetch("/api/graph");
        if (!res.ok) throw new Error(`graph fetch failed: ${res.status}`);
        const payload: GraphData = await res.json();
        if (cancelled) return;
        setData(payload);

        setLoadProgress("Building graphology index...");
        const Graph = (await import("graphology")).default;
        const SigmaModule = await import("sigma");
        const Sigma = SigmaModule.default;

        const graph = new Graph({ multi: false, type: "undirected" });
        const idx = new Map<string, Node>();
        for (const n of payload.nodes) {
          idx.set(n.id, n);
          // Outliers and long-tail topics get smaller markers so the labeled
          // top-30 clusters pop visually. Color encodes the rank already.
          const isOutlier = n.topic_id === -1;
          const isLongTail = !isOutlier && n.color === "#555555";
          const size = isOutlier ? 0.9 : isLongTail ? 1.4 : 2.6;
          graph.addNode(n.id, {
            x: n.x,
            y: n.y,
            label: n.label,
            size,
            color: n.color,
          });
        }
        nodeIndexRef.current = idx;

        setLoadProgress(`Connecting ${payload.edges.length.toLocaleString()} edges...`);
        const neighbors = new Map<string, Set<string>>();
        for (const e of payload.edges) {
          if (!graph.hasNode(e.source) || !graph.hasNode(e.target) || graph.hasEdge(e.source, e.target)) continue;
          graph.addEdge(e.source, e.target, {
            size: 0.4,
            color: "rgba(180,180,200,0.06)",
            hidden: true,
          });
          if (!neighbors.has(e.source)) neighbors.set(e.source, new Set());
          if (!neighbors.has(e.target)) neighbors.set(e.target, new Set());
          neighbors.get(e.source)!.add(e.target);
          neighbors.get(e.target)!.add(e.source);
        }
        neighborsRef.current = neighbors;

        if (cancelled || !containerRef.current) return;
        setLoadProgress("Rendering...");

        const renderer = new Sigma(graph, containerRef.current, {
          renderEdgeLabels: false,
          defaultNodeColor: "#666",
          labelColor: { color: "#cccccc" },
          labelDensity: 0.04,
          labelGridCellSize: 120,
          labelRenderedSizeThreshold: 24,
          labelFont: "Inter, sans-serif",
          labelSize: 11,
          minCameraRatio: 0.05,
          maxCameraRatio: 8,
        });

        // clickStage fires immediately after clickNode in sigma 3, so we need
        // a guard timestamp to keep the selection sticky after a node click.
        let lastNodeClick = 0;
        renderer.on("clickNode", ({ node }) => {
          lastNodeClick = Date.now();
          const n = nodeIndexRef.current.get(node);
          if (n) {
            setSelected(n);
            // Window flag for headless-screenshot harness; harmless otherwise.
            (window as unknown as { __selectedNodeId?: string }).__selectedNodeId = n.id;
          }
        });
        renderer.on("enterNode", ({ node }) => {
          const n = nodeIndexRef.current.get(node);
          if (n) setHovered(n);
        });
        renderer.on("leaveNode", () => setHovered(null));
        renderer.on("clickStage", () => {
          if (Date.now() - lastNodeClick < 300) return;
          setSelected(null);
        });

        sigmaRef.current = renderer;
        graphRef.current = graph;

        if (containerRef.current) {
          resizeObserver = new ResizeObserver(() => renderer.refresh());
          resizeObserver.observe(containerRef.current);
        }
        // Optional initial selection from ?select=<nodeId> for screenshots.
        if (typeof window !== "undefined") {
          const params = new URLSearchParams(window.location.search);
          const want = params.get("select");
          if (want && idx.has(want)) setSelected(idx.get(want)!);
        }
      } catch (e) {
        if (!cancelled) setError((e as Error).message);
      }
    })();

    return () => {
      cancelled = true;
      if (resizeObserver) resizeObserver.disconnect();
      const r = sigmaRef.current as { kill?: () => void } | null;
      if (r && typeof r.kill === "function") r.kill();
    };
  }, []);

  // Reducers: hide edges by default, show only the selected node's neighbors,
  // and dim nodes outside the active topic filter when one is set.
  useEffect(() => {
    const renderer = sigmaRef.current as { setSetting: (k: string, v: unknown) => void; refresh: () => void } | null;
    if (!renderer) return;

    const selectedId = selected?.id ?? null;
    const selectedNeighbors = selectedId ? neighborsRef.current.get(selectedId) : null;
    // Hovered node's neighbors only contribute when nothing is selected, so a
    // mouse glimpse never overrides a sticky click.
    const hoveredId = !selectedId ? hovered?.id ?? null : null;
    const hoveredNeighbors = hoveredId ? neighborsRef.current.get(hoveredId) : null;

    renderer.setSetting("nodeReducer", (id: string, attrs: Record<string, unknown>) => {
      const n = nodeIndexRef.current.get(id);
      if (!n) return attrs;

      // Topic filter dims everything outside the chosen topic.
      if (topicFilter !== null && n.topic_id !== topicFilter) {
        return { ...attrs, color: "#1a1a1a", size: 0.8, label: "" };
      }
      // Selection: highlight the chosen node and its neighbors.
      if (selectedId) {
        if (id === selectedId) return { ...attrs, size: 8, zIndex: 2, color: n.color };
        if (selectedNeighbors && selectedNeighbors.has(id)) {
          return { ...attrs, size: 5, color: n.color, zIndex: 1 };
        }
        return { ...attrs, color: "#222222", size: 0.9, label: "" };
      }
      if (topicFilter !== null) return { ...attrs, size: 4 };
      // Hover preview: gently emphasize the hovered node and its neighbors.
      if (hoveredId) {
        if (id === hoveredId) return { ...attrs, size: 5, zIndex: 2 };
        if (hoveredNeighbors && hoveredNeighbors.has(id)) {
          return { ...attrs, size: 4, zIndex: 1 };
        }
      }
      return attrs;
    });

    const g = graphRef.current as {
      source: (e: string) => string;
      target: (e: string) => string;
    } | null;

    renderer.setSetting("edgeReducer", (edgeId: string, attrs: Record<string, unknown>) => {
      if (!g) return { ...attrs, hidden: true };
      const src = g.source(edgeId);
      const tgt = g.target(edgeId);

      // Show only neighbor edges of the selected node.
      if (selectedId) {
        if (src === selectedId || tgt === selectedId) {
          const other = src === selectedId ? tgt : src;
          const otherNode = nodeIndexRef.current.get(other);
          return {
            ...attrs,
            color: otherNode?.color ?? "#ffffff",
            size: 2.5,
            hidden: false,
          };
        }
        return { ...attrs, hidden: true };
      }
      // Topic filter: keep within-topic edges visible, hide the rest.
      if (topicFilter !== null) {
        const a = nodeIndexRef.current.get(src);
        const b = nodeIndexRef.current.get(tgt);
        if (!a || !b || a.topic_id !== topicFilter || b.topic_id !== topicFilter) {
          return { ...attrs, hidden: true };
        }
        return { ...attrs, color: "rgba(255,255,255,0.3)", size: 0.6, hidden: false };
      }
      // Default: edges only show for the hovered node's neighbors. Without
      // a hover, the graph stays a clean cluster scatter.
      if (hoveredId && (src === hoveredId || tgt === hoveredId)) {
        const other = src === hoveredId ? tgt : src;
        const otherNode = nodeIndexRef.current.get(other);
        return {
          ...attrs,
          color: otherNode?.color ?? "rgba(255,255,255,0.6)",
          size: 1.4,
          hidden: false,
        };
      }
      return { ...attrs, hidden: true };
    });

    renderer.refresh();
  }, [selected, hovered, topicFilter]);

  const topTopics = useMemo(() => (data?.topics ?? []).slice(0, 30), [data]);

  return (
    <div style={{ display: "flex", height: "100vh", overflow: "hidden" }}>
      <div ref={containerRef} style={{ flex: 1, position: "relative", background: "#0a0a0a" }}>
        {!data && !error && (
          <div
            style={{
              position: "absolute",
              top: "50%",
              left: "50%",
              transform: "translate(-50%, -50%)",
              color: "#888",
              fontSize: 13,
              fontFamily: "monospace",
            }}
          >
            {loadProgress}
          </div>
        )}
        {error && (
          <div
            style={{
              position: "absolute",
              top: "50%",
              left: "50%",
              transform: "translate(-50%, -50%)",
              color: "#e74c3c",
              fontSize: 13,
              fontFamily: "monospace",
              padding: 16,
              background: "#1a0000",
              border: "1px solid #401010",
              borderRadius: 4,
            }}
          >
            Error: {error}
          </div>
        )}
        {hovered && !selected && (
          <div
            style={{
              position: "absolute",
              top: 16,
              left: 16,
              maxWidth: 480,
              padding: "10px 14px",
              background: "rgba(15,15,15,0.95)",
              border: `1px solid ${hovered.color}`,
              borderRadius: 4,
              fontSize: 12,
              lineHeight: 1.5,
              color: "#e4e4e4",
              pointerEvents: "none",
              backdropFilter: "blur(8px)",
            }}
          >
            <div style={{ color: hovered.color, fontSize: 10, textTransform: "uppercase", letterSpacing: 0.6, marginBottom: 4 }}>
              {hovered.topic_label}
            </div>
            <div>{hovered.label}</div>
          </div>
        )}
      </div>
      <aside
        style={{
          width: 380,
          padding: "18px 20px",
          borderLeft: "1px solid #1f1f1f",
          overflowY: "auto",
          background: "#0d0d0d",
          color: "#e4e4e4",
        }}
      >
        <h1 style={{ fontSize: 17, fontWeight: 600, margin: "0 0 4px 0" }}>Voice Twin Explorer</h1>
        {data && (
          <div style={{ fontSize: 11, color: "#888", margin: "0 0 18px 0" }}>
            {data.stats.n_nodes.toLocaleString()} dictations &middot;{" "}
            {data.stats.n_edges.toLocaleString()} edges &middot;{" "}
            {data.stats.n_topics} topics
          </div>
        )}

        {selected ? (
          <SelectedPanel node={selected} onBack={() => setSelected(null)} />
        ) : (
          <div>
            <p style={{ fontSize: 12, color: "#777", lineHeight: 1.55, margin: "0 0 18px 0" }}>
              Hover any node to glimpse its three nearest neighbors. Click to
              hold the highlight and read the dictation. Pick a topic to
              isolate one thread.
            </p>
            <div style={{ marginBottom: 12, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <span style={{ fontSize: 11, color: "#666", textTransform: "uppercase", letterSpacing: 0.6 }}>
                Topics ({topTopics.length} of {data?.stats.n_topics ?? 0})
              </span>
              {topicFilter !== null && (
                <button
                  onClick={() => setTopicFilter(null)}
                  style={{
                    fontSize: 11,
                    padding: "2px 8px",
                    background: "transparent",
                    color: "#888",
                    border: "1px solid #333",
                    borderRadius: 3,
                    cursor: "pointer",
                  }}
                >
                  clear
                </button>
              )}
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
              {topTopics.map((t) => (
                <button
                  key={t.id}
                  onClick={() => setTopicFilter(topicFilter === t.id ? null : t.id)}
                  style={{
                    textAlign: "left",
                    padding: "6px 10px",
                    border: "none",
                    borderRadius: 3,
                    background: topicFilter === t.id ? "#1f1f1f" : "transparent",
                    color: "#d6d6d6",
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                    fontSize: 12,
                  }}
                >
                  <span
                    style={{
                      width: 9,
                      height: 9,
                      borderRadius: "50%",
                      background: t.color,
                      flexShrink: 0,
                    }}
                  />
                  <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {t.label}
                  </span>
                  <span style={{ color: "#666", fontSize: 10, fontVariantNumeric: "tabular-nums" }}>
                    {t.count}
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}
      </aside>
    </div>
  );
}

function SelectedPanel({ node, onBack }: { node: Node; onBack: () => void }) {
  return (
    <div>
      <button
        onClick={onBack}
        style={{
          fontSize: 11,
          padding: "2px 8px",
          background: "transparent",
          color: "#888",
          border: "1px solid #333",
          borderRadius: 3,
          cursor: "pointer",
          marginBottom: 14,
        }}
      >
        &larr; back
      </button>
      <div style={{ fontSize: 11, color: "#888", marginBottom: 6 }}>
        {node.ts || "no timestamp"} &middot; {node.ctx}
      </div>
      <div
        style={{
          display: "inline-block",
          padding: "3px 9px",
          borderRadius: 3,
          background: "#1a1a1a",
          fontSize: 11,
          color: node.color,
          marginBottom: 14,
          border: `1px solid ${node.color}40`,
        }}
      >
        {node.topic_label}
      </div>
      <p style={{ fontSize: 14, lineHeight: 1.55, margin: 0, color: "#e8e8e8", whiteSpace: "pre-wrap" }}>
        {node.text}
      </p>
    </div>
  );
}
