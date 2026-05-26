import React from "react";
import ReactDOM from "react-dom/client";
import { CopilotKit } from "@copilotkit/react-core";
import "@copilotkit/react-core/v2/styles.css";
import App from "./App";
import "./styles.css";

const runtimeUrl =
  import.meta.env.VITE_COPILOT_RUNTIME_URL ?? "http://localhost:4000/copilotkit";

const agentId =
  import.meta.env.VITE_AGENT_ID ??
  import.meta.env.VITE_COPILOT_AGENT_ID ??
  "deepagent-demo";

const THREAD_STORAGE_KEY = `deepagent-thread-id:${agentId}`;

function getOrCreateThreadId() {
  const existing = localStorage.getItem(THREAD_STORAGE_KEY);
  console.log(existing)
  if (existing) return existing;

  const created = crypto.randomUUID();
  localStorage.setItem(THREAD_STORAGE_KEY, created);
  return created;
}

const threadId = getOrCreateThreadId();

ReactDOM.createRoot(document.getElementById("root")!).render(
  <CopilotKit
    runtimeUrl={runtimeUrl}
    agent={agentId}
    threadId={threadId}
    showDevConsole={false}
    enableInspector={false}
  >
    <App
      agentId={agentId}
      runtimeUrl={runtimeUrl}
      threadId={threadId}
      threadStorageKey={THREAD_STORAGE_KEY}
    />
  </CopilotKit>
);