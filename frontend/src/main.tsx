import React from "react";
import ReactDOM from "react-dom/client";
import { CopilotKit } from "@copilotkit/react-core";
import "@copilotkit/react-core/v2/styles.css";
import App from "./App";
import "./styles.css";

const runtimeUrl =
  import.meta.env.VITE_COPILOT_RUNTIME_URL ?? "http://localhost:4000/copilotkit";

const agentId =
  import.meta.env.VITE_COPILOT_AGENT_ID ?? "deepagent-demo";

const threadId =
  localStorage.getItem("deepagent-thread-id") ??
  crypto.randomUUID();

localStorage.setItem("deepagent-thread-id", threadId);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <CopilotKit
  runtimeUrl={runtimeUrl}
  agent={agentId}
  threadId={threadId}
  showDevConsole={false}
  enableInspector={false}
>
      <App agentId={agentId} runtimeUrl={runtimeUrl} />
    </CopilotKit>
  </React.StrictMode>
);