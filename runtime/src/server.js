import cors from "cors";
import dotenv from "dotenv";
import express from "express";
import {
  CopilotRuntime,
  ExperimentalEmptyAdapter,
  copilotRuntimeNodeExpressEndpoint,
} from "@copilotkit/runtime";
import { LangGraphHttpAgent } from "@copilotkit/runtime/langgraph";

dotenv.config();

const port = Number(process.env.PORT ?? 4000);
const mountPath = process.env.COPILOT_RUNTIME_PATH ?? "/copilotkit";
const agentId = process.env.AGENT_ID ?? "deepagent-demo";
const agentUrl = process.env.AGENT_URL ?? "http://backend:8000/copilotkit";
const corsOrigins = (process.env.CORS_ORIGINS ?? "http://localhost:5173,http://127.0.0.1:5173")
  .split(",")
  .map((origin) => origin.trim())
  .filter(Boolean);

const app = express();

app.use(
  cors({
    origin(origin, callback) {
      if (!origin || corsOrigins.includes(origin)) {
        callback(null, true);
        return;
      }

      callback(new Error(`Origin ${origin} is not allowed by CORS`));
    },
    credentials: true,
  }),
);

app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ extended: true, limit: "50mb" }));

app.get("/health", (_req, res) => {
  res.json({
    ok: true,
    service: "copilot-runtime",
    mountPath,
    agentId,
    agentUrl,
  });
});

const serviceAdapter = new ExperimentalEmptyAdapter();

const runtime = new CopilotRuntime({
  agents: {
    [agentId]: new LangGraphHttpAgent({
      url: agentUrl,
    }),
  },
});

const copilotHandler = copilotRuntimeNodeExpressEndpoint({
  endpoint: "/",
  runtime,
  serviceAdapter,
});

app.use(mountPath, (req, res, next) => {
  Promise.resolve(copilotHandler(req, res)).catch(next);
});

app.use((err, _req, res, _next) => {
  console.error("Unhandled Copilot Runtime error:", err);
  if (res.headersSent) {
    return;
  }

  res.status(500).json({
    error: "copilot_runtime_error",
    message: err instanceof Error ? err.message : String(err),
  });
});

app.listen(port, "0.0.0.0", () => {
  console.log(`Copilot Runtime listening on http://0.0.0.0:${port}${mountPath}`);
  console.log(`Registered agent '${agentId}' -> ${agentUrl}`);
});
