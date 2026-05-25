import { CopilotChat } from "@copilotkit/react-core/v2";

type AppProps = {
  agentId: string;
  runtimeUrl: string;
};

export default function App({ agentId, runtimeUrl }: AppProps) {
  return (
    <main className="app-shell">
      <section className="chat-card">
        <div className="chat-header">
          <center><h1>DeepAgent Chat Demo</h1></center>
        </div>

        <div className="chat-body">
          <CopilotChat
            agentId={agentId}
            labels={{
              title: "DeepAgent",
              initial: "Ask me anything.",
            }}
          />
        </div>
      </section>
    </main>
  );
}