import { CopilotChat } from "@copilotkit/react-core/v2";

type AppProps = {
  agentId: string;
  runtimeUrl: string;
  threadId: string;
  threadStorageKey: string;
};

export default function App({
  agentId,
  threadId,
  threadStorageKey,
}: AppProps) {
  function startNewChat() {
    const nextThreadId = crypto.randomUUID();
    localStorage.setItem(threadStorageKey, nextThreadId);
    window.location.reload();
  }

  return (
    <main className="app-shell">
      <section className="chat-card">
        <div className="chat-header">
          <h1>DeepAgent Chat Demo</h1>
          <button className="new-chat-button" onClick={startNewChat}>
            New chat
          </button>
        </div>

        <div className="chat-body">
          <CopilotChat
            key={threadId}
            agentId={agentId}
            threadId={threadId}
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