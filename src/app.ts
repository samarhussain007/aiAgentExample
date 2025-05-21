import "dotenv/config";
import { tavily } from "@tavily/core";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { ChatOllama } from "@langchain/ollama";
import { MemorySaver } from "@langchain/langgraph";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { AIMessage, HumanMessage } from "@langchain/core/messages";

const tvly = tavily({ apiKey: process.env.TAVILY_API_KEY });

const tavilyWeatherTool = tool(
  async ({ query }: { query: string }) => {
    const normalized = query.toLowerCase();
    // if (
    //   !normalized.includes("weather") &&
    //   normalized.match(/^[a-z\s]+$/i) // matches city names like "bengaluru"
    // ) {
    //   query = `current weather in ${query}`;
    // }
    const response = await tvly.search(query, {
      maxResults: 3,
      includeAnswer: true,
    });

    return response;
  },
  {
    schema: z.object({
      query: z.string(),
    }),
    name: "Tavily search tool",
    description:
      "Use this tool to search for real-time weather and news information from the web.",
  }
);

const agentTools = [tavilyWeatherTool];
const model = new ChatOllama({
  model: "command-r:35b",
  temperature: 0,
});

const agentCheckpointer = new MemorySaver();
const agent = createReactAgent({
  llm: model,
  tools: agentTools,
  checkpointSaver: agentCheckpointer,
});

const agentFinalState = await agent.invoke(
  {
    messages: [new HumanMessage("What is the current weather in sf")],
  },
  {
    configurable: {
      thread_id: "42",
    },
  }
);

console.log(
  "This is the first invoke: ",
  agentFinalState.messages[agentFinalState.messages.length - 1].content
);

const lastMessage = [
  agentFinalState.messages[agentFinalState.messages.length - 1].content,
];

const agentNextState = await agent.invoke(
  {
    messages: [
      ...agentFinalState.messages,
      new HumanMessage("What about bengaluru"),
    ],
  },
  {
    configurable: {
      thread_id: "42",
    },
  }
);

console.log(
  "This is the last invoke: ",
  agentNextState.messages[agentNextState.messages.length - 1].content
);
