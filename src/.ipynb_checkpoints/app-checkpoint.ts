import dotenv from "dotenv";
dotenv.config();

import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { tavily } from "@tavily/core";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { pull } from "langchain/hub";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

import { StringOutputParser } from "@langchain/core/output_parsers";
import { Annotation, END, START } from "@langchain/langgraph";
import { formatDocumentsAsString } from "langchain/util/document";
import { z } from "zod";

import { StateGraph } from "@langchain/langgraph";
import * as d3 from "d3";
import * as tslab from "tslab";

import { select, selectAll } from "d3";
import { Document, DocumentInterface } from "@langchain/core/documents";
import { tool } from "@langchain/core/tools";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { createCanvas } from "canvas";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { StructuredTool } from "@langchain/core/tools";
import { Runnable, RunnableConfig } from "@langchain/core/runnables";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { BaseMessage } from "@langchain/core/messages";
import { ChatOllama } from "@langchain/ollama";

// const llm = new ChatOllama({
//   model: "llama3.1:8b",
//   temperature: 0,
// });

async function createAgent({
  llm,
  tools,
  systemMessage,
}: {
  llm: ChatGoogleGenerativeAI;
  tools: StructuredTool[];
  systemMessage: string;
}) {
  const toolNames = tools.map((tool) => tool.name).join(", ");
  let prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "You are a helpful AI assistant, collaborating with other assistants." +
        " Use the provided tools to progress towards answering the question." +
        " If you are unable to fully answer, that's OK, another assistant with different tools " +
        " will help where you left off. Execute what you can to make progress." +
        " If you or any of the other assistants have the final answer or deliverable," +
        " prefix your response with FINAL ANSWER so the team knows to stop." +
        " You have access to the following tools: {tool_names}.\n{system_message}",
    ],
    new MessagesPlaceholder("messages"),
  ]);

  prompt = await prompt.partial({
    tool_names: toolNames,
    system_message: systemMessage,
  });

  return prompt.pipe(llm.bindTools(tools));
}

const AgentState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
  }),
  sender: Annotation<string>({
    reducer: (x, y) => y ?? x ?? "user",
    default: () => "user",
  }),
});
//@ts-ignore
const chartTool = tool(
  ({ data }: { data: { label: string; value: number }[] }) => {
    console.log("Generating bar chart with data:", data);
    const width = 500;
    const height = 500;
    const margin = { top: 20, right: 30, bottom: 30, left: 40 };

    const canvas = createCanvas(width, height);
    const ctx = canvas.getContext("2d");

    const x = d3
      .scaleBand()
      .domain(data.map((d) => d.label))
      .range([margin.left, width - margin.right])
      .padding(0.1);

    const y = d3
      .scaleLinear()
      .domain([0, d3.max(data, (d) => d.value) ?? 0])
      .nice()
      .range([height - margin.bottom, margin.top]);

    const colorPalette = [
      "#e6194B",
      "#3cb44b",
      "#ffe119",
      "#4363d8",
      "#f58231",
      "#911eb4",
      "#42d4f4",
      "#f032e6",
      "#bfef45",
      "#fabebe",
    ];

    data.forEach((d, idx) => {
      ctx.fillStyle = colorPalette[idx % colorPalette.length];
      ctx.fillRect(
        x(d.label) ?? 0,
        y(d.value),
        x.bandwidth(),
        height - margin.bottom - y(d.value)
      );
    });

    ctx.beginPath();
    ctx.strokeStyle = "black";
    ctx.moveTo(margin.left, height - margin.bottom);
    ctx.lineTo(width - margin.right, height - margin.bottom);
    ctx.stroke();

    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    x.domain().forEach((d) => {
      const xCoord = (x(d) ?? 0) + x.bandwidth() / 2;
      ctx.fillText(d, xCoord, height - margin.bottom + 6);
    });

    ctx.beginPath();
    ctx.moveTo(margin.left, height - margin.top);
    ctx.lineTo(margin.left, height - margin.bottom);
    ctx.stroke();

    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const ticks = y.ticks();
    ticks.forEach((d) => {
      const yCoord = y(d); // height - margin.bottom - y(d);
      ctx.moveTo(margin.left, yCoord);
      ctx.lineTo(margin.left - 6, yCoord);
      ctx.stroke();
      ctx.fillText(d.toString(), margin.left - 8, yCoord);
    });
    tslab.display.png(canvas.toBuffer());
    return "Chart has been generated and displayed to the user!";
  },
  {
    name: "generate_bar_chart",
    description:
      "Generates a bar chart from an array of data points using D3.js and displays it for the user.",
    schema: z.object({
      data: z
        .object({
          label: z.string(),
          value: z.number(),
        })
        .array(),
    }),
  }
);

async function runAgentNode(props: {
  state: typeof AgentState.State;
  agent: Runnable;
  name: string;
  config?: RunnableConfig;
}) {
  const { state, agent, name, config } = props;

  let result = await agent.invoke(state, config);
  if (!result?.tool_calls || result.tool_calls.length === 0) {
    result = new HumanMessage({ ...result, name: name });
  }
  return {
    messages: [result],
    sender: name,
  };
}

const llm = new ChatGoogleGenerativeAI({
  model: "gemini-2.5-flash-preview-05-20",
  apiKey: process.env.GOOGLE_API_KEY,
  temperature: 0,
});

const tvly = tavily({ apiKey: process.env.TAVILY_API_KEY });
//@ts-ignore
const tavilySearchTool = tool(
  async ({ query }: { query: string }) => {
    const response = await tvly.search(query, {
      maxResults: 3,
    });

    return response;
  },
  {
    schema: z.object({
      query: z.string(),
    }),
    name: "tavily_search_tool",
    description: "Use this tool to search information from the web.",
  }
);

// Research agent and node
const researchAgent = await createAgent({
  llm,
  tools: [tavilySearchTool],
  systemMessage: `
You are the Researcher in a multi-agent workflow whose sole job is to fetch and hand off data for charting.

1. Read the user’s request and determine what data is needed.
2. If you lack the data, call the 'tavily_search_tool' with a clear, descriptive query.
3. When you see structured data in the form of an array of objects with "label" and "value" (e.g. [{ "label": "2021", "value": 23.594 }, …]), you must immediately call the 'generate_bar_chart' tool:
   {
     "functionCall": {
       "name": "generate_bar_chart",
       "args": { "data": [ /* your array */ ] }
     }
   }
4. Do NOT add any commentary, analysis, or use the phrase “FINAL ANSWER”—that is the ChartGenerator’s role.
5. After issuing the function call, stop. Wait for the chart agent to run the tool and render the chart.

Follow these steps exactly for *any* charting task.
`,
});

async function researchNode(
  state: typeof AgentState.State,
  config?: RunnableConfig
) {
  return runAgentNode({
    state: state,
    agent: researchAgent,
    name: "Researcher",
    config,
  });
}

// Chart generation agent and node
const chartAgent = await createAgent({
  llm,
  tools: [chartTool],
  systemMessage: "Any charts you display will be visible by the user.",
});
async function chartNode(state: typeof AgentState.State) {
  return runAgentNode({
    state: state,
    agent: chartAgent,
    name: "ChartGenerator",
  });
}

const tools = [tavilySearchTool, chartTool];
const toolNodes = new ToolNode<typeof AgentState.State>(tools);

function router(state: typeof AgentState.State) {
  const messages = state.messages;
  const lastMessage = messages[messages.length - 1] as AIMessage;

  if (lastMessage?.tool_calls && lastMessage.tool_calls.length > 0) {
    // The previous agent is invoking a tool
    return "call_tool";
  }
  if (
    typeof lastMessage.content === "string" &&
    lastMessage.content.includes("FINAL ANSWER")
  ) {
    // Any agent decided the work is done
    return "end";
  }
  return "continue";
}

const workflow = new StateGraph(AgentState)
  .addNode("Researcher", researchNode)
  .addNode("ChartGenerator", chartNode)
  .addNode("call_tool", toolNodes);

workflow.addConditionalEdges("Researcher", router, {
  continue: "Researcher",
  call_tool: "call_tool",
  end: END,
});

workflow.addConditionalEdges("ChartGenerator", router, {
  // We will transition to the other agent
  continue: "Researcher",
  call_tool: "call_tool",
  end: END,
});

workflow.addConditionalEdges("call_tool", (x) => x.sender, {
  ChartGenerator: "ChartGenerator",
  Researcher: "Researcher",
});

workflow.addEdge(START, "Researcher");

const graph = workflow.compile();

const streamResults = await graph.stream(
  {
    messages: [
      new HumanMessage({
        content: "Generate a bar chart of the US gdp over the past 3 years.",
      }),
    ],
  },
  { recursionLimit: 50 }
);

const prettifyOutput = (output: Record<string, any>) => {
  const keys = Object.keys(output);
  const firstItem = output[keys[0]];

  if ("messages" in firstItem && Array.isArray(firstItem.messages)) {
    const lastMessage = firstItem.messages[firstItem.messages.length - 1];
    console.dir(
      {
        type: lastMessage._getType(),
        content: lastMessage.content,
        tool_calls: lastMessage.tool_calls,
      },
      { depth: null }
    );
  }

  if ("sender" in firstItem) {
    console.log({
      sender: firstItem.sender,
    });
  }
};

for await (const output of await streamResults) {
  if (!output?.__end__) {
    prettifyOutput(output);
    console.log("----");
  }
}
