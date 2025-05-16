import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

import { toolsCondition } from "@langchain/langgraph/prebuilt";

import {
  AIMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import {
  MemorySaver,
  MessagesAnnotation,
  StateGraph,
} from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { z } from "zod";

import { BaseMessage, isAIMessage } from "@langchain/core/messages";
import { threadId } from "worker_threads";

const model = new ChatOllama({
  model: "mistral",
  temperature: 0.3,
});

const embeddings = new OllamaEmbeddings({
  model: "mistral",
});

const vectorStore = new MemoryVectorStore(embeddings);

const pTagSelector = "p";
const cheerioLoader = new CheerioWebBaseLoader(
  "https://lilianweng.github.io/posts/2023-06-23-agent/",
  {
    selector: pTagSelector,
  }
);

const docs = await cheerioLoader.load();

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});

const allSplits = await splitter.splitDocuments(docs);

await vectorStore.addDocuments(allSplits);

const retrieveSchema = z.object({ query: z.string() });

// Define application steps
const retrieve = tool(
  async ({ query }) => {
    const retrievedDocs = await vectorStore.similaritySearch(query, 2);
    const serialized = retrievedDocs.map(
      (el) => `Source ${el.metadata.source}\nContent: ${el.pageContent}`
    );
    return [serialized, retrievedDocs];
  },
  {
    name: "retrieve",
    description:
      "Use this tool only to look up specific technical or knowledge-based questions.",
    schema: retrieveSchema,
    responseFormat: "content_and_artifact",
  }
);
// Step 1: Generate an AIMessage that may include a tool-call to be sent.
async function queryOrRespond(state: typeof MessagesAnnotation.State) {
  //This just provides the intent to the llm, saying that if you need to retrieve something you can call retrieve tool
  const llmWithTools = model.bindTools([retrieve]);
  const response = await llmWithTools.invoke(state.messages);

  console.log("HERE I KNOW WHATS HAPPENING ", state.messages);
  //MessageState appends messages to state instead of overwritting
  return { messages: [response] };
}

// Step 2: Set up the tool executor node. It runs the tool only when the LLM requests it.
const tools = new ToolNode([retrieve]);

// Step 3: Generate a response using the retrieved content.
const generate = async (state: typeof MessagesAnnotation.State) => {
  let recentToolMessages = [];
  for (let i = state["messages"].length - 1; i >= 0; i--) {
    const message = state["messages"][i];
    if (message instanceof ToolMessage) {
      recentToolMessages.push(message);
    } else {
      break;
    }
  }

  let toolMessages = recentToolMessages.reverse();

  //Format into prompt
  const docsContent = toolMessages.map((doc) => doc.content).join("\n");

  const systemMessageContent =
    "You are an assistant for question-answering tasks. " +
    "Use the following pieces of retrieved context to answer " +
    "the question. If you don't know the answer, say that you " +
    "don't know. Use three sentences maximum and keep the " +
    "answer concise." +
    "\n\n" +
    `${docsContent}`;

  const conversationMessages = state.messages.filter(
    (message) =>
      message instanceof HumanMessage ||
      message instanceof SystemMessage ||
      message instanceof ToolMessage ||
      message instanceof AIMessage
  );

  const prompt = [
    new SystemMessage(systemMessageContent),
    ...conversationMessages,
  ];

  // Run
  const response = await model.invoke(prompt);
  return { messages: [response] };
};

// Compile application and test
const graphBuilder = new StateGraph(MessagesAnnotation)
  .addNode("queryOrRespond", queryOrRespond)
  .addNode("tools", tools)
  .addNode("generate", generate)
  .addEdge("__start__", "queryOrRespond")
  .addConditionalEdges("queryOrRespond", toolsCondition, {
    __end__: "__end__",
    tools: "tools",
  })
  .addEdge("tools", "generate")
  .addEdge("generate", "__end__");

const checkpointer = new MemorySaver();
const graphWithMemory = graphBuilder.compile({ checkpointer });

// Specify an ID for the thread
const threadConfig = {
  configurable: { thread_id: "abc123" },
  streamMode: "values" as const,
};

const prettyPrint = (message: BaseMessage) => {
  let txt = `[${message.getType()}]: ${message.content}`;
  if (isAIMessage(message) && message.tool_calls?.length) {
    const tool_calls = (message as AIMessage)?.tool_calls
      ?.map((tc) => `- ${tc.name}(${JSON.stringify(tc.args)})`)
      .join("\n");
    txt += ` \nTools: \n${tool_calls}`;
  }
  console.log(txt);
};

let inputs1 = {
  messages: [
    new SystemMessage(
      "You are a helpful assistant. If the user greets you (e.g., says 'hello', 'hi', 'hey'), respond with a short, friendly greeting. Do not suggest tools or offer examples unless the user asks a specific question."
    ),
    new HumanMessage("Hello"),
  ],
};

let inputs2 = {
  messages: [
    new SystemMessage(`
You are a helpful assistant. When the user asks a technical question, respond by calling the \`retrieve\` tool using the structured tool call system, not as plain text.

Example tool call (do this, not just write it):
retrieve({ "query": "Common methods of self-reflection in autonomous agents" })

Do not describe the tool. Just invoke it when needed.
`),

    new HumanMessage("How does the Self-Reflective agent architecture work?"),
  ],
};

let inputs4 = {
  messages: [
    { role: "user", content: "Can you look up some common ways of doing it?" },
  ],
};

for await (const step of await graphWithMemory.stream(inputs2, threadConfig)) {
  const lastMessage = step.messages[step.messages.length - 1];
  prettyPrint(lastMessage);
  console.log("-----\n");
}

for await (const step of await graphWithMemory.stream(inputs4, threadConfig)) {
  const lastMessage = step.messages[step.messages.length - 1];
  prettyPrint(lastMessage);
  console.log("-----\n");
}
