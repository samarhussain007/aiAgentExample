import dotenv from "dotenv";
dotenv.config();

import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createRetrieverTool } from "langchain/tools/retriever";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { pull } from "langchain/hub";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";

import { Annotation, END, Graph, START } from "@langchain/langgraph";
import { AIMessage, BaseMessage, HumanMessage } from "@langchain/core/messages";
import { z } from "zod";
import { ChatPromptTemplate } from "@langchain/core/prompts";

import { StateGraph } from "@langchain/langgraph";
import { TaskType } from "@google/generative-ai";

import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";
import { DocumentInterface } from "@langchain/core/documents";

const urls = [
  "https://lilianweng.github.io/posts/2023-06-23-agent/",
  "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
  "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
];

const model = new ChatOllama({
  model: "llama3.1:8b",
  temperature: 0,
});

const docs = await Promise.all(
  urls.map(async (url) => {
    const loader = new CheerioWebBaseLoader(url, {
      selector: ".post-content",
    });
    const docs = await loader.load();
    return docs;
  })
);

const allDocs = docs.flat();

const textsplitters = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});

const docsSplit = await textsplitters.splitDocuments(allDocs);

const vectorStore = await MemoryVectorStore.fromDocuments(
  docsSplit,
  new OllamaEmbeddings({
    model: "llama3.1:8b",
  })
);

const retriever = vectorStore.asRetriever();

const GraphState = Annotation.Root({
  documents: Annotation<DocumentInterface[]>({
    reducer: (x, y) => y ?? x ?? [],
  }),
  question: Annotation<string>({
    reducer: (x, y) => y ?? x ?? "",
  }),
  generation: Annotation<string>({
    reducer: (x, y) => y ?? x,
  }),
});

/**
 * Retrieve documents
 *
 * @param {typeof GraphState.State} state The current state of the graph.
 * @param {RunnableConfig | undefined} config The configuration object for tracing.
 * @returns {Promise<Partial<typeof GraphState.State>>} The new state object.
 */

async function retrieve(
  state: typeof GraphState.State
): Promise<Partial<typeof GraphState.State>> {
  console.log("Retrieving documents...");
  const { question } = state;
  const documents = await retriever
    .withConfig({
      runName: "FetchRelevantDocs",
    })
    .invoke(question);
  return {
    documents,
  };
}
