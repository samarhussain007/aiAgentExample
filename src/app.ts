import dotenv from "dotenv";
dotenv.config();

import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { tavily } from "@tavily/tavily";
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
import { StringOutputParser } from "@langchain/core/output_parsers";
import { formatDocumentsAsString } from "langchain/util/document";

import { StateGraph } from "@langchain/langgraph";
import { TaskType } from "@google/generative-ai";

import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";
import { Document, DocumentInterface } from "@langchain/core/documents";
import { tool } from "@langchain/core/tools";

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

async function generate(
  state: typeof GraphState.State
): Promise<Partial<typeof GraphState.State>> {
  console.log("---GENERATE---");

  const prompt = await pull<ChatPromptTemplate>("rlm/rag-prompt");

  const { documents, question } = state;

  const ragChain = prompt.pipe(model).pipe(new StringOutputParser());

  const generation = await ragChain.invoke({
    context: formatDocumentsAsString(documents),
    question: question,
  });

  return {
    generation,
  };
}

async function gradeDocuments(
  state: typeof GraphState.State
): Promise<Partial<typeof GraphState.State>> {
  console.log("---CHECK RELEVANCE---");

  const gradeSchema = z
    .object({
      binaryScore: z
        .enum(["yes", "no"])
        .describe("Relevance score 'yes' or 'no'"),
    })
    .describe(
      "Grade the relevance of the retrieved documents to the question."
    );

  //@ts-ignore
  const llmWithTool = model.withStructuredOutput(gradeSchema, {
    name: "grade",
  });

  const prompt = ChatPromptTemplate.fromTemplate(
    `You are a grader assessing relevance of a retrieved document to a user question.
  Here is the retrieved document:

  {context}

  Here is the user question: {question}

  If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
  Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.`
  );

  // Chain
  const chain = prompt.pipe(llmWithTool);

  const filteredDocs: Array<DocumentInterface> = [];
  for await (const doc of state.documents) {
    const grade = await chain.invoke({
      context: doc.pageContent,
      question: state.question,
    });
    if (grade.binaryScore === "yes") {
      console.log("---GRADE: DOCUMENT RELEVANT---");
      filteredDocs.push(doc);
    } else {
      console.log("---GRADE: DOCUMENT NOT RELEVANT---");
    }
  }

  return {
    documents: filteredDocs,
  };
}

/**
 * Transform the query to produce a better question.
 *
 * @param {typeof GraphState.State} state The current state of the graph.
 * @param {RunnableConfig | undefined} config The configuration object for tracing.
 * @returns {Promise<Partial<typeof GraphState.State>>} The new state object.
 */

async function transformQuery(state: typeof GraphState.State) {
  console.log("---TRANSFORM QUERY---");

  // Pull in the prompt
  const prompt = ChatPromptTemplate.fromTemplate(
    `You are generating a question that is well optimized for semantic search retrieval.
  Look at the input and try to reason about the underlying sematic intent / meaning.
  Here is the initial question:
  \n ------- \n
  {question} 
  \n ------- \n
  Formulate an improved question: `
  );

  const chain = prompt.pipe(model).pipe(new StringOutputParser());

  const betterQuestion = await chain.invoke({ question: state.question });

  return {
    question: betterQuestion,
  };
}

/**
 * Web search based on the re-phrased question using Tavily API.
 *
 * @param {typeof GraphState.State} state The current state of the graph.
 * @param {RunnableConfig | undefined} config The configuration object for tracing.
 * @returns {Promise<Partial<typeof GraphState.State>>} The new state object.
 */
async function webSearch(
  state: typeof GraphState.State
): Promise<Partial<typeof GraphState.State>> {
  console.log("---WEB SEARCH---");

  const tvly = tavily({ apiKey: process.env.TAVILY_API_KEY });
  //@ts-ignore
  const tavilySearchTool = tool(
    async ({ query }: { query: string }) => {
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
      description: "Use this tool to search information from the web.",
    }
  );
  const docs = await tavilySearchTool.invoke({
    query: state.question,
  });
  const webResults = new Document({ pageContent: docs });
  const newDocuments = state.documents.concat(webResults);

  return {
    documents: newDocuments,
  };
}
