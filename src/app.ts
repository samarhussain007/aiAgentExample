import dotenv from "dotenv";
dotenv.config();

import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { tavily } from "@tavily/core";
import { pull } from "langchain/hub";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { Annotation, END, START } from "@langchain/langgraph";
import { formatDocumentsAsString } from "langchain/util/document";
import { z } from "zod";

import { StateGraph } from "@langchain/langgraph";

import { Document, DocumentInterface } from "@langchain/core/documents";
import { tool } from "@langchain/core/tools";
import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";

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
      selector: ".post-content, .article-body, article",
    });
    const docs = await loader.load();
    return docs;
  })
);

const allDocs = docs.flat();

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 250,
});

const docsSplit = await textSplitter.splitDocuments(allDocs);

const vectorStore = await MemoryVectorStore.fromDocuments(
  docsSplit,
  new OllamaEmbeddings({
    model: "nomic-embed-text",
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

  generationVQuestionGrade: Annotation<string>({
    reducer: (x, y) => y ?? x,
  }),

  generationVDocumentsGrade: Annotation<string>({
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

  console.log("Retrieved documents:", question);
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

  console.log("---GENERATED ANSWER---", generation);

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
  Formulate an improved question: 
  NOTE: Just make sure to return the question without any other text or explanation.`
  );

  const chain = prompt.pipe(model).pipe(new StringOutputParser());

  const betterQuestion = await chain.invoke({ question: state.question });

  console.log("---TRANSFORMED QUESTION---", betterQuestion);

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

  console.log(state.question);

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
  const webResults = new Document({ pageContent: docs?.answer || "" });
  const newDocuments = state.documents.concat(webResults);

  return {
    documents: newDocuments,
  };
}

/**
 * Determines whether to generate an answer, or re-generate a question.
 *
 * @param {typeof GraphState.State} state The current state of the graph.
 * @returns {"transformQuery" | "generate"} Next node to call
 */
function decideToGenerate(state: typeof GraphState.State) {
  console.log("---DECIDE TO GENERATE---");

  const filteredDocs = state.documents;
  if (filteredDocs.length === 0) {
    // All documents have been filtered checkRelevance
    // We will re-generate a new query
    console.log("---DECISION: TRANSFORM QUERY---");
    return "transformQuery";
  }

  // We have relevant documents, so generate answer
  console.log("---DECISION: GENERATE---");
  return "generate";
}

/**
 * Determines whether the generation is grounded in the document.
 *
 * @param {typeof GraphState.State} state The current state of the graph.
 * @param {RunnableConfig | undefined} config The configuration object for tracing.
 * @returns {Promise<Partial<typeof GraphState.State>>} The new state object.
 */

async function generateGenerationVDocumentsGrade(
  state: typeof GraphState.State
): Promise<Partial<typeof GraphState.State>> {
  console.log("---GENERATE GENERATION vs DOCUMENTS GRADE---");
  //@ts-ignore
  const llmWithTool = model.withStructuredOutput(
    z
      .object({
        binaryScore: z
          .enum(["yes", "no"])
          .describe("Relevance score 'yes' or 'no'"),
      })
      .describe(
        "Grade the relevance of the retrieved documents to the question. Either 'yes' or 'no'."
      ),
    {
      name: "grade",
    }
  );

  const prompt = ChatPromptTemplate.fromTemplate(
    `You are a grader assessing whether an answer is grounded in / supported by a set of facts.
  Here are the facts:
  \n ------- \n
  {documents} 
  \n ------- \n
  Here is the answer: {generation}
  Give a binary score 'yes' or 'no' to indicate whether the answer is grounded in / supported by a set of facts.`
  );

  const chain = prompt.pipe(llmWithTool);
  const score = await chain.invoke({
    documents: formatDocumentsAsString(state.documents),
    generation: state.generation,
  });
  return {
    generationVDocumentsGrade: score.binaryScore,
  };
}

function gradeGenerationVDocuments(state: typeof GraphState.State) {
  console.log("---GRADE GENERATION vs DOCUMENTS---");

  const grade = state.generationVDocumentsGrade;
  if (grade === "yes") {
    console.log("---DECISION: SUPPORTED, MOVE TO FINAL GRADE---");
    return "supported";
  }

  console.log("---DECISION: NOT SUPPORTED, GENERATE AGAIN---");
  return "not supported";
}

async function generateGenerationVQuestionGrade(
  state: typeof GraphState.State
): Promise<Partial<typeof GraphState.State>> {
  console.log("---GENERATE GENERATION vs QUESTION GRADE---");
  //@ts-ignore
  const llmWithTool = model.withStructuredOutput(
    z
      .object({
        binaryScore: z
          .enum(["yes", "no"])
          .describe("Relevance score 'yes' or 'no'"),
      })
      .describe(
        "Grade the relevance of the retrieved documents to the question. Either 'yes' or 'no'."
      ),
    {
      name: "grade",
    }
  );

  const prompt = ChatPromptTemplate.fromTemplate(
    `You are a grader assessing whether an answer is relevant to a user question.
  Here is the user question:
  \n ------- \n
  {question} 
  \n ------- \n
  Here is the answer: {generation}
  Give a binary score 'yes' or 'no' to indicate whether the answer is relevant to the user question.`
  );

  const chain = prompt.pipe(llmWithTool);
  const score = await chain.invoke({
    question: state.question,
    generation: state.generation,
  });
  return {
    generationVQuestionGrade: score.binaryScore,
  };
}

function gradeGenerationVQuestion(state: typeof GraphState.State) {
  console.log("---GRADE GENERATION vs QUESTION---");

  const grade = state.generationVQuestionGrade;
  if (grade === "yes") {
    console.log("---DECISION: USEFUL---");
    return "useful";
  }

  console.log("---DECISION: NOT USEFUL---");
  return "not useful";
}

const workflow = new StateGraph(GraphState)
  // Define the nodes
  .addNode("retrieve", retrieve)
  .addNode("gradeDocuments", gradeDocuments)
  .addNode("generate", generate)
  .addNode(
    "generateGenerationVDocumentsGrade",
    generateGenerationVDocumentsGrade
  )
  .addNode("transformQuery", transformQuery)
  .addNode(
    "generateGenerationVQuestionGrade",
    generateGenerationVQuestionGrade
  );

// Build graph
workflow.addEdge(START, "retrieve");
workflow.addEdge("retrieve", "gradeDocuments");
workflow.addConditionalEdges("gradeDocuments", decideToGenerate, {
  transformQuery: "transformQuery",
  generate: "generate",
});
workflow.addEdge("transformQuery", "retrieve");
workflow.addEdge("generate", "generateGenerationVDocumentsGrade");
workflow.addConditionalEdges(
  "generateGenerationVDocumentsGrade",
  gradeGenerationVDocuments,
  {
    supported: "generateGenerationVQuestionGrade",
    "not supported": "generate",
  }
);

workflow.addConditionalEdges(
  "generateGenerationVQuestionGrade",
  gradeGenerationVQuestion,
  {
    useful: END,
    "not useful": "transformQuery",
  }
);

// Compile
const app = workflow.compile();

const inputs = {
  question: "Explain how the different types of agent memory work.",
};
const config = { recursionLimit: 50 };

const prettifyOutput = (output: Record<string, any>) => {
  const key = Object.keys(output)[0];
  const value = output[key];
  console.log(`Node: '${key}'`);
  if (key === "retrieve" && "documents" in value) {
    console.log(`Retrieved ${value.documents.length} documents.`);
  } else if (key === "gradeDocuments" && "documents" in value) {
    console.log(
      `Graded documents. Found ${value.documents.length} relevant document(s).`
    );
  } else {
    console.dir(value, { depth: null });
  }
};

for await (const output of await app.stream(inputs, config)) {
  prettifyOutput(output);
  console.log("\n---ITERATION END---\n");
}
