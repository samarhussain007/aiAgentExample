import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatOllama } from "@langchain/ollama";
import { ChatPromptTemplate } from "@langchain/core/prompts";

const model = new ChatOllama({
  model: "llama2",
});

/*
const messages = [
  new SystemMessage("Translate the following English into Italian"),
  new HumanMessage("hi!"),
];

const stream = await model.stream(messages);

const chunks = [];
for await (const chunk of stream) {
  chunks.push(chunk);
  console.log(`${chunk.content}|`);
}
*/

const systemTemplate = "Translate the following English into {language}";

const promptTemplate = ChatPromptTemplate.fromMessages([
  ["system", systemTemplate],
  ["user", "{text}"],
]);
const promptValue = await promptTemplate.invoke({
  language: "italian",
  text: "hi!",
});

const response = await model.invoke(promptValue);

console.log(response.content);
