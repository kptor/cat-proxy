import { createAzure } from "@ai-sdk/azure";
import { generateText, streamText } from "ai";
import { z } from "zod";
import type { CoreSystemMessage, CoreUserMessage } from "ai";

const CAT_PROXY_PORT_STRING = process.env.CAT_PROXY_PORT;
if (!CAT_PROXY_PORT_STRING) {
  throw new Error("CAT_PROXY_PORT is not set");
}
const CAT_PROXY_PORT = parseInt(CAT_PROXY_PORT_STRING);

const server = Bun.serve({
  port: CAT_PROXY_PORT,
  async fetch(req) {
    const url = new URL(req.url);
    if (url.pathname === "/completions") {
      if (req.method !== "POST") {
        return new Response("Method Not Allowed", { status: 405 });
      }
      try {
        const body = await req.json();
        const parseResult = PayloadSchema.safeParse(body);
        if (!parseResult.success) {
          return new Response(
            JSON.stringify({ error: "Invalid request body", details: parseResult.error.errors }),
            { status: 400, headers: { "Content-Type": "application/json" } }
          );
        }
        const { messages, model: modelPayload } = parseResult.data;
        const deployment = modelPayload.uri;
        if (typeof deployment !== "string" || !isDeployment(deployment)) {
          return new Response(JSON.stringify({ error: "Invalid deployment in model.uri" }), { status: 400, headers: { "Content-Type": "application/json" } });
        }
        const coreMessages: (CoreSystemMessage | CoreUserMessage)[] = messages.map((msg) => {
          if (msg.role === "system") {
            // For system, join all text parts into a single string
            const text = msg.parts.map((p) => p.text).join(" ");
            return { role: "system", content: text } as CoreSystemMessage;
          } else {
            // For user, pass the array of parts
            return { role: "user", content: msg.parts } as CoreUserMessage;
          }
        });
        const modelInstance = model(deployment as CompletionDeployment);
        const result = streamText({
          model: modelInstance,
          system: SYSTEM_PROMPT,
          messages: coreMessages,
        });
        return result.toDataStreamResponse();
      } catch (err) {
        console.error("Error processing completion request:", JSON.stringify(err, null, 2));
        return new Response(JSON.stringify({ 
          error: "Failed to process request",
          details: err instanceof Error ? err.message : "Unknown error"
        }), { 
          status: 500, 
          headers: { "Content-Type": "application/json" } 
        });
      }
    }
    return new Response("Not Found", { status: 404 });
  },
});

console.log("Listening on http://localhost:" + CAT_PROXY_PORT);

// TODO: move this to a separate file

const SYSTEM_PROMPT = `You are a helpful assistant`;

function makeAzureProviderInstance() {
    const AZURE_AI_RESOURCE_NAME = process.env.AZURE_AI_RESOURCE_NAME;
    const AZURE_AI_API_KEY = process.env.AZURE_AI_API_KEY;
  
    if (!AZURE_AI_RESOURCE_NAME || !AZURE_AI_API_KEY) {
      throw new Error(
        "Missing Azure AI resource name or API key in environment variables",
      );
    }
  
    const azure = createAzure({
      resourceName: AZURE_AI_RESOURCE_NAME,
      apiKey: AZURE_AI_API_KEY,
    });
  
    return azure;
}
  
export type CompletionDeployment = "gpt-4.1" | "gpt-4o" | "gpt-4o-mini" | "gpt-4.1-nano"
  
export function isDeployment(s: string): boolean {
    return (
      s === "gpt-4.1" || s === "gpt-4o" || s === "gpt-4o-mini" || s === "o3-mini"
    );
}
  
export function model(deployment: CompletionDeployment) {
    const azure = makeAzureProviderInstance();
    return azure(deployment);
}

const MessageContentSchema = z.object({
  type: z.literal("text"),
  text: z.string(),
});

const MessageSchema = z.object({
  role: z.enum(["system", "user"]),
  parts: z.array(MessageContentSchema),
});

const ModelSchema = z.object({
  uri: z.string(),
  params: z.record(z.any()),
});

const PayloadSchema = z.object({
  messages: z.array(MessageSchema),
  model: ModelSchema,
});