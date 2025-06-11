import dotenv from "dotenv";
dotenv.config();

import { createAzure } from "@ai-sdk/azure";
import { generateText, streamText } from "ai";
import { z } from "zod";
import type { CoreSystemMessage, CoreUserMessage } from "ai";
import http from "http";
import { URL } from "url";

if (!process.env.PORT) {
  throw new Error("PORT is not set");
}
const PORT = parseInt(process.env.PORT);

// Helper function to parse JSON from request body
async function parseRequestBody(req: http.IncomingMessage): Promise<any> {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', chunk => {
      body += chunk.toString();
    });
    req.on('end', () => {
      try {
        const parsed = JSON.parse(body);
        resolve(parsed);
      } catch (error) {
        reject(error);
      }
    });
    req.on('error', reject);
  });
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url!, `http://localhost:${CAT_PROXY_PORT}`);
  
  if (url.pathname === "/v1/chat/completions") {
    if (req.method !== "POST") {
      res.writeHead(405, { "Content-Type": "text/plain" });
      res.end("Method Not Allowed");
      return;
    }
    
    try {
      const body = await parseRequestBody(req);
      const parseResult = PayloadSchema.safeParse(body);
      if (!parseResult.success) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ 
          error: "Invalid request body", 
          details: parseResult.error.errors 
        }));
        return;
      }
      
      const { messages, model: modelPayload } = parseResult.data;
      const deployment = modelPayload.uri;
      if (typeof deployment !== "string" || !isDeployment(deployment)) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "Invalid deployment in model.uri" }));
        return;
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
      
      // Convert the AI SDK stream to a Node.js response
      const stream = result.toDataStreamResponse();
      const streamBody = stream.body;
      
      if (streamBody) {
        // Set appropriate headers for streaming
        res.writeHead(200, {
          "Content-Type": "text/plain; charset=utf-8",
          "Transfer-Encoding": "chunked",
          "Cache-Control": "no-cache",
          "Connection": "keep-alive"
        });
        
        // Pipe the stream to the response
        const reader = streamBody.getReader();
        const pump = async (): Promise<void> => {
          try {
            const { done, value } = await reader.read();
            if (done) {
              res.end();
              return;
            }
            res.write(value);
            return pump();
          } catch (error) {
            console.error("Stream error:", error);
            res.end();
          }
        };
        await pump();
      } else {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "Failed to create stream" }));
      }
    } catch (err) {
      console.error("Error processing completion request:", JSON.stringify(err, null, 2));
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ 
        error: "Failed to process request",
        details: err instanceof Error ? err.message : "Unknown error"
      }));
    }
    return;
  }
  
  res.writeHead(404, { "Content-Type": "text/plain" });
  res.end("Not Found");
});

server.listen(PORT, () => {
  console.log("Listening on " + PORT);
});

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