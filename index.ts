import dotenv from "dotenv";
dotenv.config();

import { createAzure } from "@ai-sdk/azure";
import { generateText, streamText } from "ai";
import { z } from "zod";
import type { CoreSystemMessage, CoreUserMessage } from "ai";
import http from "http";
import { URL } from "url";
import { verifyToken } from "@clerk/backend";

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

// Helper function to extract bearer token from Authorization header
function extractBearerToken(authHeader: string | undefined): string | null {
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return null;
  }
  return authHeader.substring(7); // Remove 'Bearer ' prefix
}

// Helper function to verify authentication
async function verifyAuthentication(req: http.IncomingMessage): Promise<{ success: boolean; error?: string }> {
  try {
    const authHeader = req.headers.authorization;
    const token = extractBearerToken(authHeader);
    
    if (!token) {
      return { success: false, error: "Missing or invalid Authorization header" };
    }

    // Validate token with Clerk's OAuth userinfo endpoint
    const userInfoUrl = process.env.CLERK_OAUTH_USER_INFO_URL;
    if (!userInfoUrl) {
      console.error('CLERK_OAUTH_USER_INFO_URL environment variable is not set');
      return { success: false, error: "Server configuration error" };
    }

    const response = await fetch(userInfoUrl, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      }
    });

    if (!response.ok) {
      if (response.status === 401) {
        return { success: false, error: "Invalid or expired access token" };
      }
      
      console.error(`Clerk userinfo request failed: ${response.status} ${response.statusText}`);
      return { success: false, error: "Authentication service unavailable" };
    }

    const userInfo = await response.json();
    console.log('User authenticated:', userInfo.sub); // Log user ID for debugging
    
    return { success: true };
  } catch (error) {
    console.error("Authentication verification error:", error);
    return { 
      success: false, 
      error: error instanceof Error ? error.message : "Internal server error during authentication" 
    };
  }
}

// Health check function to verify system components
async function checkHealth(): Promise<{ 
  status: 'healthy' | 'unhealthy';
  checks: Record<string, { 
    status: 'ok' | 'error';
    message?: string;
  }>;
  message?: string;
}> {
  const checks: Record<string, { status: 'ok' | 'error'; message?: string }> = {};

  // Check required environment variables
  const requiredEnvVars = [
    'PORT',
    'CLERK_OAUTH_USER_INFO_URL',
    'AZURE_AI_RESOURCE_NAME',
    'AZURE_AI_API_KEY'
  ];

  checks.environment = {
    status: 'ok'
  };

  for (const envVar of requiredEnvVars) {
    if (!process.env[envVar]) {
      checks.environment = {
        status: 'error',
        message: `Missing required environment variable: ${envVar}`
      };
      break;
    }
  }

  // Check Azure AI connection
  try {
    const azure = makeAzureProviderInstance();
    checks.azureConnection = {
      status: 'ok'
    };
  } catch (error) {
    checks.azureConnection = {
      status: 'error',
      message: error instanceof Error ? error.message : 'Failed to initialize Azure AI'
    };
  }

  // Check Clerk authentication service
  try {
    const response = await fetch(process.env.CLERK_OAUTH_USER_INFO_URL || '', {
      method: 'HEAD'
    });
    checks.clerkService = {
      status: response.ok ? 'ok' : 'error',
      message: response.ok ? undefined : `Clerk service returned status ${response.status}`
    };
  } catch (error) {
    checks.clerkService = {
      status: 'error',
      message: 'Failed to connect to Clerk service'
    };
  }

  // Determine overall status - only consider environment and Azure connection
  const criticalChecks = ['environment', 'azureConnection'];
  const isHealthy = criticalChecks.every(key => checks[key]?.status === 'ok');

  return {
    status: isHealthy ? 'healthy' : 'unhealthy',
    checks,
    message: isHealthy ? undefined : 'One or more critical checks failed'
  };
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url!, `http://localhost:${PORT}`);
  
  // Add health endpoint
  if (url.pathname === "/health") {
    try {
      const healthStatus = await checkHealth();
      res.writeHead(healthStatus.status === 'healthy' ? 200 : 503, { 
        "Content-Type": "application/json" 
      });
      res.end(JSON.stringify(healthStatus));
      return;
    } catch (error) {
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ 
        status: 'unhealthy',
        error: error instanceof Error ? error.message : 'Internal server error'
      }));
      return;
    }
  }

  if (url.pathname === "/v1/chat/completions") {
    if (req.method !== "POST") {
      res.writeHead(405, { "Content-Type": "text/plain" });
      res.end("Method Not Allowed");
      return;
    }
    
    // Verify authentication before processing the request
    const authResult = await verifyAuthentication(req);
    console.log(`Authentication result: ${authResult.success ? 'Success' : 'Failed'}${authResult.error ? ` - ${authResult.error}` : ''}`);
    
    if (!authResult.success) {
      res.writeHead(401, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ 
        error: "Unauthorized", 
        details: authResult.error 
      }));
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

      // console.log("messages:", JSON.stringify(messages, null, 2));

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
  
export type CompletionDeployment = "gpt-4.1" | "gpt-4.1-mini" | "gpt-4.1-nano" | "gpt-4o" | "gpt-4o-mini" | "o3-mini"
  
export function isDeployment(s: string): boolean {
    return (
      s === "gpt-4.1" || s === "gpt-4.1-mini" || s === "gpt-4.1-nano" || s === "gpt-4o" || s === "gpt-4o-mini" || s === "o3-mini"
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
  role: z.enum(["system", "user", "assistant"]),
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