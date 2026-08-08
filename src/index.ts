import { AIQueryAPI } from "./ai-query";

export { RAGWorkflow } from "./worker";

type MY_QUEUE_MESSAGE = {
  readonly id: string;
  readonly timestamp: Date;
  readonly body: Body;
  readonly attempts: number;
  ack(): void;
  retry(options?: QueueRetryOptions): void;
}


export default {
  async fetch (request, env, ctx) {
      
    return await AIQueryAPI(request, env, ctx);

  },
  async queue(batch: { messages: MY_QUEUE_MESSAGE[]; }, env) {
  for (const message of batch.messages) {
      const r2Event = message.body;
        const instance = await env.RAG_WORKFLOW.create({params: {fileNameKey: r2Event.object?.key}});
        console.log("R2 Payload:", JSON.stringify(r2Event, null, 2));
        console.log("Action Type:", r2Event.action);
        console.log("Detected File Name:", r2Event.object?.key);
        console.log(`Successfully vectorized and saved ${instance.status.output.insertedVectorsCount} chunks into Vectorize!`)
        message.ack();
    }
  }
};