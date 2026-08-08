import { WorkflowEntrypoint, WorkflowEvent, WorkflowStep } from "cloudflare:workers";
import { extractText } from "unpdf";
import { split } from "llm-splitter";

type Params = { fileNameKey?: string };

interface Env {
  MYCV_R2_BUCKET: R2Bucket;
  AI: any;
  MYCV_VECTOR_INDEX: any;
}

export class RAGWorkflow extends WorkflowEntrypoint<Env, Params> {
  async run(event: WorkflowEvent<{ fileNameKey: string }>, step: WorkflowStep) {
    const filesKeys = event.payload.fileNameKey;

    if (!filesKeys) {
      return { error: "No files found in R2 bucket.", status: 404 };
    }

    const extractedText = await step.do("Retrieving file and extracting text", async () => {
      const object = await this.env.MYCV_R2_BUCKET.get(filesKeys);
      if (!object) {
        throw new Error(`File not found: ${filesKeys}`);
      }
      const buffer = await object.arrayBuffer();
      const { text } = await extractText(buffer);
      
      return text; 
    });

    const vectorsToUpsert = await step.do("Processing Text and Generating Vectors", async () => {
      const chunks = split(extractedText, {
        chunkStrategy: 'paragraph',
      });

      const vectors = [];
      for (let i = 0; i < chunks.length; i++) {
        let chunkText = chunks[i].text;
        if (Array.isArray(chunkText)) {
          chunkText = chunkText.join("\n");
        }

        if (!chunkText || !chunkText.trim()) continue;

        const embeddingResponse = await this.env.AI.run("@cf/baai/bge-base-en-v1.5", {
          text: [chunkText]
        });

        const rawVector = embeddingResponse.data[0];
        if (!rawVector) continue;

        const cleanVectorValues = Array.from(rawVector).map(num => Number(num));

        vectors.push({
          id: `${filesKeys}-chunk-${i}`,
          values: cleanVectorValues,
          metadata: { 
            text: chunkText, 
            source: filesKeys 
          }
        });
      }
      return vectors;    
    });

    await step.do("Upserting Vectors to Index", async () => {
      if (vectorsToUpsert && vectorsToUpsert.length > 0) {
        await this.env.MYCV_VECTOR_INDEX.insert(vectorsToUpsert);
      }
    });

    return {
      success: true,
      insertedVectorsCount: vectorsToUpsert.length
    };
  }
}