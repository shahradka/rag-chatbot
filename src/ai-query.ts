
type Message = {
    id: string;
    role: string;
    content: string;
};

type ReqBody = {
    id: string;
    messages: Message[];
};

export const AIQueryAPI = async (request: Request, env, ctx) => {
    if (request.method !== "POST") {
        return new Response("Method not allowed", { status: 405 });
    }

    try {
        const payload = (await request.json()) as ReqBody;

            const messages = payload.messages || [] 

            
            const lastUserPrompt = [...messages].reverse().find(m => m.role === 'user')
            
            const queryEmbedding = await env.AI.run("@cf/baai/bge-base-en-v1.5", {
                text: [lastUserPrompt?.content]
            })

            const queryVectorNumber = Array.from(queryEmbedding.data[0]).map(num => Number(num));
    
            const allVectorsMatches = await env.MYCV_VECTOR_INDEX.query(queryVectorNumber, {
                topK: 3,
                returnValues: false,
                returnMetadata: "all",
            })
    
            const contextDocs = allVectorsMatches.matches.map((match) => match.metadata?.text).join("\n\n");
            
            const llmResponse = await env.AI.run("@cf/meta/llama-3.1-8b-instruct-fp8", {
                prompt: `You are an AI assistant helping with questions about a resume. 
                Use the following pieces of context to answer the user's question at the end. 
                If you don't know the answer, just say you don't know.
                
                Context:
                ${contextDocs}
                
                Question: ${messages}`,
                stream: true
            });
            
            return new Response(llmResponse as ReadableStream, 
                {
                    headers: {
                        "Content-Type": "text/event-stream",
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    },
                }
            )
        }
    catch(error: any){
        return new Response(JSON.stringify({ error: error.message }), {
            status: 500,
            headers: { "content-type": "application/json" },
        });
   }
}