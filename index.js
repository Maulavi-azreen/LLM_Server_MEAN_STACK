const express = require('express');
const cors = require('cors');
const { PromptTemplate } = require('@langchain/core/prompts');
const { StringOutputParser } = require('@langchain/core/output_parsers');
const { ChatMistralAI } = require('@langchain/mistralai');
const dotenv = require('dotenv');

dotenv.config();

const app = express();
app.use(cors({
    origin: 'http://localhost:4200', // Adjust based on your Angular app's URL
    methods: ['POST', 'GET'],
    allowedHeaders: ['Content-Type'],
}));
app.use(express.json());

const model = new ChatMistralAI({
    apiKey: process.env.MISTRAL_API_KEY,
    modelName: "mistral-tiny", // or any other model name you're using
    temperature: 0,
});

// Configure prompt
const DeepThinkingPrompt = PromptTemplate.fromTemplate(
    `**Deep Thinking Analysis Task**
    Analyze user's query thoroughly. Consider:
    1. Primary intent and underlying needs
    2. Potential ambiguities or missing context
    3. Required knowledge domains
    4. Response strategy
    
    userQuery: {query}
    Step-by-step Analysis:`
);

const finalResponsePrompt = PromptTemplate.fromTemplate(
    `**Response Generation**
    Based on the analysis below, craft a comprehensive response.

    Analysis: {analysis}
    Original Query: {query} 
    Response:`
);

// Chains creation
const deepthinkingChain = DeepThinkingPrompt.pipe(model).pipe(new StringOutputParser());
const finalResponseChain = finalResponsePrompt.pipe(model).pipe(new StringOutputParser());

// Store latest query (this is just for tracking queries, can be replaced with a database)
let latestQuery = "";

// Endpoint to receive query (POST)
app.post('/chat', (req, res) => {
    const { query } = req.body;
    if (!query) {
        return res.status(400).json({ message: 'Query is required' });
    }

    latestQuery = query;
    res.json({ message: 'Query received, processing started.' });
});

// Endpoint to stream responses (GET)
app.get('/chat-stream', async (req, res) => {
    if (!latestQuery) {
        return res.status(400).json({ message: 'No query available, send a query first.' });
    }

    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders();

    try {
        let analysis = "";
        const deepThinkingStream = await deepthinkingChain.stream({ query: latestQuery });

        for await (const chunk of deepThinkingStream) {
            res.write(`event: deep-thinking\ndata: ${JSON.stringify(chunk)}\n\n`);
            analysis += chunk;
        }

        const finalResponseStream = await finalResponseChain.stream({ analysis, query: latestQuery });

        for await (const chunk of finalResponseStream) {
            res.write(`event: final-response\ndata: ${JSON.stringify(chunk)}\n\n`);
        }

        res.write('event: end\ndata: \n\n');
        console.log(`Processed query: ${latestQuery}`);
    } catch (error) {
        console.error(error);
        res.write(`event: error\ndata: ${JSON.stringify({ message: "Error processing request" })}\n\n`);
    } finally {
        res.end();
        console.log('Client disconnected');
    }
});

// Start server
const PORT = process.env.PORT || 5001;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
