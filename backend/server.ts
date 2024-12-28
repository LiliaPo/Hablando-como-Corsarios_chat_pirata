import { Hono } from "hono";
import { config } from "dotenv";
import { ChatGroq } from "@langchain/groq";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { readFile } from "fs";
import { join } from "path";
import {
	ChatPromptTemplate,
	MessagesPlaceholder,
} from "@langchain/core/prompts";
import { serve } from '@hono/node-server'
import { serveStatic } from '@hono/node-server/serve-static';

type Message = {
	content: string;
	isUser: boolean;
};

config();
const app = new Hono({});

app.use('/frontend/*', serveStatic({ root: './' }));
app.use('/images/*', serveStatic({ root: './' }));

const model = new ChatGroq({
	apiKey: process.env.GROQ_API_KEY || "",
	maxTokens: undefined,
	model: "mixtral-8x7b-32768",
	temperature: 0.7,
});

// Ruta para servir el archivo HTML
app.get("/", async (c) => {
	console.log('📄 Solicitud recibida en la ruta principal');
	// Leer el contenido del archivo index.html
	const htmlPath = join(process.cwd(), "frontend", "index.html");
	const htmlContent = await new Promise<string>((resolve, reject) => {
		readFile(htmlPath, "utf8", (err, data) => {
			if (err) reject(err);
			else resolve(data);
		});
	});

	// Establecer el tipo de contenido como HTML
	c.header("Content-Type", "text/html");

	// Devolver el contenido del archivo HTML
	return c.body(htmlContent);
});

app.post("/chat", async (c) => {
	console.log('💬 Nueva solicitud de chat recibida');
	const { messages } = (await c.req.json()) as {
		messages: Message[];
	};

	try {
		const parsedMessages = messages.map((msg) =>
			msg.isUser
				? new HumanMessage({
					content: msg.content,
				})
				: new AIMessage({
					content: msg.content,
				})
		);

		const prompt = ChatPromptTemplate.fromMessages([
			[
				"system",
				"INSTRUCCIONES: " +
				"1. Eres un asistente pirata que responde en español. " +
				"2. Responde de forma natural y educada, pero sin excesos de formalidad. " +
				"3. A un saludo, responde brevemente, pero en tono de pirata amable.(ejemplo: 'Hola, ¿en qué puedo ayudarte?'). " +
				"4. Proporciona respuestas directas y útiles. " +
				"5. No uses frases artificiales ni excesivamente entusiastas. " +
				"6. Mantén un tono profesional y cercano, como un colega pirata."
			],
			new MessagesPlaceholder("chat_history"),
		]);

		const chain = prompt.pipe(model);

		const completion = await chain.invoke({ 
			chat_history: parsedMessages
		});

		return c.json({ response: completion.content });
	} catch (error) {
		console.error("Error al procesar la solicitud de chat:", error);
		return c.json({ error: "Error al procesar la solicitud de chat" }, 500);
	}
});

const port = 3000;
try {
	serve({ 
		fetch: app.fetch, 
		port 
	});
	console.log(`🚀 Servidor corriendo en http://localhost:${port}`);
} catch (err) {
	console.error('❌ Error al iniciar el servidor:', err);
	process.exit(1);
}
// export default { fetch: app.fetch, port: 3030 };
