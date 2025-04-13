import { spawn } from 'child_process';
import type {
	INodeType,
	INodeTypeDescription,
	INodeExecutionData,
	IExecuteFunctions,
} from 'n8n-workflow';
import { NodeConnectionType } from 'n8n-workflow';
import path from 'path';

// Helper function to run the Python script
const runPythonScript = async (scriptPath: string, args: Record<string, any>): Promise<any> => {
	return await new Promise((resolve, reject) => {
		console.log(`Starting Python script: ${scriptPath}`);

		// Prepare environment variables
		const envVars = Object.fromEntries(
			Object.entries(args).map(([key, value]) => [
				`ARG${key}`,
				typeof value === 'string' ? value : JSON.stringify(value),
			]),
		);

		const pythonProcess = spawn('python3', [scriptPath], {
			env: { ...process.env, ...envVars },
			stdio: ['pipe', 'pipe', 'pipe'],
		});

		let data = '';

		// Capture standard output
		pythonProcess.stdout.on('data', (chunk) => {
			data += chunk.toString();
		});

		// Capture standard error
		pythonProcess.stderr.on('data', (error) => {
			console.error(`Python error: ${error.toString().trim()}`);
		});

		// Handle script close event
		pythonProcess.on('close', (code) => {
			console.log(`Python process exited with code ${code}`);
			if (code === 0) {
				// Parse JSON output
				try {
					const jsonData = JSON.parse(data.trim());
					resolve(jsonData);
				} catch (error) {
					console.error(`Failed to parse JSON data: ${error.message}`);
					reject('Failed to parse JSON data');
				}
			} else {
				reject(`Python process exited with code ${code}`);
			}
		});
	});
};

// Define the custom n8n node
export class OpenAlex implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'OpenAlex Fetcher',
		name: 'openAlex',
		group: ['transform'],
		icon: { light: 'file:OpenAlex.svg', dark: 'file:OpenAlex.dark.svg' },
		version: 1,
		description: 'Fetches data from OpenAlex using a Python script',
		defaults: { name: 'OpenAlexFetcher' },
		inputs: [NodeConnectionType.Main],
		outputs: [NodeConnectionType.Main],
		properties: [
			{
				displayName: 'Title and Abstract Query',
				name: 'query',
				type: 'string',
				default: 'business management visualization',
				required: true,
				description: 'Search query for the title and abstract',
			},
			{
				displayName: 'Relevant Terms (JSON)',
				name: 'relevantTerms',
				type: 'json',
				default: '{}',
				description: 'JSON object where each key maps to an array of relevant terms',
			},
			{
				displayName: 'Exclude Terms (JSON)',
				name: 'excludeTerms',
				type: 'json',
				default: '[]',
				description: 'JSON object where each key maps to an array of exclusion terms',
			},
		],
	};

	// Execute method for n8n node
	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		const items = this.getInputData();
		const returnData: INodeExecutionData[] = [];

		// Loop over all input items
		for (let i = 0; i < items.length; i++) {
			// Get parameters from node input
			const query = this.getNodeParameter('query', i, '') as string;
			const relevantTerms = this.getNodeParameter('relevantTerms', i, {}) as Record<
				string,
				string[]
			>;
			const excludeTerms = this.getNodeParameter('excludeTerms', i, {}) as Record<string, string[]>;

			// Path to the Python script
			const pythonScriptPath = path.join(__dirname, 'fetch_openalex_data.py');
			try {
				// Run the Python script with the provided parameters
				const result = await runPythonScript(pythonScriptPath, {
					1: query,
					2: relevantTerms,
					3: excludeTerms,
				});

				// Add result to return data
				returnData.push({ json: result });
			} catch (error) {
				// Handle Python script errors
				console.error(`Error during Python script execution: ${error}`);
				returnData.push({ json: { error: `Error executing Python script: ${error}` } });
			}
		}

		// Return the final result
		return [returnData];
	}
}
