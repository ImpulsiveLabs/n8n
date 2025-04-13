import { spawn } from 'child_process';
import fs from 'fs';
import type {
	INodeType,
	INodeTypeDescription,
	INodeExecutionData,
	IExecuteFunctions,
} from 'n8n-workflow';
import { NodeConnectionType } from 'n8n-workflow';
import path from 'path';
import { promisify } from 'util';

// Helper function to unlink files
const unlinkFile = promisify(fs.unlink);

// Function to run the Python script
const runPythonScript = async (
	scriptPath: string,
	inputData: any,
	args: Record<string, any>,
): Promise<any> => {
	return await new Promise(async (resolve, reject) => {
		try {
			// Write input data to a temporary file
			const tempFilePath = path.join(__dirname, `input_${Date.now()}.json`);
			await promisify(fs.writeFile)(tempFilePath, JSON.stringify(inputData));

			const envVars = Object.fromEntries(
				Object.entries(args).map(([key, value]) => [
					`ARG${key}`,
					typeof value === 'string' ? value : JSON.stringify(value),
				]),
			);

			const pythonProcess = spawn('python3', [scriptPath, tempFilePath], {
				env: { ...process.env, ...envVars },
				stdio: ['pipe', 'pipe', 'pipe'],
			});

			let data = '';

			// Capture Python stdout (output)
			pythonProcess.stdout.on('data', (chunk) => {
				data += chunk.toString();
			});

			// Capture Python stderr (errors)
			pythonProcess.stderr.on('data', (error) => {
				console.error(`Python error: ${error.toString().trim()}`);
			});

			// Handle process completion
			pythonProcess.on('close', async (code) => {
				// Log Python exit code
				console.log(`Python process exited with code ${code}`);

				// Delete the temporary file
				await unlinkFile(tempFilePath);

				// If process was successful
				if (code === 0) {
					try {
						const jsonData = JSON.parse(data.trim());
						resolve(jsonData);
					} catch (error) {
						console.error(`Failed to parse JSON output: ${error.message}`);
						reject('Failed to parse JSON output');
					}
				} else {
					reject(`Python process exited with code ${code}`);
				}
			});
		} catch (error) {
			reject(`Error preparing input data: ${error.message}`);
		}
	});
};

export class Cluster implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Cluster',
		name: 'cluster',
		group: ['transform'],
		icon: { light: 'file:Cluster.svg', dark: 'file:Cluster.dark.svg' },
		version: 1,
		description: 'Clusters data based on different criteria using a Python script',
		defaults: { name: 'ClusterNode' },
		inputs: [NodeConnectionType.Main],
		outputs: [NodeConnectionType.Main],
		properties: [
			{
				displayName: 'Sentence Transformer Model',
				name: 'model',
				type: 'string',
				default: 'all-MiniLM-L6-v2',
				required: true,
				description: 'Model for sentence transformer',
			},

			{
				displayName: 'Cluster Criteria',
				name: 'clusterCriteria',
				type: 'options',
				options: [
					{ name: 'Similarity', value: 'similarity' },
					{ name: 'Author', value: 'author' },
					{ name: 'Country', value: 'country' },
				],
				default: 'similarity',
				required: true,
			},
			{
				displayName: 'Number of Clusters',
				name: 'numClusters',
				type: 'string',
				default: '3-9',
				required: true,
			},
			{
				displayName: 'Link Criteria',
				name: 'linkCriteria',
				type: 'options',
				options: [
					{ name: 'Country', value: 'country' },
					{ name: 'Co-Authorship', value: 'coauthorship' },
					{ name: 'Custom Field', value: 'custom' },
				],
				default: 'country',
				required: true,
			},
			{
				displayName: 'Custom Field (if Selected)',
				name: 'customField',
				type: 'string',
				default: '',
				displayOptions: { show: { linkCriteria: ['custom'] } },
			},
		],
	};

	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		const items = this.getInputData();
		const returnData: INodeExecutionData[] = [];

		// Iterate through input items
		for (let i = 0; i < items.length; i++) {
			const model = this.getNodeParameter('model', i) as string;
			const clusterCriteria = this.getNodeParameter('clusterCriteria', i) as string;
			const numClusters = this.getNodeParameter('numClusters', i) as string;
			const linkCriteria = this.getNodeParameter('linkCriteria', i) as string;
			const customField =
				linkCriteria === 'custom' ? (this.getNodeParameter('customField', i) as string) : '';

			// Convert JSON input
			const inputData = JSON.stringify(items[i].json, (_, value) => {
				if (value === true) return 'True';
				if (value === false) return 'False';
				if (value === null) return 'None';
				return value;
			});

			// Define Python script path
			const pythonScriptPath = path.join(__dirname, 'cluster_creation.py');

			// Log before running the Python script
			console.log(`Input data for Python script: ${inputData}`);

			// Run Python script with file input
			try {
				const result = await runPythonScript(pythonScriptPath, inputData, {
					1: model,
					2: clusterCriteria,
					3: numClusters,
					4: linkCriteria,
					5: customField,
				});

				// Push the result into returnData
				returnData.push({ json: result });
			} catch (error) {
				console.error(`Error during Python script execution: ${error}`);
			}
		}

		// Return final data
		return [returnData];
	}
}
