import { readFileSync, existsSync } from 'fs';
import type {
	INodeType,
	INodeTypeDescription,
	INodeExecutionData,
	IExecuteFunctions,
	IHttpRequestOptions,
	IHttpRequestMethods,
} from 'n8n-workflow';
import { NodeConnectionType, ApplicationError } from 'n8n-workflow';
import type { WorkResponse } from 'openalex-ts';
import { SourceType } from 'openalex-ts';
import { OpenAlex as OpenAlexInstance } from 'openalex-ts';
import { v4 as uuidv4 } from 'uuid';

function parseTSV(fileContent: string): Array<Record<string, string>> {
	const lines = fileContent.split('\n').filter(line => line.trim());
	const headers = lines[0].split('\t').map(header => header.trim());
	return lines.slice(1).map(line => {
		const values = line.split('\t').map(value => value.trim());
		const record: Record<string, string> = {};
		headers.forEach((header, index) => {
			record[header] = values[index] || '';
		});
		return record;
	});
}

function createInvertedIndex(abstract: string): Record<string, number[]> {
	if (!abstract) return {};
	const words = abstract.split(/\s+/).filter(Boolean);
	const index: Record<string, number[]> = {};
	words.forEach((word, i) => {
		word = word.replace(/[^\w\s-]/g, '').toLowerCase();
		if (word) {
			index[word] = index[word] || [];
			index[word].push(i);
		}
	});
	return index;
}

function getAuthorLastname(name: string, isTSV = false): string {
	if (!name) return '';
	if (isTSV) {
		const parts = name.split(',').map(part => part.trim()).filter(Boolean);
		return parts[0].toLowerCase();
	}
	const words = name.split(/\s+/).filter(Boolean);
	return words.length ? words[words.length - 1].toLowerCase() : '';
}

async function sendHttpPost(
	this: IExecuteFunctions,
	article: { title: string; authors: string },
	journal: string,
	abstract: string,
	serverUrl: string
): Promise<any> {
	const payload = [{
		title: article.title,
		abstract_inverted_index: createInvertedIndex(abstract),
		inverted: true,
		referenced_works: [],
		journal_display_name: journal
	}];
	const requestOptions: IHttpRequestOptions = {
		method: 'POST' as IHttpRequestMethods,
		url: `${serverUrl}/invocations`,
		headers: { 'Content-Type': 'application/json' },
		body: payload,
		json: true,
		timeout: 120000
	};
	try {
		return await this.helpers.httpRequest(requestOptions);
	} catch (error) {
		return { error: (error as Error).message };
	}
}

async function searchOpenAlex(title: string, authors: string): Promise<{ found: boolean; work?: WorkResponse }> {
	try {
		const openAlex = OpenAlexInstance.getInstance({
			openalexUrl: 'https://api.openalex.org',
			email: 'test-email@outlook.com',
			userAgent: 'openalex-ts/1.0.0',
			maxRetries: 3,
			retryBackoffFactor: 0.2,
			retryHttpCodes: [429, 500, 503],
		});
		const filters = { 'title_and_abstract.search': title };
		const response = await openAlex
			.getWork()
			.filter(filters)
			.sort('relevance_score', 'desc')
			.get() as WorkResponse[];

		const tsvAuthorLastnames = authors
			.split(';')
			.map(author => getAuthorLastname(author.trim(), true))
			.filter(Boolean);

		for (const work of response) {
			if (work.title && work.title.toLowerCase() === title.toLowerCase()) {
				const openAlexAuthorLastnames = (work.authorships || [])
					.map(authorship => getAuthorLastname(authorship.author?.display_name || '', false))
					.filter(Boolean);

				if (tsvAuthorLastnames.some(tsvLastname =>
					openAlexAuthorLastnames.some(openAlexLastname => openAlexLastname.includes(tsvLastname))
				)) {
					return { found: true, work };
				}
			}
		}
		return { found: false };
	} catch (error) {
		return { found: false };
	}
}

function constructWorkResponse(record: Record<string, string>, postResponse: any): WorkResponse {
	const id = `https://openalex.org/W${uuidv4().replace(/-/g, '')}`;
	const authors = record.AU ? record.AU.split(';').map(a => a.trim()).filter(Boolean) : ['Unknown Author'];
	const publicationYear = parseInt(record.PY, 10) || 2021;
	const topics = Array.isArray(postResponse) && postResponse[0] ? postResponse[0] : [];
	const primaryTopic = topics[0] || null;

	return {
		id,
		display_name: record.TI,
		title: record.TI,
		publication_year: publicationYear,
		cited_by_count: parseInt(record.TC, 10) || 0,
		counts_by_year: [],
		created_date: null,
		authorships: authors.map(author => ({
			author: { display_name: author }
		})) as any[],
		primary_topic: primaryTopic ? {
			id: `https://openalex.org/T${primaryTopic.topic_id}`,
			display_name: primaryTopic.topic_label.replace(/^\d+:\s*/, '').trim()
		} : null as any,
		concepts: topics.map((topic: any) => ({
			id: `https://openalex.org/T${topic.topic_id}`,
			display_name: topic.topic_label.replace(/^\d+:\s*/, '').trim(),
			score: topic.topic_score
		})),
		keywords: [],
		primary_location: {
			is_accepted: false,
			is_oa: false,
			is_published: false,
			landing_page_url: '',
			source: {
				display_name: record.SO,
				id: '',
				issn_l: '',
				issn: [],
				host_organization_lineage: [],
				host_organization_name: '',
				is_core: false,
				is_oa: false,
				is_in_doaj: false,
				type: SourceType.EDUCATION
			}
		} as any
	} as any;
}

export class Wos implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Web of Science Fetcher',
		name: 'wos',
		group: ['transform'],
		icon: { light: 'file:Wos.svg', dark: 'file:Wos.dark.svg' },
		version: 1,

		description: 'Processes Web of Science TSV files and fetches topics from a classifier server, combining with OpenAlex input',
		defaults: { name: 'Wos' },
		inputs: [NodeConnectionType.Main],
		outputs: [NodeConnectionType.Main],
		properties: [
			{
				displayName: 'TSV File Path',
				name: 'tsvFilePath',
				type: 'string',
				default: '/home/node/extras/Articles.txt',
				description: 'Path to the Web of Science TSV file. If not provided or file does not exist, processes input data.'
			},
			{
				displayName: 'Classifier Server URL',
				name: 'classifierServerUrl',
				type: 'string',
				default: 'http://openalexclassifier:8080',
				required: true,
				description: 'URL of the server for topic classification (e.g., http://localhost:8080)'
			}
		]
	};

	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		const items = this.getInputData();
		const returnData: INodeExecutionData[] = [];

		for (let i = 0; i < items.length; i++) {
			try {
				const tsvFilePath = this.getNodeParameter('tsvFilePath', i, '') as string;
				const classifierServerUrl = this.getNodeParameter('classifierServerUrl', i, '') as string;

				if (!classifierServerUrl) {
					throw new ApplicationError('Classifier Server URL is required');
				}

				// Get input WorkResponse[] from previous node (e.g., OpenAlex)
				let inputWorks: WorkResponse[] = [];
				if (items[i].binary?.data) {
					const binaryBuffer = await this.helpers.getBinaryDataBuffer(i, 'data');
					inputWorks = JSON.parse(binaryBuffer.toString('utf8'));
				}

				// Track processed titles to avoid duplicates
				const processedTitles = new Set(inputWorks.map(work => work.title?.toLowerCase()).filter(Boolean));

				// Initialize output with input works
				const workResponses: WorkResponse[] = [...inputWorks];

				// Process TSV if file exists
				if (tsvFilePath && existsSync(tsvFilePath)) {
					const records = parseTSV(readFileSync(tsvFilePath, 'utf8'));

					for (const record of records) {
						const title = record.TI;
						const authors = record.AU || '';
						const journal = record.SO || '';
						const abstract = record.AB || '';

						// Skip if title is already processed
						if (title && processedTitles.has(title.toLowerCase())) {
							continue;
						}

						if (!title || !authors) {
							const article = { title: title || '(Missing Title)', authors: authors || '(Missing Authors)' };
							const postResponse = await sendHttpPost.call(this, article, journal, abstract, classifierServerUrl);
							workResponses.push(constructWorkResponse({ ...record, TI: title || '(Missing Title)', AU: authors || '(Missing Authors)' }, postResponse));
							continue;
						}

						const result = await searchOpenAlex(title, authors);
						if (result.found && result.work) {
							workResponses.push(result.work);
						} else {
							const article = { title, authors };
							const postResponse = await sendHttpPost.call(this, article, journal, abstract, classifierServerUrl);
							workResponses.push(constructWorkResponse(record, postResponse));
						}
					}
				}

				// Prepare binary output
				const fileName = `work-responses-${uuidv4()}.json`;
				const buffer = Buffer.from(JSON.stringify(workResponses, null, 2), 'utf8');

				returnData.push({
					binary: {
						data: await this.helpers.prepareBinaryData(buffer, fileName)
					},
					json: { success: true }
				});
			} catch (error) {
				if (this.continueOnFail()) {
					returnData.push({ json: { error: `Error: ${(error as Error).message}` } });
				} else {
					throw error;
				}
			}
		}

		return [returnData];
	}
}