import type {
  INodeType,
  INodeTypeDescription,
  INodeExecutionData,
  IExecuteFunctions,
} from 'n8n-workflow';
import { NodeConnectionType } from 'n8n-workflow';
import { OpenAlex as OpenAlexInstance, WorkResponse } from 'openalex-ts';
import { v4 as uuidv4 } from 'uuid';

const cleanText = (text: string | null | undefined): string => {
  if (!text || typeof text !== 'string') return '';
  return text
    .replace(/\\u003[CE]/g, '')
    .replace(/<[^>]+>/g, '')
    .replace(/\s+/g, ' ')
    .trim();
};

const isRelevant = (
  work: WorkResponse,
  relevantTerms: Record<string, string[]>,
  excludeTerms: string[],
): boolean => {
  const title = cleanText(work.title || '').toLowerCase();
  const topics = (work.topics || []).map((t: any) => cleanText(t.display_name || '').toLowerCase());
  const keywords = (work.keywords || []).map((k: any) => cleanText(k.display_name || '').toLowerCase());
	const abstract = cleanText(work.abstract || '').toLowerCase();
  const allText = new Set([title, ...topics, ...keywords, abstract]);

  let matchingCategories = 0;
  for (const terms of Object.values(relevantTerms)) {
    if (terms.some(term => Array.from(allText).some(text => text.includes(term.toLowerCase())))) {
      matchingCategories++;
    }
  }

  const isExcluded = excludeTerms.some(term =>
    Array.from(allText).some(text => text.includes(term.toLowerCase())),
  );

  return matchingCategories >= 2 && !isExcluded;
};

const fetchOpenAlexData = async (
  query: string,
  relevantTerms: Record<string, string[]>,
  excludeTerms: string[],
): Promise<WorkResponse[]> => {
  const openAlex = OpenAlexInstance.getInstance({
    openalexUrl: 'https://api.openalex.org',
    email: 'test-email@outlook.com',
    userAgent: 'openalex-ts/1.0.0',
    maxRetries: 3,
    retryBackoffFactor: 0.2,
    retryHttpCodes: [429, 500, 503],
  });

  const filters = { 'title_and_abstract.search': query, is_oa: true };
  const allWorks: WorkResponse[] = [];
  const paginator = openAlex.getWork()
    .filter(filters)
    .sort('relevance_score', 'desc')
    .paginate('cursor', 100, null);

  for await (const page of paginator) {
    allWorks.push(...page);
  }

  const relevantWorks = allWorks.filter(work => isRelevant(work, relevantTerms, excludeTerms));
  return relevantWorks;
};

export class OpenAlex implements INodeType {
  description: INodeTypeDescription = {
    displayName: 'OpenAlex Fetcher',
    name: 'openAlex',
    group: ['transform'],
    icon: { light: 'file:OpenAlex.svg', dark: 'file:OpenAlex.dark.svg' },
    version: 1,
    description: 'Fetches and filters data from OpenAlex using openalex-ts',
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
        default: '{"management": ["management", "business", "organization", "strategy", "leadership", "administration", "enterprise"], "decision_making": ["decision", "decisions", "decision-making", "choice", "judgment", "planning"], "visualization": ["visualization", "visualizations", "data visualization", "visual", "graph", "chart", "dashboard", "analytics"]}',
        description: 'JSON object where each key maps to an array of relevant terms',
      },
      {
        displayName: 'Exclude Terms (JSON)',
        name: 'excludeTerms',
        type: 'json',
        default: '["medicine", "medical", "surgery", "anesthesia", "fetal", "cancer", "disease", "biology", "clinical", "healthcare"]',
        description: 'JSON array of exclusion terms',
      },
    ],
  };

  async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
    const items = this.getInputData();
    const returnData: INodeExecutionData[] = [];

    for (let i = 0; i < items.length; i++) {
      try {
        const query = this.getNodeParameter('query', i, '') as string;

        const relevantTermsRaw = this.getNodeParameter('relevantTerms', i, {}) as string | Record<string, string[]>;
        const relevantTerms = typeof relevantTermsRaw === 'string' ? JSON.parse(relevantTermsRaw) : relevantTermsRaw;

        const excludeTermsRaw = this.getNodeParameter('excludeTerms', i, []) as string | string[];
        const excludeTerms = typeof excludeTermsRaw === 'string' ? JSON.parse(excludeTermsRaw) : excludeTermsRaw;

        const result = await fetchOpenAlexData(query, relevantTerms, excludeTerms);

        // Directly prepare binary without writing to disk
        const fileName = `openalex-result-${uuidv4()}.json`;
        const buffer = Buffer.from(JSON.stringify(result, null, 2), 'utf8');

        returnData.push({
          binary: {
            data: await this.helpers.prepareBinaryData(buffer, fileName),
          },
          json: { success: true }, // Small json payload to avoid empty json
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